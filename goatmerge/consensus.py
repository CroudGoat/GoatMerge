"""Single-pass sign-consensus accumulators for the streaming merge.

Reference semantics (mergekit GTA, per tensor):

    weighted[i]   = alpha_i * delta_i
    sign[i]      = weighted[i].sign()
    majority     = (weighted.sum(0) >= 0)          [sum method]
                 (sign.sum(0)   >= 0)             [count method]
    mask[i]      = sign[i] == majority
    mixed       = (weighted * mask).sum(0)
    divisor     = (weights * mask).sum(0);  divisor[divisor == 0] = 1
    result      = (base + mixed).to(base.dtype)

GoatMerge streams each delta exactly once per pass and accumulates, per
tensor:

    acc  = sum_i alpha_i * delta_i            (base dtype)
    l1   = sum_i |alpha_i * delta_i|         (base dtype)
    c    = sum_i sign(alpha_i * delta_i)     (int8, count method only)

Exact identity (zeros contribute 0 in both forms, matching the reference
mask which drops sign-0 elements):

    mixed = (acc + M * l1) / 2,   M = +1 if acc >= 0 else -1   [sum]
    mixed = (acc + M * l1) / 2,   M = +1 if c   >= 0 else -1   [count]

so the masked sum needs no second pass over the deltas. The divisor
(per-element weight sum over sign-matching TVs) is built from 2-bit-packed
signs cached during accumulation — again with no re-read — so each tensor
is read from disk exactly once per merge.
"""

from __future__ import annotations

from enum import Enum
from typing import List, Optional

import torch

from .merge_method import MergeKernel


class ConsensusMethod(str, Enum):
    none = "none"
    sum = "sum"
    count = "count"


def _pack_signs(delta: torch.Tensor) -> torch.Tensor:
    """Pack ``delta.sign()`` into 2 bits per element (uint8, LSB-first).

    Encoding 0b00 = zero, 0b01 = +1, 0b10 = -1: a zero entry never equals a
    majority sign (+1/-1), mirroring the reference mask which drops sign-0
    elements. The trailing byte is zero-padded (also never matching).
    Comparison-based (no ``sign()`` temporary) and shape-agnostic.
    """
    d = delta.reshape(-1)
    lo = (d > 0).to(torch.uint8)
    lo |= ((d < 0).to(torch.uint8) << 1)
    pad = (-lo.numel()) % 4
    if pad:
        lo = torch.cat([lo, torch.zeros(pad, dtype=torch.uint8)])
    code = lo.reshape(-1, 4)
    return (code[:, 0] | (code[:, 1] << 2) | (code[:, 2] << 4) | (code[:, 3] << 6)).contiguous()


def _unpack_match(packed: torch.Tensor, shape, majority: torch.Tensor) -> torch.Tensor:
    """uint8 mask: 1 where the cached 2-bit sign equals ``majority`` (+1/-1).

    Inverse of :func:`_pack_signs`. Zero-sign entries (0b00, including
    padding) never match, exactly like the reference mask.
    """
    n = 1
    for s in shape:
        n *= s
    b = packed.reshape(-1, 1)
    shifts = torch.tensor([0, 2, 4, 6], dtype=torch.uint8, device=packed.device)
    codes = ((b >> shifts) & 3).reshape(-1)[:n].reshape(shape)
    plus = majority > 0
    return torch.where(plus, codes == 0b01, codes == 0b10).to(torch.uint8)


class ConsensusAccumulator:
    """Streams deltas into per-element sign-consensus accumulators.

    All accumulators live in the base dtype (``c`` is int8), so the
    bookkeeping costs S bytes each instead of k*S. ``l1`` is only allocated
    when a consensus mask is actually needed (``method != none``). With
    ``store_signs``, each contributing delta's sign is additionally cached
    2-bit-packed (k*S/4 transient bytes), so the consensus divisor can be
    built without re-reading the deltas.
    """

    def __init__(self, base: torch.Tensor, method: ConsensusMethod, store_signs: bool = False) -> None:
        self.method = method
        self.acc = torch.zeros_like(base)          # sum_i alpha_i * delta_i
        self.l1: Optional[torch.Tensor] = None     # sum_i |alpha_i * delta_i|
        if method != ConsensusMethod.none:
            self.l1 = torch.zeros_like(base)
        self.c: Optional[torch.Tensor] = None      # int8 sign count
        self.nz: Optional[torch.Tensor] = None     # int8 count of non-zero signs
        self._majority: Optional[torch.Tensor] = None
        if method == ConsensusMethod.count:
            self.c = torch.zeros_like(base, dtype=torch.int8)
            # number of sources with a non-zero effective sign, per element.
            # The agreement count is (nz + |c|) / 2 — using the raw source
            # count k instead would over-count when (sparse) deltas contain
            # exact zeros, since those carry sign 0.
            self.nz = torch.zeros_like(base, dtype=torch.int8)
        # 2-bit-packed sign(alpha_i * delta_i) per contributing delta, with
        # its weight: the divisor is built from these (see divisor_from_signs)
        # instead of a second streamed pass. Only needed when a divisor will
        # actually be computed (consensus + normalize).
        self._store_signs = store_signs and method != ConsensusMethod.none
        self._sign_packets: List[torch.Tensor] = []
        self._sign_alphas: List[float] = []
        # Number of (present) deltas accumulated and the sum of their
        # weights. Mirrors mergekit, whose divisor / normalized weight sum
        # only covers the task vectors that actually contributed this tensor.
        self.n = 0
        self.wsum = 0.0

    def accumulate(self, delta: torch.Tensor, alpha: float) -> None:
        """Add one (weighted) delta in place. ``delta`` is sparsified.

        The weighted product ``delta * alpha`` is computed as a bf16·bf16
        tensor multiply (matching the reference ``stacked * weights``), NOT
        as a Python-float scalar ``add_`` — whose internal product precision
        diverges from the reference at near-tie elements and flips the
        per-element majority sign.
        """
        self.n += 1
        self.wsum += alpha
        w = torch.tensor(alpha, dtype=delta.dtype)
        delta.mul_(w)              # delta = delta * alpha (bf16 * bf16, in place)
        self.acc.add_(delta)
        if self.l1 is not None:
            self.l1.add_(delta.abs())
        if self.c is not None:
            # delta is already weighted (delta * alpha), so its sign *is*
            # sign(alpha * delta) — always add it (mergekit's count consensus
            # sums sign(w_i * d_i) over the weighted deltas).
            s = delta.sign().to(torch.int8)
            self.nz.add_(s.abs())
            self.c.add_(s)
        if self._store_signs:
            # Cache the weighted, sparsified signs for the divisor: delta is
            # final here (weighted above, sparsified by the caller before
            # accumulate), so no re-read and no re-sparsify is ever needed.
            self._sign_packets.append(_pack_signs(delta))
            self._sign_alphas.append(alpha)

    def majority(self) -> torch.Tensor:
        """Per-element majority sign as an int8 tensor of +1/-1.

        Must be called before ``masked_sum_inplace`` (which overwrites
        ``acc`` with the mixed sum).
        """
        cond = (self.acc if self.method == ConsensusMethod.sum else self.c) >= 0
        return (2 * cond.to(torch.int8)) - 1

    def masked_sum_inplace(self) -> torch.Tensor:
        """Compute ``mixed = (acc + M * l1) / 2`` in place into ``acc``.

        Returns the mixed tensor (``acc``). ``l1`` is consumed and may be
        freed by the caller.
        """
        m = self.majority()
        self._majority = m  # keep for the pass-2 masks
        self.l1.mul_(m)          # l1 = M * l1 (in place)
        self.acc.add_(self.l1)   # acc = acc + M * l1
        self.acc.mul_(0.5)       # acc = mixed
        return self.acc

    def divisor_from_signs(self, divisor: torch.Tensor) -> None:
        """divisor += alpha_i * [sign_i == majority], from cached signs.

        Requires ``masked_sum_inplace`` to have run (it stores the majority
        sign in ``self._majority``). Cached packets are freed. Zero-sign
        entries never match, mirroring the reference mask.
        """
        majority = self._majority
        shape = tuple(majority.shape)
        for packet, alpha in zip(self._sign_packets, self._sign_alphas):
            match = _unpack_match(packet, shape, majority)
            divisor.add_(match, alpha=alpha)
            del match
        self._sign_packets.clear()
        self._sign_alphas.clear()


class GtaKernel(MergeKernel):
    """Generalized Task Arithmetic kernel (the reference method).

    Streams each weighted delta exactly once. The no-consensus path is a
    single pass; the consensus path builds the per-element weight-sum
    divisor from cached signs (no second streamed pass, no re-reads).
    """

    def __init__(self, base: torch.Tensor, settings) -> None:
        super().__init__(base, settings)
        s = settings
        self._acc = ConsensusAccumulator(
            base, s.consensus,
            # Signs are only needed for the divisor, which is only computed
            # under consensus + normalize. Otherwise each delta is touched
            # once and forgotten.
            store_signs=(s.consensus != ConsensusMethod.none and s.normalize),
        )

    def accumulate(self, delta: torch.Tensor, weight: float) -> None:
        self._acc.accumulate(delta, weight)

    def finish(self, entries, readers, key: str, start: Optional[int] = None, end: Optional[int] = None, chunk: int = 0, masker=None) -> torch.Tensor:
        s = self.settings
        acc = self._acc
        base = self.base
        if s.consensus == ConsensusMethod.none:
            mixed = acc.acc
            if s.normalize:
                # mergekit divides by the weight sum of the TVs that actually
                # contributed this tensor (missing / skipped ones excluded).
                wsum = acc.wsum
                if abs(wsum) < 1e-8:
                    wsum = 1.0
                mixed.div_(wsum)
            if s.lambda_ != 1.0:
                mixed.mul_(s.lambda_)
            return (base + mixed).to(base.dtype)
        # consensus: mixed = (acc + M*l1)/2 in place
        mixed = acc.masked_sum_inplace()
        del acc.l1
        acc.l1 = None
        if s.normalize:
            # The divisor comes from the signs cached during accumulation —
            # no second streamed pass over the deltas.
            divisor = torch.zeros_like(base)
            acc.divisor_from_signs(divisor)
            divisor[divisor == 0] = 1
            mixed.div_(divisor)
        if s.lambda_ != 1.0:
            mixed.mul_(s.lambda_)
        return (base + mixed).to(base.dtype)
