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

so the masked sum needs no second pass over the deltas; only ``divisor``
(per-element weight sum over sign-matching TVs) requires a second pass.
"""

from __future__ import annotations

from enum import Enum
from typing import Optional

import torch

from .io import load_delta, load_delta_chunk
from .merge_method import MergeKernel
from .sparsify import merge_generator, sparsify_delta


class ConsensusMethod(str, Enum):
    none = "none"
    sum = "sum"
    count = "count"


class ConsensusAccumulator:
    """Streams deltas into per-element sign-consensus accumulators.

    All accumulators live in the base dtype (``c`` is int8), so the
    bookkeeping costs S bytes each instead of k*S. ``l1`` is only allocated
    when a consensus mask is actually needed (``method != none``).
    """

    def __init__(self, base: torch.Tensor, method: ConsensusMethod) -> None:
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

    def mask_for(self, delta: torch.Tensor, alpha: float) -> torch.Tensor:
        """uint8 mask: 1 where sign(alpha*delta) == majority, else 0.

        Requires ``masked_sum_inplace`` to have run (it stores the
        majority sign in ``self._majority``).
        """
        s = delta.sign().to(torch.int8)
        if alpha < 0:
            s.neg_()
        return (s == self._majority).to(torch.uint8)

    def divisor_accumulate(self, divisor: torch.Tensor, delta: torch.Tensor, alpha: float) -> None:
        """divisor += alpha * mask (in place)."""
        divisor.add_(self.mask_for(delta, alpha), alpha=alpha)


class GtaKernel(MergeKernel):
    """Generalized Task Arithmetic kernel (the reference method).

    Streams each weighted delta once per pass. The no-consensus path is a single
    pass; the consensus path needs a second streamed pass for the per-element
    weight-sum divisor (re-reads the deltas via ``readers`` — only the row
    range ``[start, end)`` when merging in chunked mode, so peak RAM stays
    O(chunk) instead of one full delta).
    """

    def __init__(self, base: torch.Tensor, settings) -> None:
        super().__init__(base, settings)
        self._acc = ConsensusAccumulator(base, settings.consensus)

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
        divisor = torch.zeros_like(base)
        if masker is not None:
            masker.begin_pass(chunk)
        for i, entry in enumerate(entries):
            if start is not None:
                delta = load_delta_chunk(base, entry.dir, entry.kind, key, readers, start, end)
            else:
                delta = load_delta(base, entry.dir, entry.kind, key, readers)
            if delta is None:
                continue
            if masker is None or not masker.apply(i, delta):
                # same mask as the accumulation pass: same per-chunk
                # generator seed, or the same global mask parameters
                sparsify_delta(
                    delta, s.method,
                    density=s.density, n=s.n, m=s.m, gamma=s.gamma,
                    epsilon=s.epsilon, rescale=s.rescale,
                    chunk_elements=s.chunk_elements,
                    generator=merge_generator(key, i, chunk),
                )
            acc.divisor_accumulate(divisor, delta, entry.weight)
            del delta
        divisor[divisor == 0] = 1
        if s.normalize:
            mixed.div_(divisor)
        if s.lambda_ != 1.0:
            mixed.mul_(s.lambda_)
        return (base + mixed).to(base.dtype)
