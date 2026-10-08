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

from .io import load_delta
from .merge_method import MergeKernel
from .sparsify import sparsify_delta


class ConsensusMethod(str, Enum):
    none = "none"
    sum = "sum"
    count = "count"


class ConsensusAccumulator:
    """Streams deltas into per-element sign-consensus accumulators.

    All accumulators live in the base dtype (``c`` is int8), so the
    bookkeeping costs S bytes each instead of k*S.
    """

    def __init__(self, base: torch.Tensor, method: ConsensusMethod) -> None:
        self.method = method
        self.acc = torch.zeros_like(base)          # sum_i alpha_i * delta_i
        self.l1 = torch.zeros_like(base)           # sum_i |alpha_i * delta_i|
        self.c: Optional[torch.Tensor] = None      # int8 sign count
        self._majority: Optional[torch.Tensor] = None
        if method == ConsensusMethod.count:
            self.c = torch.zeros_like(base, dtype=torch.int8)

    def accumulate(self, delta: torch.Tensor, alpha: float) -> None:
        """Add one (weighted) delta in place. ``delta`` is sparsified.

        The weighted product ``delta * alpha`` is computed as a bf16·bf16
        tensor multiply (matching the reference ``stacked * weights``), NOT
        as a Python-float scalar ``add_`` — whose internal product precision
        diverges from the reference at near-tie elements and flips the
        per-element majority sign.
        """
        w = torch.tensor(alpha, dtype=delta.dtype)
        delta.mul_(w)              # delta = delta * alpha (bf16 * bf16, in place)
        self.acc.add_(delta)
        self.l1.add_(delta.abs())
        if self.c is not None:
            s = delta.sign().to(torch.int8)
            if alpha < 0:
                self.c.sub_(s)
            else:
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
    weight-sum divisor (re-reads the deltas via ``readers``). Peak RAM stays
    ~5-7 S because only one delta is resident at a time.
    """

    def __init__(self, base: torch.Tensor, settings) -> None:
        super().__init__(base, settings)
        self._acc = ConsensusAccumulator(base, settings.consensus)

    def accumulate(self, delta: torch.Tensor, weight: float) -> None:
        self._acc.accumulate(delta, weight)

    def finish(self, entries, readers, key: str) -> torch.Tensor:
        s = self.settings
        acc = self._acc
        base = self.base
        if s.consensus == ConsensusMethod.none:
            mixed = acc.acc
            if s.normalize:
                wsum = sum(e.weight for e in entries)
                if abs(wsum) < 1e-8:
                    wsum = 1.0
                mixed.div_(wsum)
            if s.lambda_ != 1.0:
                mixed.mul_(s.lambda_)
            return (base + mixed).to(base.dtype)
        # consensus: mixed = (acc + M*l1)/2 in place
        mixed = acc.masked_sum_inplace()
        del acc.l1
        divisor = torch.zeros_like(base)
        for entry in entries:
            delta = load_delta(base, entry.dir, entry.kind, key, readers)
            if delta is None:
                continue
            sparsify_delta(
                delta, s.method,
                density=s.density, n=s.n, m=s.m, gamma=s.gamma,
                epsilon=s.epsilon, rescale=s.rescale,
                chunk_elements=s.chunk_elements,
            )
            acc.divisor_accumulate(divisor, delta, entry.weight)
            del delta
        divisor[divisor == 0] = 1
        if s.normalize:
            mixed.div_(divisor)
        if s.lambda_ != 1.0:
            mixed.mul_(s.lambda_)
        return (base + mixed).to(base.dtype)
