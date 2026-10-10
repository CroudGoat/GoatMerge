"""Dedicated high-speed, memory-saving streaming kernels for non-GTA merge
methods.

Each kernel is a :class:`.merge_method.MergeKernel` that streams each
weighted delta exactly once, in place, with no ``torch.stack``. The weighted
product ``delta * weight`` is a bf16·bf16 tensor multiply (matching the GTA
reference ``stacked * weights``), so the numerics are consistent across
methods. Peak resident memory stays ~5-7 S (S = the largest tensor's bytes)
because only one delta is resident at a time.

Methods:
  - ``LinearKernel``:  ``mixed = sum_i w_i * delta_i``
  - ``MixtureKernel``:``mixed = (sum_i w_i * delta_i) / sum_i w_i``
  - ``SlerpKernel``:  spherical-linear interpolation between ``base`` and
    ``base + sum_i w_i * delta_i``.
  - ``TiesKernel``:   sign-agreement masked sum / agreement count
    (reuses the sign-consensus accumulators; single pass).
"""

from __future__ import annotations

from typing import Optional

import torch

from .consensus import ConsensusAccumulator, ConsensusMethod
from .merge_method import MergeKernel


def _weighted(delta: torch.Tensor, weight: float) -> None:
    """``delta = delta * weight`` in place (bf16·bf16 tensor multiply)."""
    w = torch.tensor(weight, dtype=delta.dtype)
    delta.mul_(w)


class LinearKernel(MergeKernel):
    """Plain weighted sum: ``mixed = sum_i w_i * delta_i`` (single pass)."""

    def __init__(self, base: torch.Tensor, settings) -> None:
        super().__init__(base, settings)
        self.acc = torch.zeros_like(base)

    def accumulate(self, delta: torch.Tensor, weight: float) -> None:
        _weighted(delta, weight)
        self.acc.add_(delta)

    def finish(self, entries, readers, key: str, start: Optional[int] = None, end: Optional[int] = None, chunk: int = 0, masker=None) -> torch.Tensor:
        return (self.base + self.acc).to(self.base.dtype)


class MixtureKernel(MergeKernel):
    """Weighted average: ``mixed = (sum_i w_i * delta_i) / sum_i w_i``."""

    def __init__(self, base: torch.Tensor, settings) -> None:
        super().__init__(base, settings)
        self.acc = torch.zeros_like(base)
        self.wsum = 0.0

    def accumulate(self, delta: torch.Tensor, weight: float) -> None:
        _weighted(delta, weight)
        self.acc.add_(delta)
        self.wsum += weight

    def finish(self, entries, readers, key: str, start: Optional[int] = None, end: Optional[int] = None, chunk: int = 0, masker=None) -> torch.Tensor:
        mixed = self.acc
        if abs(self.wsum) < 1e-8:
            self.wsum = 1.0
        mixed.div_(self.wsum)
        return (self.base + mixed).to(self.base.dtype)


class SlerpKernel(MergeKernel):
    """Spherical-linear interpolation between ``base`` and
    ``base + sum_i w_i * delta_i`` (single pass; the interpolation is one
    tensor-level operation in ``finish``).
    """

    def __init__(self, base: torch.Tensor, settings) -> None:
        super().__init__(base, settings)
        self.acc = torch.zeros_like(base)

    def accumulate(self, delta: torch.Tensor, weight: float) -> None:
        _weighted(delta, weight)
        self.acc.add_(delta)

    def finish(self, entries, readers, key: str, start: Optional[int] = None, end: Optional[int] = None, chunk: int = 0, masker=None) -> torch.Tensor:
        """Canonical SLERP between ``v1 = base`` and ``v2 = base + acc``.

        Follows the reference (mergekit ``slerp``): normalize both vectors, take
        the dot of the *normalized* vectors, and interpolate with

            s0 = sin(theta - t*theta) / sin(theta)
            s1 = sin(t*theta) / sin(theta)
            result = s0 * v1 + s1 * v2

        so ``t = 1`` returns ``v2`` exactly. Near-parallel vectors fall back to
        a linear interpolation. Only one full-size temporary is allocated
        (``v2``); the interpolation is applied to it in place.
        """
        t = 1.0
        v1 = self.base
        v2 = self.base + self.acc  # single O(S) temporary, then reused
        n1 = v1.norm().clamp_min(1e-8)
        n2 = v2.norm().clamp_min(1e-8)
        # cos(theta) of the normalized vectors, without materializing u1/u2
        dot = torch.dot(v1.reshape(-1), v2.reshape(-1)) / (n1 * n2)
        dot = dot.clamp(-1.0, 1.0)
        theta = torch.acos(dot)
        theta_f = float(theta)
        if theta_f < 1e-8 or float(torch.sin(theta)) < 1e-8:
            # (near-)identical directions: nothing to interpolate
            return v2.to(self.base.dtype)
        if float(dot) > 0.9995:
            # colinear: linear interpolation (stable where slerp is not)
            s0, s1 = 1.0 - t, t
        else:
            s0 = float(torch.sin(theta - t * theta)) / float(torch.sin(theta))
            s1 = float(torch.sin(t * theta)) / float(torch.sin(theta))
        v2.mul_(s1)
        v2.add_(v1, alpha=s0)
        return v2.to(self.base.dtype)


class TiesKernel(MergeKernel):
    """TIES: keep the elements whose weighted-delta signs agree with the
    majority, averaged over the agreeing sources.

    Reuses the sign-consensus accumulators (``acc`` / ``l1`` / ``c`` / ``nz``)
    but normalizes by the per-element *agreement count* instead of the weight
    sum, so it is a single pass (no second streamed pass). For ``nz`` sources
    with a non-zero effective sign and sign-sum ``c``, the number agreeing
    with the majority sign is ``(nz + |c|) / 2``.
    """

    def __init__(self, base: torch.Tensor, settings) -> None:
        super().__init__(base, settings)
        self._acc = ConsensusAccumulator(base, ConsensusMethod.count)

    def accumulate(self, delta: torch.Tensor, weight: float) -> None:
        self._acc.accumulate(delta, weight)

    def finish(self, entries, readers, key: str, start: Optional[int] = None, end: Optional[int] = None, chunk: int = 0, masker=None) -> torch.Tensor:
        acc = self._acc
        base = self.base
        mixed = acc.masked_sum_inplace()
        del acc.l1
        acc.l1 = None
        c = acc.c
        divisor = (c.abs().to(base.dtype) + acc.nz.to(base.dtype)) / 2
        divisor[divisor == 0] = 1
        mixed.div_(divisor)
        return (base + mixed).to(base.dtype)
