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

    def finish(self, entries, readers, key: str) -> torch.Tensor:
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

    def finish(self, entries, readers, key: str) -> torch.Tensor:
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

    def finish(self, entries, readers, key: str) -> torch.Tensor:
        """Spherical-linear interpolation in the base dtype (bf16).

        Computes the SLERP entirely in bf16 to avoid allocating float32
        temporaries (4–5 full-size copies). The result is already in the
        base dtype, so no final cast is needed.
        """
        t = 1.0
        v1 = self.base
        v2 = self.base + self.acc
        n1 = v1.norm().clamp_min(1e-8)
        n2 = v2.norm().clamp_min(1e-8)
        u1 = v1 / n1
        u2 = v2 / n2
        cos_theta = (u1 * u2).sum() / (n1 * n2)
        cos_theta = cos_theta.clamp(-1.0, 1.0)
        theta = torch.acos(cos_theta)
        if theta < 1e-8:
            return (self.base + self.acc).to(self.base.dtype)
        s1 = torch.sin(t * theta)
        s2 = torch.sin((1.0 - t) * theta)
        direction = (u1 * s1 + u2 * s2) / torch.sin(theta)
        result = direction * n2
        return result.to(self.base.dtype)


class TiesKernel(MergeKernel):
    """TIES: keep the elements whose weighted-delta signs agree with the
    majority, averaged over the agreeing sources.

    Reuses the sign-consensus accumulators (``acc`` / ``l1`` / ``c``) but
    normalizes by the per-element *agreement count* ``(|c| + 1) / 2`` instead
    of the weight sum, so it is a single pass (no second streamed pass).
    """

    def __init__(self, base: torch.Tensor, settings) -> None:
        super().__init__(base, settings)
        self._acc = ConsensusAccumulator(base, ConsensusMethod.count)

    def accumulate(self, delta: torch.Tensor, weight: float) -> None:
        self._acc.accumulate(delta, weight)

    def finish(self, entries, readers, key: str) -> torch.Tensor:
        acc = self._acc
        base = self.base
        mixed = acc.masked_sum_inplace()
        del acc.l1
        c = acc.c
        divisor = (c.abs().to(base.dtype) + 1) / 2
        divisor[divisor == 0] = 1
        mixed.div_(divisor)
        return (base + mixed).to(base.dtype)
