"""Per-merge-method plugin abstraction.

The engine is a *streaming* merge engine: each source tensor (task vector
or source-model tensor) is streamed exactly once per pass, in place, with no
``torch.stack``. A :class:`MergeKernel` owns the per-element accumulators for
one merge method and exposes a uniform contract:

    kernel = build_kernel(base, settings)
    for entry in entries:
        delta = load_delta(...)          # one tensor at a time
        sparsify_delta(delta, settings)
        kernel.accumulate(delta, entry.weight)
    result = kernel.finish(entries, readers)   # -> base + mixed

Each kernel is a *dedicated* high-speed, memory-saving kernel for its method:
peak resident memory stays ~5-7 S (S = the largest tensor's bytes) because
only one delta is resident at a time.

``build_kernel`` dispatches on ``settings.merge_method``. The lazy imports
inside ``build_kernel`` avoid an import cycle with the kernel modules.
"""

from __future__ import annotations

from enum import Enum
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:  # pragma: no cover
    from .merge import MergeSettings


class MergeMethod(str, Enum):
    """The merge algorithm applied per tensor."""

    gta = "gta"          # Generalized Task Arithmetic (reference)
    linear = "linear"    # plain weighted sum  sum_i w_i * delta_i
    mixture = "mixture"  # weighted average  (sum_i w_i * delta_i) / sum_i w_i
    slerp = "slerp"      # spherical-linear interpolation between base and base+sum
    ties = "ties"        # sign-agreement masked sum / agreement count


class MergeKernel:
    """Streaming kernel for one merge method.

    Contract:
      - ``__init__(base, settings)``: allocate per-element accumulators in the
        base dtype (S bytes each).
      - ``accumulate(delta, weight)``: add one weighted delta in place.
      - ``finish(entries, readers, key) -> torch.Tensor``: return ``base +
        mixed``. ``entries`` / ``readers`` / ``key`` are passed so a kernel
        that needs a second streamed pass (the GTA weight-sum divisor) can
        re-read the deltas; single-pass kernels ignore them.

    No ``torch.stack``: at most one delta is resident at a time.
    """

    def __init__(self, base: torch.Tensor, settings: "MergeSettings") -> None:
        self.base = base
        self.settings = settings

    def accumulate(self, delta: torch.Tensor, weight: float) -> None:
        raise NotImplementedError

    def finish(self, entries, readers, key: str) -> torch.Tensor:
        raise NotImplementedError


def build_kernel(base: torch.Tensor, settings: "MergeSettings") -> "MergeKernel":
    """Instantiate the kernel for ``settings.merge_method``."""
    m = settings.merge_method
    if m == MergeMethod.gta:
        from .consensus import GtaKernel
        return GtaKernel(base, settings)
    if m == MergeMethod.linear:
        from .kernels import LinearKernel
        return LinearKernel(base, settings)
    if m == MergeMethod.mixture:
        from .kernels import MixtureKernel
        return MixtureKernel(base, settings)
    if m == MergeMethod.slerp:
        from .kernels import SlerpKernel
        return SlerpKernel(base, settings)
    if m == MergeMethod.ties:
        from .kernels import TiesKernel
        return TiesKernel(base, settings)
    raise ValueError(f"unknown merge method: {m}")
