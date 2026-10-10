"""Tests for the per-method merge kernels (Linear, Mixture, Slerp, Ties).

Each kernel is exercised against a small reference implementation that
reimplements the method's math directly (stack-based, fine for small
tensors). The kernels must produce the same result while streaming
one delta at a time.
"""

import torch

from goatmerge.kernels import LinearKernel, MixtureKernel, SlerpKernel, TiesKernel
from goatmerge.merge_method import MergeKernel


# --------------------------------------------------------------------------- #
# Reference implementations (stack-based, for small tensors)
# --------------------------------------------------------------------------- #

def _ref_linear(base: torch.Tensor, deltas: list[torch.Tensor], weights: list[float]) -> torch.Tensor:
    """mixed = sum_i w_i * delta_i;  result = base + mixed."""
    if not deltas:
        return base
    acc = torch.zeros_like(base)
    for d, w in zip(deltas, weights):
        acc += d * w
    return (base + acc).to(base.dtype)


def _ref_mixture(base: torch.Tensor, deltas: list[torch.Tensor], weights: list[float]) -> torch.Tensor:
    """mixed = (sum_i w_i * delta_i) / sum_i w_i;  result = base + mixed."""
    if not deltas:
        return base
    acc = torch.zeros_like(base)
    wsum = 0.0
    for d, w in zip(deltas, weights):
        acc += d * w
        wsum += w
    if abs(wsum) < 1e-8:
        wsum = 1.0
    mixed = acc / wsum
    return (base + mixed).to(base.dtype)


def _ref_slerp(base: torch.Tensor, deltas: list[torch.Tensor], weights: list[float]) -> torch.Tensor:
    """Canonical SLERP between base and base + sum(w_i * delta_i), t=1.0.

    Normalize both vectors, dot the *normalized* vectors, and interpolate
    with s0*v1 + s1*v2 (the mergekit formulation).
    """
    if not deltas:
        return base
    acc = torch.zeros_like(base)
    for d, w in zip(deltas, weights):
        acc += d * w
    t = 1.0
    v1 = base.float()
    v2 = base.float() + acc.float()
    n1 = v1.norm().clamp_min(1e-8)
    n2 = v2.norm().clamp_min(1e-8)
    u1 = v1 / n1
    u2 = v2 / n2
    dot = (u1 * u2).sum().clamp(-1.0, 1.0)
    theta = torch.acos(dot)
    if theta < 1e-8:
        return (base + acc).to(base.dtype)
    sin_theta = torch.sin(theta)
    s0 = torch.sin(theta - t * theta) / sin_theta
    s1 = torch.sin(t * theta) / sin_theta
    result = s0 * v1 + s1 * v2
    return result.to(base.dtype)


def _ref_ties(base: torch.Tensor, deltas: list[torch.Tensor], weights: list[float]) -> torch.Tensor:
    """TIES: sign-agreement masked sum / agreement count.

    Uses the same identity as the consensus accumulator:
      acc = sum(w_i * d_i),  l1 = sum(|w_i * d_i|),  c = sum(sign(w_i * d_i))
      M = +1 if c >= 0 else -1
      mixed = (acc + M * l1) / 2
      divisor = (nz + |c|) / 2  where nz = number of sources with a
                 non-zero effective sign (the agreement count;
                 divisor[divisor == 0] = 1)
      result = base + mixed / divisor
    """
    if not deltas:
        return base
    # Weighted deltas (bf16 * bf16 tensor multiply)
    weighted = [d * torch.tensor(w, dtype=base.dtype) for d, w in zip(deltas, weights)]
    # acc = sum of weighted deltas
    acc = sum(weighted, torch.zeros_like(base))
    # l1 = sum of |weighted deltas|
    l1 = sum([w.abs() for w in weighted], torch.zeros_like(base))
    # c = sum of sign(weighted deltas)
    c = sum([w.sign() for w in weighted], torch.zeros_like(base, dtype=torch.int8))
    # nz = number of sources with a non-zero effective sign
    nz = sum([(w.sign() != 0).to(base.dtype) for w in weighted], torch.zeros_like(base))
    # M = +1 if c >= 0 else -1
    M = torch.where(c >= 0, torch.ones_like(base), -torch.ones_like(base))
    # mixed = (acc + M * l1) / 2
    mixed = (acc + M * l1) / 2
    # divisor = (nz + |c|) / 2  (bf16 division, matching the kernel)
    divisor = (c.abs().to(base.dtype) + nz) / 2
    divisor[divisor == 0] = 1
    mixed = mixed / divisor
    return (base + mixed).to(base.dtype)


# --------------------------------------------------------------------------- #
# Test data generation
# --------------------------------------------------------------------------- #

def _make_tensors(n: int = 8, seed: int = 42) -> tuple[torch.Tensor, list[torch.Tensor], list[float]]:
    """Generate a base tensor and a list of deltas with random values."""
    torch.manual_seed(seed)
    base = torch.randn(n, n, dtype=torch.bfloat16)
    deltas = [torch.randn(n, n, dtype=torch.bfloat16) for _ in range(3)]
    weights = [0.7, 0.5, 0.3]
    return base, deltas, weights


# --------------------------------------------------------------------------- #
# Tests
# --------------------------------------------------------------------------- #

def test_linear_kernel_matches_reference():
    base, deltas, weights = _make_tensors()
    kernel = LinearKernel(base, None)
    for d, w in zip(deltas, weights):
        kernel.accumulate(d.clone(), w)
    result = kernel.finish([], {}, "test")
    expected = _ref_linear(base, deltas, weights)
    assert result.shape == expected.shape
    torch.testing.assert_close(result.float(), expected.float(), rtol=2e-2, atol=1e-2)


def test_mixture_kernel_matches_reference():
    base, deltas, weights = _make_tensors()
    kernel = MixtureKernel(base, None)
    for d, w in zip(deltas, weights):
        kernel.accumulate(d.clone(), w)
    result = kernel.finish([], {}, "test")
    expected = _ref_mixture(base, deltas, weights)
    assert result.shape == expected.shape
    torch.testing.assert_close(result.float(), expected.float(), rtol=2e-2, atol=1e-2)


def test_slerp_kernel_matches_reference():
    base, deltas, weights = _make_tensors()
    kernel = SlerpKernel(base, None)
    for d, w in zip(deltas, weights):
        kernel.accumulate(d.clone(), w)
    result = kernel.finish([], {}, "test")
    expected = _ref_slerp(base, deltas, weights)
    assert result.shape == expected.shape
    torch.testing.assert_close(result.float(), expected.float(), rtol=2e-2, atol=1e-2)


def test_slerp_t_one_returns_target():
    """t = 1.0 must return ``base + sum(w_i * delta_i)`` exactly."""
    base, deltas, weights = _make_tensors()
    kernel = SlerpKernel(base, None)
    for d, w in zip(deltas, weights):
        kernel.accumulate(d.clone(), w)
    result = kernel.finish([], {}, "test")
    target = base.clone()
    for d, w in zip(deltas, weights):
        target += d * w
    torch.testing.assert_close(result.float(), target.float(), rtol=2e-2, atol=1e-2)


def test_ties_kernel_matches_reference():
    base, deltas, weights = _make_tensors()
    kernel = TiesKernel(base, None)
    for d, w in zip(deltas, weights):
        kernel.accumulate(d.clone(), w)
    result = kernel.finish([], {}, "test")
    expected = _ref_ties(base, deltas, weights)
    assert result.shape == expected.shape
    torch.testing.assert_close(result.float(), expected.float(), rtol=2e-2, atol=1e-2)


def test_linear_single_delta():
    """With one delta and weight=1, result should be base + delta."""
    base, deltas, _ = _make_tensors()
    d = deltas[0]
    kernel = LinearKernel(base, None)
    kernel.accumulate(d.clone(), 1.0)
    result = kernel.finish([], {}, "test")
    expected = (base + d).to(base.dtype)
    torch.testing.assert_close(result.float(), expected.float(), rtol=2e-2, atol=1e-2)


def test_mixture_single_delta_weight_one():
    """With one delta and weight=1, mixture == linear (no normalization needed)."""
    base, deltas, _ = _make_tensors()
    d = deltas[0]
    kernel = MixtureKernel(base, None)
    kernel.accumulate(d.clone(), 1.0)
    result = kernel.finish([], {}, "test")
    expected = (base + d).to(base.dtype)
    torch.testing.assert_close(result.float(), expected.float(), rtol=2e-2, atol=1e-2)


def test_kernels_produce_correct_shape():
    """All kernels must produce a tensor with the same shape as base."""
    base, deltas, weights = _make_tensors()
    for cls in (LinearKernel, MixtureKernel, SlerpKernel, TiesKernel):
        kernel = cls(base, None)
        for d, w in zip(deltas, weights):
            kernel.accumulate(d.clone(), w)
        result = kernel.finish([], {}, "test")
        assert result.shape == base.shape, f"{cls.__name__} shape mismatch"


def test_linear_zero_weights():
    """All zero weights should give base unchanged."""
    base, deltas, _ = _make_tensors()
    kernel = LinearKernel(base, None)
    for d in deltas:
        kernel.accumulate(d.clone(), 0.0)
    result = kernel.finish([], {}, "test")
    torch.testing.assert_close(result.float(), base.float(), rtol=2e-2, atol=1e-2)


def test_mixture_zero_weights():
    """All zero weights: wsum=0 -> divisor=1, acc=0, so result=base."""
    base, deltas, _ = _make_tensors()
    kernel = MixtureKernel(base, None)
    for d in deltas:
        kernel.accumulate(d.clone(), 0.0)
    result = kernel.finish([], {}, "test")
    torch.testing.assert_close(result.float(), base.float(), rtol=2e-2, atol=1e-2)


def test_ties_all_same_sign():
    """When all deltas have the same sign per element, TIES should keep all."""
    base, deltas, _ = _make_tensors()
    # Make all deltas positive
    pos_deltas = [torch.abs(d) for d in deltas]
    weights = [0.7, 0.5, 0.3]
    # Use the general reference (handles the c-based divisor correctly)
    expected = _ref_ties(base, pos_deltas, weights)
    kernel = TiesKernel(base, None)
    for d, w in zip(pos_deltas, weights):
        kernel.accumulate(d.clone(), w)
    result = kernel.finish([], {}, "test")
    torch.testing.assert_close(result.float(), expected.float(), rtol=2e-2, atol=1e-2)


def test_ties_negative_weights():
    """Negative weights must not double-negate the sign count."""
    base, deltas, _ = _make_tensors(seed=21)
    weights = [0.7, -0.5, 0.3, -0.2]
    kernel = TiesKernel(base, None)
    for d, w in zip(deltas, weights):
        kernel.accumulate(d.clone(), w)
    result = kernel.finish([], {}, "test")
    expected = _ref_ties(base, deltas, weights)
    torch.testing.assert_close(result.float(), expected.float(), rtol=2e-2, atol=1e-2)


def test_ties_sparse_deltas_zero_signs():
    """Sparse deltas contain exact zeros: the divisor must be the agreement
    count (nz + |c|) / 2, not (k + |c|) / 2."""
    torch.manual_seed(9)
    base = torch.randn(32, 32, dtype=torch.bfloat16)
    deltas = [torch.randn(32, 32, dtype=torch.bfloat16) * (torch.rand(32, 32) > 0.7) for _ in range(3)]
    weights = [0.6, 0.4, 0.5]
    kernel = TiesKernel(base, None)
    for d, w in zip(deltas, weights):
        kernel.accumulate(d.clone(), w)
    result = kernel.finish([], {}, "test")
    expected = _ref_ties(base, deltas, weights)
    torch.testing.assert_close(result.float(), expected.float(), rtol=2e-2, atol=1e-2)
