"""Benchmark: GoatMerge per-method kernels vs mergekit reference.

Creates a 50MB bf16 base tensor + 3 delta tensors, runs each merge method
(Linear, Mixture, Slerp, Ties) through both GoatMerge's streaming kernels
and mergekit's stack-based reference, and reports time + peak RAM.

Peak RAM is reported as the theoretical resident tensor bytes (the
tracemalloc API does not capture torch tensor allocations on CPU).
"""

import time

import torch

from goatmerge.kernels import LinearKernel, MixtureKernel, SlerpKernel, TiesKernel

# --------------------------------------------------------------------------- #
# Reference implementations (stack-based, matching mergekit semantics)
# --------------------------------------------------------------------------- #

def ref_linear(base: torch.Tensor, deltas: list[torch.Tensor], weights: list[float]) -> torch.Tensor:
    stacked = torch.stack(deltas, dim=0)
    w = torch.tensor(weights, dtype=stacked.dtype, device=stacked.device)
    while stacked.dim() > w.dim():
        w = w.unsqueeze(-1)
    weighted = stacked * w
    mixed = weighted.sum(0)
    return (base + mixed).to(base.dtype)


def ref_mixture(base: torch.Tensor, deltas: list[torch.Tensor], weights: list[float]) -> torch.Tensor:
    stacked = torch.stack(deltas, dim=0)
    w = torch.tensor(weights, dtype=stacked.dtype, device=stacked.device)
    while stacked.dim() > w.dim():
        w = w.unsqueeze(-1)
    weighted = stacked * w
    mixed = weighted.sum(0)
    wsum = sum(weights)
    if abs(wsum) < 1e-8:
        wsum = 1.0
    mixed = mixed / wsum
    return (base + mixed).to(base.dtype)


def ref_slerp(base: torch.Tensor, deltas: list[torch.Tensor], weights: list[float]) -> torch.Tensor:
    stacked = torch.stack(deltas, dim=0)
    w = torch.tensor(weights, dtype=stacked.dtype, device=stacked.device)
    while stacked.dim() > w.dim():
        w = w.unsqueeze(-1)
    acc = (stacked * w).sum(0)
    t = 1.0
    v1 = base.float()
    v2 = (base.float() + acc.float())
    n1 = v1.norm().clamp_min(1e-8)
    n2 = v2.norm().clamp_min(1e-8)
    u1 = v1 / n1
    u2 = v2 / n2
    cos_theta = (u1 * u2).sum() / (n1 * n2)
    cos_theta = cos_theta.clamp(-1.0, 1.0)
    theta = torch.acos(cos_theta)
    if theta < 1e-8:
        return (base + acc).to(base.dtype)
    s1 = torch.sin(t * theta)
    s2 = torch.sin((1.0 - t) * theta)
    direction = (u1 * s1 + u2 * s2) / torch.sin(theta)
    result = direction * n2
    return result.to(base.dtype)


def ref_ties(base: torch.Tensor, deltas: list[torch.Tensor], weights: list[float]) -> torch.Tensor:
    stacked = torch.stack(deltas, dim=0)
    w = torch.tensor(weights, dtype=stacked.dtype, device=stacked.device)
    while stacked.dim() > w.dim():
        w = w.unsqueeze(-1)
    weighted = stacked * w
    signs = [torch.sign(wi) for wi in weighted]
    sign_sum = sum(signs, torch.zeros_like(base, dtype=torch.int8))
    M = torch.where(sign_sum >= 0, torch.ones_like(base), -torch.ones_like(base))
    acc = sum([wi for wi in weighted], torch.zeros_like(base))
    l1 = sum([wi.abs() for wi in weighted], torch.zeros_like(base))
    mixed = (acc + M * l1) / 2
    c = sign_sum
    divisor = (c.abs().to(base.dtype) + 1) / 2
    divisor[divisor == 0] = 1
    mixed = mixed / divisor
    return (base + mixed).to(base.dtype)


# --------------------------------------------------------------------------- #
# Benchmark harness
# --------------------------------------------------------------------------- #

def bench_goat(kernel_cls, base, deltas, weights, n_warmup=1):
    """Run a GoatMerge kernel and measure time."""
    # Warm up
    for _ in range(n_warmup):
        k = kernel_cls(base, None)
        for d, w in zip(deltas, weights):
            k.accumulate(d.clone(), w)
        k.finish([], {}, "test")

    # Timed run
    t0 = time.perf_counter()
    k = kernel_cls(base, None)
    for d, w in zip(deltas, weights):
        k.accumulate(d.clone(), w)
    result = k.finish([], {}, "test")
    t1 = time.perf_counter()
    return (t1 - t0) * 1000, result


def bench_ref(ref_fn, base, deltas, weights, n_warmup=1):
    """Run a reference (stack-based) merge and measure time."""
    # Warm up
    for _ in range(n_warmup):
        ref_fn(base, [d.clone() for d in deltas], weights)

    # Timed run
    t0 = time.perf_counter()
    result = ref_fn(base, [d.clone() for d in deltas], weights)
    t1 = time.perf_counter()
    return (t1 - t0) * 1000, result


def main():
    # 50MB bf16 tensor: 50_000_000 / 2 = 25_000_000 elements
    # Shape: (5000, 5000) = 25_000_000 elements * 2 bytes = 50MB
    n = 5000
    torch.manual_seed(42)
    base = torch.randn(n, n, dtype=torch.bfloat16)
    deltas = [torch.randn(n, n, dtype=torch.bfloat16) for _ in range(3)]
    weights = [0.7, 0.5, 0.3]

    S = base.numel() * base.element_size()  # bytes per tensor
    k = len(deltas)

    print(f"Tensor size: {S / 1e6:.1f} MB (bf16, {n}x{n}), {k} deltas")
    print()

    # Theoretical peak RAM:
    #   Goat (streaming): base + 1 delta + acc = 3S (plus small bookkeeping)
    #   Ref  (stacked):   base + k deltas + stacked + weighted = (k+2)S
    goat_ram_mb = 3 * S / 1e6
    ref_ram_mb = (k + 2) * S / 1e6

    results = {}

    # --- Linear ---
    t_g, _ = bench_goat(LinearKernel, base, deltas, weights)
    t_r, _ = bench_ref(ref_linear, base, deltas, weights)
    results["linear"] = (t_g, t_r)

    # --- Mixture ---
    t_g, _ = bench_goat(MixtureKernel, base, deltas, weights)
    t_r, _ = bench_ref(ref_mixture, base, deltas, weights)
    results["mixture"] = (t_g, t_r)

    # --- Slerp ---
    t_g, _ = bench_goat(SlerpKernel, base, deltas, weights)
    t_r, _ = bench_ref(ref_slerp, base, deltas, weights)
    results["slerp"] = (t_g, t_r)

    # --- Ties ---
    t_g, _ = bench_goat(TiesKernel, base, deltas, weights)
    t_r, _ = bench_ref(ref_ties, base, deltas, weights)
    results["ties"] = (t_g, t_r)

    # --- Report ---
    print(f"{'Method':<10} {'Goat (ms)':>10} {'Ref (ms)':>10} {'Speedup':>8} {'Goat RAM (MB)':>14} {'Ref RAM (MB)':>12} {'RAM ratio':>10}")
    print("-" * 78)
    for name, (tg, tr) in results.items():
        speedup = tr / tg if tg > 0 else float("inf")
        print(f"{name:<10} {tg:>10.1f} {tr:>10.1f} {speedup:>8.2f}x {goat_ram_mb:>14.1f} {ref_ram_mb:>12.1f} {ref_ram_mb / goat_ram_mb:>10.2f}x")

    print()
    print(f"Theoretical peak RAM:")
    print(f"  Goat (streaming): 3S = {goat_ram_mb:.1f} MB  (base + 1 delta + acc)")
    print(f"  Ref  (stacked):   (k+2)S = {ref_ram_mb:.1f} MB  (base + {k} deltas + stacked + weighted)")
    print(f"  RAM savings: {ref_ram_mb / goat_ram_mb:.2f}x")


if __name__ == "__main__":
    main()
