import torch

from goatmerge.sparsify import _bottom_threshold_from, _threshold_from


def _refs(t, k):
    n = t.numel()
    kc = max(0, min(n, k))
    flat = t.abs().float().reshape(-1)
    ref_top = 0.0 if kc == 0 else float(torch.kthvalue(flat, n - kc + 1).values)
    ref_bot = 0.0 if kc == 0 else float(torch.kthvalue(flat, kc).values)
    return ref_top, ref_bot


def test_select_threshold_exact_vs_sort():
    torch.manual_seed(0)
    cases = {
        "rand-f32": torch.randn(200_000),
        "rand-bf16": torch.randn(200_000, dtype=torch.bfloat16),
        "ties": torch.randint(0, 50, (100_000,)).float(),
        "all-equal": torch.full((10_000,), 3.25),
    }
    for t in cases.values():
        n = t.numel()
        for k in [0, 1, n // 3, n - 1, n, n + 5]:
            ref_top, ref_bot = _refs(t, k)
            assert _threshold_from(t, k) == ref_top
            assert _bottom_threshold_from(t, k) == ref_bot


def test_select_threshold_factory_matches_tensor():
    torch.manual_seed(1)
    t = torch.randn(300_000, dtype=torch.bfloat16)
    n = t.numel()
    chunks = [t[:100_000], t[100_000:200_000], t[200_000:]]
    for k in [1, n // 3, n]:
        fac = lambda: (c for c in chunks)
        assert _threshold_from(fac, k, n) == _threshold_from(t, k)
        assert _bottom_threshold_from(fac, k, n) == _bottom_threshold_from(t, k)
