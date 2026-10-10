import torch

from goatmerge.sparsify import (
    RescaleNorm,
    SparsificationMethod,
    bs_mask,
    bs_mask_chunked,
    magnitude_mask,
    magnitude_mask_chunked,
    magnitude_outliers_mask,
    magnitude_outliers_mask_chunked,
    sparsify_inplace,
)


def test_magnitude_mask_keeps_top_k():
    torch.manual_seed(0)
    t = torch.randn(1000)
    d = 0.3
    k = int(round(d * t.numel()))
    mask = magnitude_mask(t, d)
    assert int(mask.sum()) == k
    w = t.abs().reshape(-1)
    top = set(torch.topk(w, k).indices.tolist())
    kept = set(torch.nonzero(mask.reshape(-1)).flatten().tolist())
    assert kept == top


def test_magnitude_mask_ties_deterministic():
    t = torch.tensor([1.0, 1.0, -1.0, 0.0, 0.5, 0.5, 0.5, 0.5])
    mask = magnitude_mask(t, 0.5)  # k = 4
    flat = mask.reshape(-1)
    assert int(flat.sum()) == 4
    # the three +-1.0 magnitudes are always kept; among the tied 0.5s
    # exactly one is kept (topk tie order is implementation-defined)
    assert flat.tolist()[:3] == [1, 1, 1]
    assert int(flat[3:].sum()) == 1


def test_magnitude_outliers_middle():
    torch.manual_seed(1)
    t = torch.randn(1000)
    d, g = 0.5, 0.1
    mask = magnitude_outliers_mask(t, d, g)
    w = t.abs().reshape(-1)
    n = w.numel()
    target = int(d * n)
    n_top = int(g * n)
    n_bot = n - target - n_top
    idx = torch.sort(w, descending=False).indices
    ref = torch.zeros(n, dtype=torch.uint8)
    ref[idx[n_bot:-n_top]] = 1
    assert (mask.reshape(-1) == ref).all()


def test_bs_mask_blocks():
    torch.manual_seed(2)
    t = torch.randn(8, 16)  # 8 blocks of 16
    n, m = 4, 16
    mask = bs_mask(t, n, m)
    flat = t.reshape(-1)
    for b in range(8):
        block = flat[b * m : (b + 1) * m]
        top = torch.topk(block.abs(), n).indices
        expected = torch.zeros(m, dtype=torch.uint8)
        expected[top] = 1
        assert (mask.reshape(-1)[b * m : (b + 1) * m] == expected).all()


def test_bs_mask_trailing_zeroed():
    t = torch.randn(20)  # one full block of 16 + 4 trailing
    mask = bs_mask(t, 2, 16)
    flat = mask.reshape(-1)
    assert int(flat[16:].sum()) == 0
    assert int(flat[:16].sum()) == 2


def test_chunked_magnitude_matches_global():
    torch.manual_seed(3)
    t = torch.randn(1_000_000)
    d = 0.02
    m1 = magnitude_mask(t, d)
    m2 = magnitude_mask_chunked(t, d, 100_000)
    assert (m1 == m2).all()


def test_chunked_bs_matches():
    torch.manual_seed(4)
    t = torch.randn(100_000)
    m1 = bs_mask(t, 64, 256)
    m2 = bs_mask_chunked(t, 64, 256, 10_000)
    assert (m1 == m2).all()


def test_sparsify_inplace_rescale_l1():
    torch.manual_seed(5)
    t = torch.randn(1000)
    t2 = t.clone()
    sparsify_inplace(
        t2, 0.5, SparsificationMethod.magnitude, rescale_norm=RescaleNorm.l1
    )
    before = t.abs().sum().item()
    after = t2.abs().sum().item()
    assert abs(before - after) < 1e-4 * before


def test_sparsify_inplace_density_ge_1_noop():
    t = torch.randn(100)
    t2 = t.clone()
    sparsify_inplace(t2, 1.0, SparsificationMethod.magnitude)
    assert torch.equal(t, t2)


def test_sparsify_inplace_zero_density():
    t = torch.randn(100)
    t2 = t.clone()
    sparsify_inplace(t2, 0.0, SparsificationMethod.magnitude)
    assert torch.equal(t2, torch.zeros_like(t))


def test_chunked_masks_match_on_tie_heavy_tensors():
    """Ties must be filled from *global* strict counts: the chunked masks keep
    exactly k / density*n elements even when many |values| are equal."""
    torch.manual_seed(11)
    t = torch.randint(0, 3, (4000,)).float()  # only 3 distinct magnitudes
    for density in (0.25, 0.5, 0.9):
        expected_n = int(density * t.numel())
        got = magnitude_mask_chunked(t, density, 700)
        assert int(got.sum()) == expected_n, (density, int(got.sum()), expected_n)
        assert int(magnitude_mask(t, density).sum()) == expected_n
    for density, gamma in ((0.5, 0.1), (0.3, 0.2), (0.25, 0.0)):
        expected_n = int(density * t.numel())
        got = magnitude_outliers_mask_chunked(t, density, gamma, 700)
        assert int(got.sum()) == expected_n, (density, gamma, int(got.sum()), expected_n)


def test_chunked_masks_match_continuous_tensors():
    torch.manual_seed(12)
    t = torch.randn(4096)
    assert torch.equal(magnitude_mask(t, 0.3), magnitude_mask_chunked(t, 0.3, 512))
    assert torch.equal(
        magnitude_outliers_mask(t, 0.5, 0.1),
        magnitude_outliers_mask_chunked(t, 0.5, 0.1, 512),
    )
    assert torch.equal(bs_mask(t, 8, 32), bs_mask_chunked(t, 8, 32, 512))


def test_k_uses_truncation_like_mergekit():
    """mergekit uses int(density * n) (truncation), not round()."""
    torch.manual_seed(13)
    t = torch.randn(55)
    assert int(magnitude_mask(t, 0.1).sum()) == int(0.1 * 55)
