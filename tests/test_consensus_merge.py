"""Parity of GoatMerge's streaming engine against the mergekit GTA kernel.

The reference ``_ref_gta`` reimplements ``mergekit.merge_methods.
generalized_task_arithmetic.GTATask.execute`` exactly (stack-based,
fine for small tensors):

    weighted  = stack(deltas) * weights
    consensus: sign = weighted.sign(); majority per element (sum or count);
               mask = sign == majority
               mixed = (weighted * mask).sum(0)
               divisor = (weights * mask).sum(0); divisor[divisor==0] = 1
    else:      mixed = weighted.sum(0); divisor = weights.sum()
    normalize: mixed /= divisor;  lambda scaling;  (base + mixed).to(base.dtype)

GoatMerge must produce the same result while never stacking all deltas.
"""

import os

import safetensors.torch
import torch

from goatmerge.consensus import ConsensusMethod
from goatmerge.extract import extract_task_vector
from goatmerge.io import ShardReader, ShardedTensorIndex
from goatmerge.merge import ModelEntry, MergeSettings, merge_tensor
from goatmerge.cli import main as cli_main
from goatmerge.fingerprint import compute_base_fingerprint
from goatmerge.inspect import verify_against_base
from goatmerge.metadata import load_metadata, validate_metadata
from goatmerge.sparsify import RescaleNorm, SparsificationMethod, sparsify_inplace


# --------------------------------------------------------------------------- #
# Reference kernel (mergekit GTA semantics, stack-based)
# --------------------------------------------------------------------------- #
def _ref_gta(
    base: torch.Tensor,
    deltas: list[torch.Tensor],
    alphas: list[float],
    *,
    lam: float = 1.0,
    normalize: bool = True,
    consensus: str | None = None,
    sparsify_fn=None,
) -> torch.Tensor:
    if not deltas:
        return base
    if sparsify_fn is not None:
        deltas = [sparsify_fn(d.clone()) for d in deltas]
    stacked = torch.stack(deltas, dim=0)
    weights = torch.tensor(alphas, dtype=stacked.dtype, device=stacked.device)
    while stacked.dim() > weights.dim():
        weights = weights.unsqueeze(-1)
    weighted = stacked * weights

    if consensus in ("sum", "count"):
        sign = weighted.sign()
        # mergekit get_mask: majority sign is +/-1, mask = sign == majority
        if consensus == "sum":
            majority = ((weighted.sum(0) >= 0).to(torch.int8) * 2 - 1)
        else:
            majority = ((sign.sum(0) >= 0).to(torch.int8) * 2 - 1)
        mask = (sign == majority.unsqueeze(0))
        mixed = (weighted * mask).sum(0)
        divisor = (weights * mask).sum(0)
        divisor[divisor == 0] = 1
    else:
        mixed = weighted.sum(0)
        wsum = weights.sum()
        if wsum.abs() < 1e-8:
            wsum = torch.tensor(1.0, dtype=stacked.dtype, device=stacked.device)
        divisor = wsum

    if normalize:
        mixed = mixed / divisor
    if lam != 1.0:
        mixed = mixed * lam
    return (base + mixed).to(base.dtype)


def _write_model_dir(path: str, tensors: dict[str, torch.Tensor]) -> None:
    os.makedirs(path, exist_ok=True)
    safetensors.torch.save_file(tensors, os.path.join(path, "model.safetensors"))


def _load_model_tensors(path: str) -> dict[str, torch.Tensor]:
    out = {}
    with safetensors.torch.safe_open(os.path.join(path, "model.safetensors"), framework="pt") as st:
        for k in st.keys():
            out[k] = st.get_tensor(k)
    return out


def _goat_engine(base_dir: str, tv_dirs: list[str], weights: list[float], settings: MergeSettings) -> dict[str, torch.Tensor]:
    """Run GoatMerge's streaming engine over a base dir + TV dirs."""
    base_index = ShardedTensorIndex.from_dir(base_dir)
    readers = {base_dir: ShardReader(base_index)}
    for d in tv_dirs:
        readers[d] = ShardReader(ShardedTensorIndex.from_dir(d))
    entries = [ModelEntry(dir=d, kind="tv", weight=w) for d, w in zip(tv_dirs, weights)]
    results = {}
    try:
        br = readers[base_dir]
        for key in base_index.keys():
            b = br.get_tensor(key).clone()
            results[key] = merge_tensor(b, entries, key, readers, settings)
    finally:
        for r in readers.values():
            r.close()
    return results


def _ref_from_dirs(base_dir: str, tv_dirs: list[str], weights: list[float], settings: MergeSettings) -> dict[str, torch.Tensor]:
    """Reference GTA over the same tensors loaded from the same directories."""
    base_t = _load_model_tensors(base_dir)
    tvs = [_load_model_tensors(d) for d in tv_dirs]
    out = {}
    for key, base in base_t.items():
        deltas = []
        alphas = []
        for i, tv in enumerate(tvs):
            if key not in tv:
                continue
            t = tv[key]
            if t.shape != base.shape:
                if (
                    base.dim() >= 2
                    and t.dim() >= 2
                    and t.shape[0] >= base.shape[0]
                    and t.shape[1] >= base.shape[1]
                    and t.shape[2:] == base.shape[2:]
                ):
                    t = t[: base.shape[0], : base.shape[1], *base.shape[2:]]
                else:
                    continue
            deltas.append(t.to(base.dtype))
            alphas.append(weights[i])
        out[key] = _ref_gta(
            base,
            deltas,
            alphas,
            lam=settings.lambda_,
            normalize=settings.normalize,
            consensus=settings.consensus.value if settings.consensus else None,
            sparsify_fn=(
                (lambda t: sparsify_inplace(
                    t, settings.density, settings.method,
                    n=settings.n, m=settings.m, gamma=settings.gamma,
                    epsilon=settings.epsilon,
                    rescale_norm=(RescaleNorm.l1 if settings.rescale else None),
                ))
                if settings.method is not None
                else None
            ),
        )
    return out


def _close(a: torch.Tensor, b: torch.Tensor) -> None:
    # bf16 has ~3 decimal digits; the two engines accumulate in different
    # orders, so compare with a bf16-appropriate tolerance.
    assert torch.allclose(a.float(), b.float(), rtol=2e-2, atol=1e-2), (
        f"max diff {torch.abs(a.float() - b.float()).max().item()}"
    )


# --------------------------------------------------------------------------- #
# Synthetic fixtures
# --------------------------------------------------------------------------- #
def _fixture(tmp_path):
    """base (48x64 / 64) bf16 + two fine-tuned sources, TV dirs extracted."""
    torch.manual_seed(11)
    base_t = {"w": torch.randn(48, 64, dtype=torch.bfloat16), "b": torch.randn(64, dtype=torch.bfloat16)}
    base_dir = str(tmp_path / "base")
    _write_model_dir(base_dir, base_t)

    src1_t = {k: (v + torch.randn_like(v) * 0.4) for k, v in base_t.items()}
    src2_t = {k: (v + torch.randn_like(v) * 0.3) for k, v in base_t.items()}
    src1_dir = str(tmp_path / "src1")
    src2_dir = str(tmp_path / "src2")
    _write_model_dir(src1_dir, src1_t)
    _write_model_dir(src2_dir, src2_t)

    tv1 = str(tmp_path / "tv1")
    tv2 = str(tmp_path / "tv2")
    extract_task_vector(base_dir, src1_dir, tv1)
    extract_task_vector(base_dir, src2_dir, tv2)
    return base_dir, tv1, tv2


# --------------------------------------------------------------------------- #
# Tensor-level parity
# --------------------------------------------------------------------------- #
def test_parity_no_consensus(tmp_path):
    base_dir, tv1, tv2 = _fixture(tmp_path)
    s = MergeSettings(density=1.0, normalize=True, lambda_=0.9)
    got = _goat_engine(base_dir, [tv1, tv2], [0.5, 0.7], s)
    ref = _ref_from_dirs(base_dir, [tv1, tv2], [0.5, 0.7], s)
    for k in got:
        _close(got[k], ref[k])


def test_parity_no_consensus_unnormalized(tmp_path):
    base_dir, tv1, tv2 = _fixture(tmp_path)
    s = MergeSettings(density=1.0, normalize=False, lambda_=1.0)
    got = _goat_engine(base_dir, [tv1, tv2], [0.5, 0.7], s)
    ref = _ref_from_dirs(base_dir, [tv1, tv2], [0.5, 0.7], s)
    for k in got:
        _close(got[k], ref[k])


def test_parity_consensus_sum(tmp_path):
    base_dir, tv1, tv2 = _fixture(tmp_path)
    s = MergeSettings(density=1.0, consensus=ConsensusMethod.sum)
    got = _goat_engine(base_dir, [tv1, tv2], [0.5, 0.7], s)
    ref = _ref_from_dirs(base_dir, [tv1, tv2], [0.5, 0.7], s)
    for k in got:
        _close(got[k], ref[k])


def test_parity_consensus_count(tmp_path):
    base_dir, tv1, tv2 = _fixture(tmp_path)
    s = MergeSettings(density=1.0, consensus=ConsensusMethod.count)
    got = _goat_engine(base_dir, [tv1, tv2], [0.5, 0.7], s)
    ref = _ref_from_dirs(base_dir, [tv1, tv2], [0.5, 0.7], s)
    for k in got:
        _close(got[k], ref[k])


def test_parity_magnitude_sparsify(tmp_path):
    # density 0.25 on 48*64=3072 elems -> k=768 exactly (no round/trunc edge)
    base_dir, tv1, tv2 = _fixture(tmp_path)
    s = MergeSettings(
        density=0.25,
        method=SparsificationMethod.magnitude,
        rescale=True,
        consensus=ConsensusMethod.sum,
    )
    got = _goat_engine(base_dir, [tv1, tv2], [0.5, 0.7], s)
    ref = _ref_from_dirs(base_dir, [tv1, tv2], [0.5, 0.7], s)
    for k in got:
        _close(got[k], ref[k])


def test_parity_magnitude_outliers(tmp_path):
    base_dir, tv1, tv2 = _fixture(tmp_path)
    s = MergeSettings(
        density=0.5,
        method=SparsificationMethod.magnitude_outliers,
        gamma=0.1,
        rescale=False,
        consensus=ConsensusMethod.count,
    )
    got = _goat_engine(base_dir, [tv1, tv2], [0.5, 0.7], s)
    ref = _ref_from_dirs(base_dir, [tv1, tv2], [0.5, 0.7], s)
    for k in got:
        _close(got[k], ref[k])


def test_parity_missing_tensor_skip(tmp_path):
    # tv2 lacks tensor "b": the engine must skip it for that tensor only.
    base_dir, tv1, tv2 = _fixture(tmp_path)
    tv2b = str(tmp_path / "tv2b")
    t2 = _load_model_tensors(tv2)
    _write_model_dir(tv2b, {k: v for k, v in t2.items() if k != "b"})

    s = MergeSettings(density=1.0, consensus=ConsensusMethod.sum)
    got = _goat_engine(base_dir, [tv1, tv2b], [0.5, 0.7], s)
    ref = _ref_from_dirs(base_dir, [tv1, tv2b], [0.5, 0.7], s)
    for k in got:
        _close(got[k], ref[k])


def test_parity_embed_submatrix_truncation(tmp_path):
    # TV "w" is (64, 64) vs base (48, 64): engine must truncate to (48, 64).
    base_dir, tv1, tv2 = _fixture(tmp_path)
    tv1b = str(tmp_path / "tv1b")
    t1 = _load_model_tensors(tv1)
    big = torch.randn(64, 64, dtype=torch.bfloat16)  # superset of base (48, 64)
    t1["w"] = big
    _write_model_dir(tv1b, t1)

    s = MergeSettings(density=1.0, consensus=ConsensusMethod.sum)
    got = _goat_engine(base_dir, [tv1b, tv2], [0.5, 0.7], s)
    ref = _ref_from_dirs(base_dir, [tv1b, tv2], [0.5, 0.7], s)
    for k in got:
        _close(got[k], ref[k])


# --------------------------------------------------------------------------- #
# End-to-end: extract -> CLI merge -> verify output + fingerprint gate
# --------------------------------------------------------------------------- #
def test_end_to_end_cli_merge(tmp_path):
    base_dir, tv1, tv2 = _fixture(tmp_path)
    out_dir = str(tmp_path / "merged")
    rc = cli_main(
        [
            "merge",
            "--base", base_dir,
            "--out", out_dir,
            "--tv", f"{tv1}:0.5",
            "--tv", f"{tv2}:0.7",
            "--consensus", "sum",
        ]
    )
    assert rc == 0

    # output layout: single shard + config.json
    assert os.path.exists(os.path.join(out_dir, "model.safetensors"))
    meta = load_metadata(out_dir)
    validate_metadata(meta)
    assert meta["model_type"] == "merged_model"
    assert meta["num_tensors"] == 2
    assert meta["skipped_tensors"] == []

    # values must match the reference GTA kernel on the same TVs
    s = MergeSettings(density=1.0, consensus=ConsensusMethod.sum)
    got = _goat_engine(base_dir, [tv1, tv2], [0.5, 0.7], s)
    ref = _ref_from_dirs(base_dir, [tv1, tv2], [0.5, 0.7], s)
    out_tensors = _load_model_tensors(out_dir)
    for k in got:
        _close(out_tensors[k], ref[k])


def test_fingerprint_gate_accepts_matching_base(tmp_path):
    base_dir, tv1, tv2 = _fixture(tmp_path)
    assert verify_against_base(base_dir, [tv1, tv2]) == []


def test_fingerprint_gate_rejects_wrong_base(tmp_path):
    base_dir, tv1, tv2 = _fixture(tmp_path)
    other_base = str(tmp_path / "other_base")
    torch.manual_seed(99)
    _write_model_dir(other_base, {"w": torch.randn(48, 64, dtype=torch.bfloat16), "b": torch.randn(64, dtype=torch.bfloat16)})

    # tv1 was extracted against base_dir, not other_base
    assert verify_against_base(other_base, [tv1]) == [tv1]

    # the CLI must refuse the merge with exit code 3
    out_dir = str(tmp_path / "merged_bad")
    rc = cli_main(
        [
            "merge",
            "--base", other_base,
            "--out", out_dir,
            "--tv", f"{tv1}:0.5",
        ]
    )
    assert rc == 3
    assert not os.path.exists(os.path.join(out_dir, "model.safetensors"))


def test_extract_roundtrip(tmp_path):
    base_dir, tv1, tv2 = _fixture(tmp_path)
    base_t = _load_model_tensors(base_dir)
    src1_t = _load_model_tensors(str(tmp_path / "src1"))
    tv1_t = _load_model_tensors(tv1)
    for k in base_t:
        # TV stored in base dtype and equal to source - base (bf16 arithmetic)
        expected = (src1_t[k].to(torch.bfloat16) - base_t[k])
        assert torch.equal(tv1_t[k], expected)
    meta = load_metadata(tv1)
    validate_metadata(meta)
    assert meta["base_fingerprint"] == compute_base_fingerprint(base_dir)
