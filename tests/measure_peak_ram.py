"""Measure peak RSS of the streaming merge engine on a synthetic model.

Creates a base model + 2 TV models with a single tensor of known size S,
runs the full merge (consensus=sum), and reports the merge's peak RSS
contribution via resource.getrusage high-water-mark deltas.

Expected: merge peak ≈ 5–7 S (S = largest tensor bytes).
"""

import json
import resource
import shutil
import tempfile
from pathlib import Path

import torch

from goatmerge.io import ShardedTensorIndex
from goatmerge.merge import MergeSettings, ModelEntry, merge_model
from goatmerge.consensus import ConsensusMethod


def _write_safetensors(path: Path, tensors: dict[str, torch.Tensor]) -> None:
    """Write a single-shard safetensors file."""
    import safetensors
    safetensors.torch.save_file(tensors, str(path))


def make_model_dir(d: Path, tensors: dict[str, torch.Tensor]) -> None:
    d.mkdir(parents=True, exist_ok=True)
    _write_safetensors(d / "model.safetensors", tensors)
    index = {k: "model.safetensors" for k in tensors}
    (d / "model.safetensors.index.json").write_text(
        json.dumps({"metadata": {}, "weight_map": index})
    )


def main() -> None:
    # --- choose S (largest tensor bytes) ---
    # 100 MB bf16 tensor: 100 * 1024 * 1024 / 2 = 52_428_800 elements
    S_bytes = 100 * 1024 * 1024  # 100 MB
    n_elements = S_bytes // 2  # bf16 = 2 bytes/element
    shape = (n_elements,)

    # --- create temp dirs ---
    tmp = Path(tempfile.mkdtemp(prefix="goatmerge_ram_"))
    base_dir = tmp / "base"
    tv1_dir = tmp / "tv1"
    tv2_dir = tmp / "tv2"
    out_dir = tmp / "out"

    # --- create tensors (these are the "on-disk" models) ---
    base_t = torch.randn(n_elements, dtype=torch.bfloat16)
    tv1_t = (base_t.float() + torch.randn(n_elements, dtype=torch.float32) * 0.01).to(torch.bfloat16)
    tv2_t = (base_t.float() + torch.randn(n_elements, dtype=torch.float32) * 0.01).to(torch.bfloat16)

    make_model_dir(base_dir, {"weight": base_t})
    make_model_dir(tv1_dir, {"weight": tv1_t})
    make_model_dir(tv2_dir, {"weight": tv2_t})

    # Free the in-memory tensors (they're on disk now)
    del base_t, tv1_t, tv2_t
    torch.cuda.empty_cache()  # no-op on CPU, but harmless

    # --- record RSS high-water before merge ---
    rss_before = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss  # KB

    # --- run merge ---
    settings = MergeSettings(
        consensus=ConsensusMethod.sum,
        normalize=True,
        lambda_=1.0,
    )
    entries = [
        ModelEntry(dir=tv1_dir, kind="tv", weight=0.5),
        ModelEntry(dir=tv2_dir, kind="tv", weight=0.7),
    ]

    result = merge_model(
        base_dir=base_dir,
        entries=entries,
        settings=settings,
        out_dir=out_dir,
    )

    # --- record RSS high-water after merge ---
    rss_after = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss  # KB

    # --- report ---
    S_mb = S_bytes / (1024 * 1024)
    merge_peak_mb = (rss_after - rss_before) / 1024  # KB -> MB
    ratio = merge_peak_mb / S_mb

    print(f"S (largest tensor):  {S_mb:.1f} MB")
    print(f"Merge peak RSS:      {merge_peak_mb:.1f} MB")
    print(f"Peak / S:            {ratio:.2f}  (target: 5-7)")
    print(f"Tensors merged:      {result['num_tensors']}")
    print(f"Skipped:             {len(result['skipped_tensors'])}")

    # Cleanup
    shutil.rmtree(tmp)


if __name__ == "__main__":
    main()
