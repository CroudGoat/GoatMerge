"""Practical benchmark: GoatMerge pure Task-Arithmetic merge on real HF-cached models.

Runs merge_model for each source model (individually) against the base,
measuring wall time and peak RSS.

Usage:
    python bench_real_models.py
"""

from __future__ import annotations

import json
import resource
import time
from pathlib import Path

from goatmerge.consensus import ConsensusMethod
from goatmerge.hf import resolve_model_dir
from goatmerge.merge import MergeSettings, ModelEntry, merge_model
from goatmerge.merge_method import MergeMethod
from goatmerge.sparsify import SparsificationMethod

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

BASE_REPO = "Qwen/Qwen3.5-0.8B"

SOURCES = [
    "CloudGoat/Qwen3.5-0.8B-JP-Tuned-v1.1",
    "Takenoko12345678/Qwen3.5-0.8B-Japanese-SFT-v2",
    "zosmaai/Qwen3.5-0.8B-GRPO-Math",
]

# Pure Task Arithmetic: default GTA, no sparsification, no consensus
SETTINGS = MergeSettings(
    merge_method=MergeMethod.gta,
    density=1.0,
    method=None,
    n=64,
    m=256,
    gamma=0.0,
    epsilon=0.0,
    rescale=True,
    normalize=True,
    lambda_=1.0,
    consensus=ConsensusMethod.none,
    chunk_elements=10_000_000,
)

OUT_ROOT = Path("/tmp/goatmerge_bench_out")


def _peak_rss_mb() -> float:
    """Return peak RSS of this process in MB (Linux)."""
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024


def main() -> None:
    base = resolve_model_dir(BASE_REPO)
    print(f"Base model: {base}")
    print(f"Base tensors: {len(list(base.glob('*.safetensors')))} safetensors file(s)")
    print()

    results: list[dict] = []

    for src_repo in SOURCES:
        src = resolve_model_dir(src_repo)
        out_dir = OUT_ROOT / src_repo.replace("/", "_")
        out_dir.mkdir(parents=True, exist_ok=True)

        entry = ModelEntry(dir=src, kind="model", weight=1.0)

        # Reset peak RSS baseline (ru_maxrss is monotonically increasing)
        baseline_rss = _peak_rss_mb()

        t0 = time.perf_counter()
        summary = merge_model(base, [entry], SETTINGS, out_dir)
        t1 = time.perf_counter()

        elapsed_ms = (t1 - t0) * 1000
        peak_rss = _peak_rss_mb()
        delta_rss = peak_rss - baseline_rss  # approximate delta

        results.append({
            "source": src_repo,
            "elapsed_ms": round(elapsed_ms, 1),
            "peak_rss_mb": round(peak_rss, 1),
            "delta_rss_mb": round(delta_rss, 1),
            "num_tensors": summary["num_tensors"],
        })

        print(f"  {src_repo}")
        print(f"    time:   {elapsed_ms:.1f} ms")
        print(f"    peak RSS: {peak_rss:.1f} MB (delta: {delta_rss:.1f} MB)")
        print(f"    tensors: {summary['num_tensors']}")
        print()

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print("=" * 60)
    print("BENCHMARK SUMMARY — Pure Task Arithmetic (GTA)")
    print("=" * 60)
    print(f"{'Source':<45} {'Time (ms)':>10} {'Peak RSS (MB)':>14}")
    print("-" * 70)
    for r in results:
        print(f"{r['source']:<45} {r['elapsed_ms']:>10.1f} {r['peak_rss_mb']:>14.1f}")

    # Save JSON
    out_json = OUT_ROOT / "results.json"
    with open(out_json, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: {out_json}")


if __name__ == "__main__":
    main()
