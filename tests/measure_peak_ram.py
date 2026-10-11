"""Measure peak RSS of the streaming merge engine on a synthetic model.

Creates a base model + 2 TV models with a single tensor of known size S and
reports each merge mode's peak RSS contribution. The fixtures are built in a
child process and every merge runs in its own child, so no allocation of the
setup can land in the measured process's high-water mark.

Usage::

    python tests/measure_peak_ram.py

Expected (S = largest tensor bytes):
  standard merge   ~5-7 S   (base + acc + l1 + c/nz + one delta + sparsify temp)
  Chunk Merge      ~2-3 S   (the output buffer + one row chunk per operand)
"""

import json
import multiprocessing
import resource
import shutil
import tempfile
from pathlib import Path

import torch

from goatmerge.consensus import ConsensusMethod
from goatmerge.merge import MergeSettings, ModelEntry, merge_model

S_BYTES = 100 * 1024 * 1024  # 100 MB
N_ELEMENTS = S_BYTES // 2  # bf16 = 2 bytes/element


def _write_safetensors(path: Path, tensors: dict[str, torch.Tensor]) -> None:
    import safetensors

    safetensors.torch.save_file(tensors, str(path))


def make_model_dir(d: Path, tensors: dict[str, torch.Tensor]) -> None:
    d.mkdir(parents=True, exist_ok=True)
    _write_safetensors(d / "model.safetensors", tensors)
    index = {k: "model.safetensors" for k in tensors}
    (d / "model.safetensors.index.json").write_text(
        json.dumps({"metadata": {}, "weight_map": index})
    )


def _build_fixtures(tmp: Path) -> None:
    """Child-process job: write the on-disk models (memory-heavy, discarded)."""
    torch.manual_seed(0)
    base_t = torch.randn(N_ELEMENTS, dtype=torch.bfloat16)
    tv1_t = (base_t.float() + torch.randn(N_ELEMENTS) * 0.01).to(torch.bfloat16)
    tv2_t = (base_t.float() + torch.randn(N_ELEMENTS) * 0.01).to(torch.bfloat16)

    make_model_dir(tmp / "base", {"weight": base_t})
    make_model_dir(tmp / "tv1", {"weight": tv1_t})
    make_model_dir(tmp / "tv2", {"weight": tv2_t})


def _run_merge(tmp: Path, out_dir: str, chunk_elements, result_file: Path) -> None:
    """Child-process job: run one merge and report its RSS high-water delta."""
    settings = MergeSettings(
        consensus=ConsensusMethod.sum,
        normalize=True,
        lambda_=1.0,
        chunk_elements=chunk_elements,
    )
    entries = [
        ModelEntry(dir=tmp / "tv1", kind="tv", weight=0.5),
        ModelEntry(dir=tmp / "tv2", kind="tv", weight=0.7),
    ]
    rss_before = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    merge_model(
        base_dir=tmp / "base",
        entries=entries,
        settings=settings,
        out_dir=tmp / out_dir,
    )
    rss_after = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    result_file.write_text(str(rss_after - rss_before), encoding="utf-8")


def _measure(tmp: Path, out_dir: str, chunk_elements):
    ctx = multiprocessing.get_context("fork")
    result_file = tmp / f"peak_{out_dir}.txt"
    p = ctx.Process(target=_run_merge, args=(tmp, out_dir, chunk_elements, result_file))
    p.start()
    p.join()
    if p.exitcode != 0:
        raise RuntimeError(f"merge subprocess failed with code {p.exitcode}")
    peak_mb = int(result_file.read_text(encoding="utf-8")) / 1024  # KB -> MB
    return peak_mb / (S_BYTES / (1024 * 1024)), peak_mb


def main() -> None:
    s_mb = S_BYTES / (1024 * 1024)
    tmp = Path(tempfile.mkdtemp(prefix="goatmerge_ram_"))
    try:
        ctx = multiprocessing.get_context("fork")
        builder = ctx.Process(target=_build_fixtures, args=(tmp,))
        builder.start()
        builder.join()
        if builder.exitcode != 0:
            raise RuntimeError("fixture build failed")

        for label, out_dir, chunk in (
            ("standard merge", "out_standard", None),
            ("Chunk Merge (16M elems)", "out_chunk", 16 * 1024 * 1024),
        ):
            ratio, peak_mb = _measure(tmp, out_dir, chunk)
            print(f"{label:24s} peak {peak_mb:8.1f} MB   peak/S {ratio:5.2f}")
    finally:
        shutil.rmtree(tmp)


if __name__ == "__main__":
    main()
