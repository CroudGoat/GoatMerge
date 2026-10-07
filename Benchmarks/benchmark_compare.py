"""Compare GoatMerge's streaming engine against mergekit GTA (stack-based)
on 3x300MB synthetic task vectors: wall time and peak RSS.

Run (from the repo root, with the repo root on PYTHONPATH):
    python Benchmarks/benchmark_compare.py

Each engine runs in its own subprocess so peak RSS is measured cleanly.
A third subprocess verifies numerical parity between the two engines.

Subprocess protocol (internal):
    python Benchmarks/benchmark_compare.py goat     <base> <tv1> <tv2> <tv3> <w1> <w2> <w3> <out>
    python Benchmarks/benchmark_compare.py mergekit <base> <tv1> <tv2> <tv3> <w1> <w2> <w3>
    python Benchmarks/benchmark_compare.py parity   <base> <tv1> <tv2> <tv3> <w1> <w2> <w3>
"""

from __future__ import annotations

import json
import os
import resource
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

# Ensure the workspace root is on sys.path so goatmerge is importable
# when the script is run as a file (sys.path[0] = script directory).
_ROOT = str(Path(__file__).resolve().parent.parent)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import safetensors.torch
import torch

# 300 MB bf16 tensor
S_BYTES = 300 * 1024 * 1024          # 300 MB
N_ELEMS = S_BYTES // 2               # bf16 = 2 bytes/element
N_TV = 3
WEIGHTS = [0.5, 0.7, 0.3]


# --------------------------------------------------------------------------- #
# Reference kernel: mergekit GTA (stack-based), copied from test_consensus_merge
# --------------------------------------------------------------------------- #
def _ref_gta(base, deltas, alphas, *, lam=1.0, normalize=True, consensus=None):
    if not deltas:
        return base
    stacked = torch.stack(deltas, dim=0)
    weights = torch.tensor(alphas, dtype=stacked.dtype, device=stacked.device)
    while stacked.dim() > weights.dim():
        weights = weights.unsqueeze(-1)
    weighted = stacked * weights

    if consensus in ("sum", "count"):
        sign = weighted.sign()
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


def _load_dir(path: str) -> dict[str, torch.Tensor]:
    out = {}
    with safetensors.torch.safe_open(os.path.join(path, "model.safetensors"), framework="pt") as st:
        for k in st.keys():
            out[k] = st.get_tensor(k)
    return out


def _peak_rss_mb() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0  # KB -> MB


def _read_vm_rss_mb() -> float:
    """Read current VmRSS from /proc/self/status in MB."""
    with open("/proc/self/status") as f:
        for line in f:
            if line.startswith("VmRSS:"):
                return int(line.split()[1]) / 1024.0  # kB -> MB
    return 0.0


def _sample_peak_rss_mb(run) -> tuple[float, float, float]:
    """Run ``run()`` while sampling VmRSS; return (elapsed_s, peak_rss_mb, baseline_rss_mb).

    ``ru_maxrss`` is a lifetime high-water mark that includes transient
    allocations (e.g. the float32 temporaries made while building the
    tensors), so it overstates the merge's steady-state peak. Sampling
    ``/proc/self/status`` VmRSS during the run captures the true peak.

    ``baseline_rss_mb`` is the VmRSS read just before the run starts,
    so ``marginal_peak = peak_rss_mb - baseline_rss_mb`` isolates the
    memory increase attributable to the merge operation itself.
    """
    import threading

    stop = threading.Event()
    peak = [0.0]

    def sampler():
        while not stop.is_set():
            try:
                with open("/proc/self/status") as f:
                    for line in f:
                        if line.startswith("VmRSS:"):
                            peak[0] = max(peak[0], int(line.split()[1]) / 1024.0)  # kB -> MB
                            break
            except Exception:
                pass
            time.sleep(0.005)

    baseline = _read_vm_rss_mb()
    t = threading.Thread(target=sampler)
    t0 = time.perf_counter()
    t.start()
    run()
    t1 = time.perf_counter()
    stop.set()
    t.join()
    return t1 - t0, peak[0], baseline


# --------------------------------------------------------------------------- #
# Child entry points (each runs in its own subprocess)
# --------------------------------------------------------------------------- #
def child_goat(base_dir, tv_dirs, weights, out_dir):
    from goatmerge.io import ShardedTensorIndex
    from goatmerge.merge import MergeSettings, ModelEntry, merge_model
    from goatmerge.consensus import ConsensusMethod

    settings = MergeSettings(consensus=ConsensusMethod.sum, normalize=True, lambda_=1.0)
    entries = [ModelEntry(dir=Path(d), kind="tv", weight=w) for d, w in zip(tv_dirs, weights)]

    t0 = time.perf_counter()
    res = merge_model(base_dir=Path(base_dir), entries=entries, settings=settings, out_dir=Path(out_dir))
    t1 = time.perf_counter()
    return t1 - t0, res


def child_mergekit(base_dir, tv_dirs, weights):
    base_t = _load_dir(base_dir)
    base = base_t["weight"].clone()
    deltas = [t["weight"].clone() for t in (_load_dir(d) for d in tv_dirs)]

    t0 = time.perf_counter()
    result = _ref_gta(base, deltas, weights, lam=1.0, normalize=True, consensus="sum")
    t1 = time.perf_counter()
    return t1 - t0, result


def child_parity(base_dir, tv_dirs, weights):
    from goatmerge.io import ShardReader, ShardedTensorIndex
    from goatmerge.merge import MergeSettings, ModelEntry, merge_tensor
    from goatmerge.consensus import ConsensusMethod

    base_t = _load_dir(base_dir)
    base = base_t["weight"].clone()
    settings = MergeSettings(consensus=ConsensusMethod.sum, normalize=True, lambda_=1.0)
    entries = [ModelEntry(dir=Path(d), kind="tv", weight=w) for d, w in zip(tv_dirs, weights)]
    readers = {Path(d): ShardReader(ShardedTensorIndex.from_dir(Path(d))) for d in tv_dirs}
    goat = merge_tensor(base.clone(), entries, "weight", readers, settings)
    mk = _ref_gta(base, [t["weight"].clone() for t in (_load_dir(d) for d in tv_dirs)],
                  weights, lam=1.0, normalize=True, consensus="sum")
    return (goat.float() - mk.float()).abs().max().item()


# --------------------------------------------------------------------------- #
# Parent orchestration
# --------------------------------------------------------------------------- #
def _build_models(root: Path) -> tuple[str, list[str]]:
    torch.manual_seed(0)
    base = torch.randn(N_ELEMS, dtype=torch.bfloat16)
    tvs = [
        (base.float() + torch.randn(N_ELEMS, dtype=torch.float32) * 0.01).to(torch.bfloat16)
        for _ in range(N_TV)
    ]

    base_dir = root / "base"
    base_dir.mkdir(parents=True, exist_ok=True)
    safetensors.torch.save_file({"weight": base}, str(base_dir / "model.safetensors"))

    tv_dirs = []
    for i, tv in enumerate(tvs):
        d = root / f"tv{i+1}"
        d.mkdir(parents=True, exist_ok=True)
        safetensors.torch.save_file({"weight": tv}, str(d / "model.safetensors"))
        tv_dirs.append(str(d))

    del base, tvs
    return str(base_dir), tv_dirs


def _run_child(args: list[str]) -> dict:
    # Ensure the workspace root is on PYTHONPATH so child subprocesses
    # can import goatmerge (the script directory alone is not enough).
    root = str(Path(__file__).resolve().parent.parent)
    env = dict(os.environ)
    existing = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = root + (":" + existing if existing else "")
    proc = subprocess.run(
        [sys.executable, str(Path(__file__).resolve())] + args,
        capture_output=True, text=True, env=env,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"child failed:\n{proc.stderr}")
    return json.loads(proc.stdout.strip().splitlines()[-1])


def main() -> None:
    tmp = Path(tempfile.mkdtemp(prefix="gm_bench_"))
    base_dir, tv_dirs = _build_models(tmp)
    out_dir = str(tmp / "out")
    weights = WEIGHTS

    goat = _run_child(["goat", base_dir, *tv_dirs, *[str(w) for w in weights], out_dir])
    mk = _run_child(["mergekit", base_dir, *tv_dirs, *[str(w) for w in weights]])
    parity = _run_child(["parity", base_dir, *tv_dirs, *[str(w) for w in weights]])

    s_mb = S_BYTES / (1024 * 1024)
    print(f"\n=== Benchmark: {N_TV}x{s_mb:.0f}MB bf16 task vectors ===")
    print(f"{'engine':<16} {'time (s)':>10} {'peak RSS (MB)':>14} {'marginal (MB)':>14}  notes")
    print(f"{'GoatMerge':<16} {goat['time']:>10.3f} {goat['peak_mb']:>14.1f} {goat['marginal_peak_mb']:>14.1f}  streaming (no stack)")
    print(f"{'mergekit GTA':<16} {mk['time']:>10.3f} {mk['peak_mb']:>14.1f} {mk['marginal_peak_mb']:>14.1f}  stack-based reference")
    print(f"{'parity max|d|':<16} {parity['max_diff']:>10.4f} {'(bf16 rtol=2e-2)':>14}")
    print(f"\nSpeedup (mergekit/GoatMerge): {mk['time']/goat['time']:.2f}x")
    print(f"Memory (GoatMerge/mergekit):   {goat['peak_mb']/mk['peak_mb']:.3f}")
    print(f"Marginal (GoatMerge/mergekit): {goat['marginal_peak_mb']/mk['marginal_peak_mb']:.3f}")

    shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    if len(sys.argv) > 1:
        cmd = sys.argv[1]
        base_dir = sys.argv[2]
        tv_dirs = sys.argv[3:3 + N_TV]
        weights = [float(x) for x in sys.argv[3 + N_TV:3 + 2 * N_TV]]
        if cmd == "goat":
            t, peak, baseline = _sample_peak_rss_mb(lambda: child_goat(
                base_dir, tv_dirs, weights, sys.argv[3 + 2 * N_TV]))
            out = {"time": t, "peak_mb": peak, "baseline_mb": baseline,
                 "marginal_peak_mb": peak - baseline}
        elif cmd == "mergekit":
            t, peak, baseline = _sample_peak_rss_mb(lambda: child_mergekit(base_dir, tv_dirs, weights))
            out = {"time": t, "peak_mb": peak, "baseline_mb": baseline,
                 "marginal_peak_mb": peak - baseline}
        elif cmd == "parity":
            out = {"max_diff": child_parity(base_dir, tv_dirs, weights)}
        else:
            raise SystemExit(f"unknown cmd {cmd}")
        print(json.dumps(out))
    else:
        main()
