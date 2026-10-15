# Testing & benchmarks

Interpreter: `/home/CloudGoat/venvs/mergekit/bin/python` (mergekit lives in the
same venv). Dependencies: Python ≥ 3.10, `torch>=2.1`, `safetensors>=0.4`,
`pyyaml>=6.0`.

```bash
/home/CloudGoat/venvs/mergekit/bin/python -m pytest            # 97 passed is the baseline
/home/CloudGoat/venvs/mergekit/bin/python -m goatmerge.cli merge -c examples/merge_recipe.yaml
/home/CloudGoat/venvs/mergekit/bin/python -m goatmerge.cli extract --base B --source S --out TV
/home/CloudGoat/venvs/mergekit/bin/python -m goatmerge.cli extract --base B --source S --out TV --chunk-elements 33554432
/home/CloudGoat/venvs/mergekit/bin/python -m goatmerge.cli inspect --dir MODEL
```

## Tests

| File | Content |
|---|---|
| `tests/test_consensus_merge.py` | mergekit GTA parity, CLI e2e, fingerprint gate, extract round-trip, chunked-extract equality/fallback |
| `tests/test_chunk_merge.py` | Chunk Merge vs standard-path equality, sparsify validity, consensus combos |
| `tests/test_kernels.py` | linear / mixture / slerp / ties kernels |
| `tests/test_sparsify.py` | Each sparsification method, default-chunk-budget equality, bernoulli determinism |
| `tests/test_select_threshold.py` | `_select_threshold` exactly matches sorting (f32/bf16/ties/all-equal, factory match) |
| `tests/test_fingerprint.py` | Fingerprint determinism / sensitivity |
| `tests/test_io.py` | Sharded / single-shard / row-slice reads, writer naming and sharding, drop-caches advise |
| `tests/test_metadata.py` | `goatmerge.json` envelope |
| `tests/measure_peak_ram.py` | Peak-RAM harness (outside pytest `testpaths`; child-process isolation, reports peak/S for both paths) |

## Parity method

Changes to merge logic must be checked against the mergekit reference
implementation using the `tests/test_consensus_merge.py` patterns:
bf16 tensors, rtol=2e-2, atol=1e-2. The critical subtleties are the bf16·bf16
weighted product and the `(acc + M·l1)/2` identity — see
[streaming-kernel.md](streaming-kernel.md) and invariant #1/#2 in
[architecture.md](architecture.md).

## Benchmarks

- `benchmark.py` / `bench_real_models.py` / `tests/measure_peak_ram.py`
- `Benchmarks/benchmark_compare.py`, `Benchmarks/benchmark_task_arithmetic.py`,
  `Benchmarks/RESULTS.md` — GoatMerge vs mergekit GTA methodology and results.
