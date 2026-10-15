# Memory model

`S` = bytes of the largest single tensor.

- **Design budget, standard path:** peak ≈ 5–7 S
  (base + `acc` + `l1` + `c` + signs + 1 delta + sparsify temp).
- **Chunk Merge path:** O(S) + O(chunk) — one output buffer plus one chunk of
  working state. See [chunk-merge.md](chunk-merge.md).
- **TV extraction:** 2 S standard; O(S) + O(chunk) with `--chunk-elements`.
- **Fingerprint anchor hash:** O(chunk) streaming (was ~3.02 S, now ~0.62 S).

## Measured numbers

- Synthetic 100 MB bf16 tensor (2 TVs, consensus=sum): **105.6 MB peak**
  (Peak / S = **1.06**). Tensors are freed between passes (`l1` before the
  divisor assembly, each `delta` right after accumulation), so the measured
  value beats the 5–7 S worst-case budget.
- Isolated harness (`tests/measure_peak_ram.py`, subprocess isolation):
  standard path **6.14 S**, Chunk Merge **3.21 S** (100 MB bf16, consensus=sum).
- GoatMerge vs mergekit GTA (3 × 300 MB bf16, consensus=sum): **3394.6 MB**
  vs **7289.4 MB** peak RSS (≈47%; ≈44% on merge-attributable memory),
  max|d| = 0.0625.

Rough per-layer numbers:

| Model scale | S (bf16, per layer) | GoatMerge | mergekit GTA |
|---|---|---|---|
| 9B-class | ≈ 1.8 GB | ≈ 2–4 GB | ≈ 22–34 GB |
| 400B-class | ≈ 1.7 GB | ≈ 2 GB | ≈ 34–53 GB (impractical on one GPU) |

## Measuring it yourself

```bash
/home/CloudGoat/venvs/mergekit/bin/python -m pytest   # 97 passed is the baseline
/home/CloudGoat/venvs/mergekit/bin/python tests/measure_peak_ram.py
```

`tests/measure_peak_ram.py` is outside pytest's `testpaths`; it builds fixtures
and runs each merge mode in a child process, reporting peak/S for both the
standard and Chunk Merge paths. `Benchmarks/` holds the GoatMerge-vs-mergekit
comparison scripts and results.

Page-cache note: process RSS is not the whole story on bulk jobs. `--drop-caches`
keeps the *system* file cache from ballooning (+17.2 GB → +2.7 GB on a 9.3 GB
model run, 96 s → 45 s, same 2.3 GB process peak). See [io.md](io.md).
