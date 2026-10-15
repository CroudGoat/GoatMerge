# GoatMerge — Overview

**GoatMerge** is a *streaming* task-arithmetic merge engine. It merges
fine-tuned models (task vectors, TVs) into a base model without ever calling
`torch.stack`, keeping peak memory far below mergekit's GTA while retaining
numerical parity with it.

- **No `torch.stack`** — deltas are streamed one at a time and accumulated in place.
- **Peak RAM ≈ 5–7 S** (S = largest tensor bytes); measured at **~1.06 S** on a
  100 MB bf16 tensor and **6.14 S** in the isolated peak-RAM harness
  (see [memory.md](memory.md)).
- **Numerical parity with mergekit GTA** — rtol = 2e-2, atol = 1e-2 on bf16.
- **Chunk Merge** — merges tensors of any size in O(S) + O(chunk) memory, with
  results identical to the standard path. See [chunk-merge.md](chunk-merge.md).
- **HF-sharded safetensors** — reads and writes the standard HuggingFace layout
  as-is. See [io.md](io.md).
- **Task-vector extraction and reuse** — `extract` once (`T = W_source - W_base`),
  then merge TVs without re-reading source models. See [quickstart.md](quickstart.md).
- **Base fingerprint verification** — refuses TVs extracted against another base
  (exit code 3). See [fingerprint.md](fingerprint.md).

## Why GoatMerge

mergekit's GTA stacks every delta into a `(k, shape)` tensor with `torch.stack`.
For a 400B-class model (S ≈ 1.7 GB) with k = 4 task vectors that alone peaks at
**34–53 GB**, which does not fit on a single GPU.

GoatMerge streams one delta at a time and derives the consensus from three
accumulators — `acc`, `l1`, `c`. The masked sum is replaced by the exact
identity `(acc + M·l1)/2`, so no k-sized mask tensor is ever materialized.

| Aspect | GoatMerge | mergekit GTA |
|---|---|---|
| Delta handling | Streams one at a time | `torch.stack` over all deltas |
| Peak RAM (k TVs, 1 tensor) | ≈ 5–7 S | (4k+4) S – (6k+7) S |
| Consensus | Identity `(acc + M·l1)/2` | Full mask multiply over all weighted deltas |
| Very large tensors | Chunk Merge: O(S) + O(chunk) | All deltas resident |
| I/O | HF-sharded safetensors | HF-sharded safetensors |

Benchmark (3 × 300 MB bf16 tensors, consensus=sum): GoatMerge peaks at roughly
**47%** of mergekit's RSS (≈44% on merge-attributable memory), with
max|d| = 0.0625. See [testing.md](testing.md) for methodology.

## Where to go next

1. New here? Start at [quickstart.md](quickstart.md).
2. Running merges daily? Keep [cli.md](cli.md) and [yaml-recipes.md](yaml-recipes.md) handy.
3. Tuning quality? Read [merge-methods.md](merge-methods.md) then
   [sparsification.md](sparsification.md).
4. Big tensors or tight RAM? Read [chunk-merge.md](chunk-merge.md) and
   [memory.md](memory.md).
5. Contributing? Read [architecture.md](architecture.md) first — it lists the
   invariants you must not break.
