# GoatMerge

![GoatMerge](GoatMerge.png)

[日本語版はこちら](README-ja.md)

**GoatMerge** is a *streaming* task-arithmetic merge engine. It merges
fine-tuned models (task vectors) into a base model without ever calling
`torch.stack`, keeping peak memory far below mergekit's GTA while retaining
numerical parity with it.

- **No `torch.stack`** — deltas are streamed one at a time and accumulated in place
- **Peak RAM ≈ 5–7 S** (S = largest tensor bytes); measured at **1.06 S**
- **Numerical parity with mergekit GTA** — rtol = 2e-2, atol = 1e-2 on bf16
- **Chunked mode** — handles tensors of any size in O(S) + O(chunk) memory
- **HF-sharded safetensors** — reads and writes the standard HuggingFace layout
- **Task-vector extraction and reuse** — no re-reading of source models
- **Base fingerprint verification** — refuses TVs extracted against another base

---

## Why GoatMerge

mergekit's GTA stacks every delta into a `(k, shape)` tensor with
`torch.stack`. For a 400B-class model (S ≈ 1.7 GB) with k = 4 task vectors
that alone peaks at **34–53 GB**, which does not fit on a single GPU.

GoatMerge streams one delta at a time and derives the consensus from three
accumulators — `acc`, `l1`, `c`. The masked sum is replaced by the exact
identity `(acc + M·l1)/2`, so no k-sized mask tensor is ever materialized.

| Aspect | GoatMerge | mergekit GTA |
|---|---|---|
| Delta handling | Streams one at a time | `torch.stack` over all deltas |
| Peak RAM (k TVs, 1 tensor) | ≈ 5–7 S | (4k+4) S – (6k+7) S |
| Consensus | Identity `(acc + M·l1)/2` | Full mask multiply over all weighted deltas |
| Very large tensors | Chunked: O(S) + O(chunk) | All deltas resident |
| I/O | HF-sharded safetensors | HF-sharded safetensors |

### Benchmark (3 × 300 MB bf16 tensors, consensus=sum)

| Engine | Peak RSS | Incremental | Numerical parity |
|---|---|---|---|
| **GoatMerge** | 3394.6 MB | 3009.1 MB | max\|d\| = 0.0625 |
| **mergekit GTA** | 7289.4 MB | 6903.8 MB | same |

GoatMerge peaks at roughly **47%** of mergekit's RSS, and at roughly **44%**
when only the memory attributable to the merge itself is counted. See
[`Benchmarks/RESULTS.md`](Benchmarks/RESULTS.md) for the methodology.

---

## Installation

```bash
pip install -e .
```

Requirements:

- Python ≥ 3.10
- PyTorch, `safetensors`
- `pyyaml` (for YAML recipes)

---

## Quick start

### 1. Extract task vectors (recommended)

Compute the difference against the base model once and store it as a task
vector. Later merges never read the source models again.

```bash
goatmerge extract \
  --base   /path/to/base_model \
  --source /path/to/finetuned_model \
  --out    /path/to/tv_jp
```

### 2. Merge

```bash
goatmerge merge \
  --base /path/to/base_model \
  --tv /path/to/tv_jp:0.7 \
  --tv /path/to/tv_math:0.5 \
  --out /path/to/merged_model \
  --consensus sum
```

Weights are given as `--tv DIR:WEIGHT` (the same syntax works for `--model`).
Use `--model` to merge source models directly, in which case the delta is
computed as `source − base` on the fly.

### 3. Inspect the result

```bash
goatmerge inspect --dir /path/to/merged_model
```

Every task vector and merged model carries a `goatmerge.json` sidecar
recording the base model, the task vectors, the applied settings, and a
fingerprint. At merge time the fingerprint stored in each TV is compared with
the fingerprint of the base you passed, and the merge is refused (exit code 3)
when they disagree. Use `--skip-fingerprint-check` to bypass the check.

A merged model directory is **directly loadable by `transformers`**: the base
model's `config.json` and tokenizer files are copied over automatically, since
GoatMerge only rewrites weights and those files are unchanged.

```
merged_model/
  model.safetensors           merged weights
  config.json                 copied from the base model (HF model config)
  tokenizer_config.json ...   copied when present in the base
  goatmerge.json              GoatMerge metadata
```

> Keep your merge *recipes* in YAML (`goatmerge merge -c recipe.yaml`).
> `goatmerge.json` is not a replacement for a recipe — it records what a
> directory was produced from.

Model references may be a local directory or a HuggingFace repo id that is
already present in the local HF cache.

---

## CLI reference

### `merge`

| Flag | Default | Description |
|---|---|---|
| `--base` | required | Base model directory, or a cached HF repo id |
| `--tv DIR:WEIGHT` | — | Task vector to merge (repeatable) |
| `--model DIR:WEIGHT` | — | Source model to merge (repeatable) |
| `--out` | required | Output directory |
| `--merge-method` | `gta` | `gta` \| `linear` \| `mixture` \| `slerp` \| `ties` |
| `--consensus` | `none` | `none` \| `sum` \| `count` |
| `--density` | `1.0` | Sparsification density; `1.0` disables it |
| `--method` | (none) | `magnitude` \| `random` \| `magnitude_outliers` \| `della_magprune` \| `bs` |
| `--n` | `64` | Kept values per block for BS (n:m) |
| `--m` | `256` | BS block size |
| `--gamma` | `0.0` | Fraction dropped from the top by `magnitude_outliers` |
| `--epsilon` | `0.0` | Probability spread for `della_magprune` |
| `--no-rescale` | false | Disable norm re-normalization after sparsification |
| `--no-normalize` | false | Disable divisor normalization |
| `--lambda` | `1.0` | Scale applied to the mixed delta |
| `--chunk-elements` | (none) | Chunk tensors above this many elements |
| `--skip-fingerprint-check` | false | Skip the base fingerprint comparison |
| `-c, --config` | (none) | YAML recipe file (see below) |

### `extract`

| Flag | Default | Description |
|---|---|---|
| `--base` | required | Base model directory, or a cached HF repo id |
| `--source` | required | Fine-tuned model directory, or a cached HF repo id |
| `--out` | required | Where the task vector is written |

### `inspect`

| Flag | Default | Description |
|---|---|---|
| `--dir` | required | Task-vector or merged-model directory |

---

## YAML recipes

Pass `-c recipe.yaml` to keep a merge configuration in a file. CLI flags
override the YAML values, so a recipe can serve as a baseline with one value
tweaked on the command line.

**Minimal recipe** (only the required fields):

```yaml
base: /path/to/base_model
out: /path/to/merged_model
tv:
  - dir: /path/to/tv_jp
    weight: 0.7
```

**Full recipe**:

```yaml
# --- Required ---
base: /path/to/base_model          # base model (HF-sharded safetensors)
out: /path/to/merged_model         # output directory

# --- Merge inputs (at least one of tv / model) ---
tv:
  - dir: /path/to/tv_jp            # task vector
    weight: 0.7                    # merge weight
  - dir: /path/to/tv_math
    weight: 0.3
# model:
#   - dir: /path/to/source_model   # source model (delta computed on the fly)
#     weight: 0.5

# --- Merge method ---
merge_method: gta                  # gta | linear | mixture | slerp | ties

# --- Consensus ---
consensus: sum                     # none | sum | count

# --- Sparsification ---
density: 1.0                       # 1.0 keeps everything
method: null                       # null | magnitude | random | magnitude_outliers | della_magprune | bs
n: 64                              # --n
m: 256                              # --m
gamma: 0.0                         # --gamma
epsilon: 0.0                       # --epsilon
no_rescale: false                  # true disables norm re-normalization
no_normalize: false                # true disables divisor normalization
lambda: 1.0                        # --lambda

# --- Very large tensors ---
chunk_elements: null               # --chunk-elements

# --- Fingerprint check ---
skip_fingerprint_check: false      # true skips the comparison
```

---

## Merge methods

| `merge_method` | Definition | Notes |
|---|---|---|
| `gta` | Task arithmetic, identical to mergekit GTA | Standard; numerical parity with mergekit |
| `linear` | `base + Σ wᵢ·δᵢ` | Plain weighted sum |
| `mixture` | `base + Σ wᵢ·δᵢ / Σ wᵢ` | Weighted average |
| `slerp` | Spherical interpolation between `base` and `base + Σ wᵢ·δᵢ` | Returns `base + Σ wᵢ·δᵢ` at `t = 1` |
| `ties` | Average over sources whose sign agrees with the majority | TIES-style sign pruning |

Only `gta` has numerical parity with mergekit's method of the same name.
`linear`, `mixture`, `slerp`, and `ties` are GoatMerge's own variants and
differ from mergekit's same-named methods.

### Sparsification methods

Set with `--method`; disabled when `--density 1.0`. Sparsification is applied
to each delta **before** the consensus.

| `method` | Definition | Flags |
|---|---|---|
| `magnitude` | Keep the top `density` fraction by absolute value | — |
| `magnitude_outliers` | Drop the top `gamma` and the bottom, keep the middle | `--gamma` |
| `random` | Keep each element with probability `density` (DARE-style) | — |
| `della_magprune` | Keep probability depends on the within-row magnitude rank | `--epsilon` |
| `bs` | Keep the top `n` of every `m`-element block | `--n`, `--m` |

Stochastic methods (`random`, `della_magprune`) are seeded deterministically
from the tensor name, source, and chunk, so repeated runs and repeated
passes over the same source always produce the same mask.

---

## Design

### Streaming kernel

Per tensor, every delta is streamed exactly once per pass and accumulated as:

```
acc  = Σᵢ αᵢ · δᵢ         (base dtype, in place)
l1   = Σᵢ |αᵢ · δᵢ|      (base dtype, in place; consensus only)
c    = Σᵢ sign(αᵢ·δᵢ)    (int8; count consensus only)
```

The weighted product `δᵢ · αᵢ` is a **bf16·bf16 tensor multiply**, matching the
reference `stacked · weights`. A Python-float scalar `add_(alpha=...)` has
different internal product precision and can flip the per-element majority
sign at near-tie elements.

### Consensus identity

```
mixed = (acc + M · l1) / 2      M = +1 if acc ≥ 0 else −1   (sum)
                               M = +1 if c   ≥ 0 else −1    (count)
```

Elements whose sign disagrees with the majority cancel out of this identity,
so the masked sum never materializes a k-sized mask. The final result is

```
result = base + mixed / divisor        divisor = weight sum of sign-matching TVs
```

`divisor` is per element, so it is computed in a second streamed pass over
the deltas. In chunked mode that second pass re-reads **only the chunk's row
range**, so a full delta is never resident.

### Chunked mode

With `--chunk-elements N` (or `chunk_elements` in YAML), each tensor larger
than N elements is processed in row chunks: load from disk, accumulate, write
into a pre-allocated output buffer. Peak RAM becomes **O(S) + O(chunk)**
(S is the output tensor itself and cannot be avoided).

- Global sparsification (`magnitude`, `magnitude_outliers`) scans the tensor
  once to obtain thresholds and tie budgets in O(1) memory, then applies them
  per chunk
- The consensus divisor pass re-reads only the chunk's rows
- Stochastic masks are seeded per chunk, so both passes agree
- The result is **identical** to the non-chunked path
- `slerp` depends on a global norm and falls back to the non-chunked path
  with a warning

### I/O

The standard HF-sharded safetensors layout is read and written as-is
(`model.safetensors.index.json` + `model-XXXXX-of-NNNNN.safetensors`, or a
single `model.safetensors`). Only one tensor is resident at a time. Shard
headers are parsed once per shard and cached in the `ShardReader`.

Row ranges are read by computing the byte offset from the safetensors header
`data_offsets` — no mmap, and no full-tensor load.

### Fingerprints

A TV stores the fingerprint of the base it was extracted against (tensor
names, shapes, dtypes, and a hash of an anchor tensor). At merge time this is
recomputed and compared, so a TV extracted against a different base cannot
be mixed in by accident.

---

## Peak memory

Measured on a synthetic 100 MB bf16 tensor (2 TVs, consensus=sum):

```
S (largest tensor):  100.0 MB
Merge peak RSS:      105.6 MB
Peak / S:            1.06
```

The measured value beats the 5–7 S worst-case budget because tensors are
freed between passes (`l1` before the divisor pass, each `delta` right after
it is accumulated). Chunked mode lowers the per-tensor peak further to
O(S) + O(chunk).

Rough per-layer numbers:

| Model scale | S (bf16, per layer) | GoatMerge | mergekit GTA |
|---|---|---|---|
| 9B-class | ≈ 1.8 GB | ≈ 2–4 GB | ≈ 22–34 GB |
| 400B-class | ≈ 1.7 GB | ≈ 2 GB | ≈ 34–53 GB (impractical on one GPU) |

---

## Testing

```bash
python -m pytest tests/ -v
```

74 tests cover:

- Streaming merge (no consensus / sum / count, negative weights, missing tensors)
- Numerical parity against mergekit GTA (rtol=2e-2, atol=1e-2)
- Every sparsification method, including tie-heavy tensors
- Kernel unit tests for linear / mixture / slerp / ties
- Chunked-path equivalence across methods, consensus modes, and 1–3-D tensors
- Fingerprint verification
- I/O (sharded, single-shard, submatrix truncation)
- Metadata envelope

---

## Files

```
goatmerge/
  __init__.py      # package init
  cli.py           # CLI (extract / merge / inspect, YAML recipes)
  consensus.py     # ConsensusAccumulator, GtaKernel (streaming kernel)
  extract.py       # task-vector extraction (T = W_source − W_base)
  fingerprint.py   # base fingerprint computation and verification
  hf.py            # local / HF-cache model directory resolution
  inspect.py       # task-vector and merged-model inspection
  io.py            # ShardReader, TensorWriter, ShardedTensorIndex
  kernels.py       # linear / mixture / slerp / ties kernels
  merge.py         # merge_model, merge_tensor (incl. the chunked path)
  merge_method.py  # MergeMethod and build_kernel dispatch
  metadata.py      # goatmerge.json provenance sidecar
  sparsify.py      # sparsification kernels, chunked variants, global masks
tests/
  test_consensus_merge.py   # mergekit GTA parity and merge logic
  test_chunked_merge.py     # chunked-path equivalence
  test_kernels.py           # kernel unit tests
  test_sparsify.py          # sparsification methods
  test_fingerprint.py       # fingerprint verification
  test_io.py                # I/O layer
  test_metadata.py          # metadata sidecar
  measure_peak_ram.py       # peak-RAM measurement
bench_real_models.py        # benchmark on real cached models
Benchmarks/                 # GoatMerge vs mergekit GTA results
```
