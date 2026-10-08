# GoatMerge

![GoatMerge](GoatMerge.png)

[日本語版はこちら](README-ja.md)

Streaming task-arithmetic merge engine. Merges fine-tuned models (task vectors)
into a base model with **no `torch.stack`**, peak RAM ≈ **5–7 S** (S = largest
tensor bytes), and **numerical parity** with mergekit's Generalized Task
Arithmetic (GTA).

## Key properties

| Property | GoatMerge | mergekit GTA |
|---|---|---|
| Stacks all deltas? | No — streams one delta at a time | Yes (`torch.stack`) |
| Peak RAM (k TVs, 1 tensor) | ≈ 5–7 S | (4k+4) S – (6k+7) S |
| Consensus (masked sum) | Exact identity `(acc + M·l1)/2` | `stacked · weights` then mask |
| Sparsify | Before consensus (per delta) | Before consensus |
| I/O | HF-sharded safetensors | HF-sharded safetensors |

## Comparison with mergekit

### Memory (measured, 100 MB bf16 tensor, 2 TVs, consensus=sum)

| | GoatMerge | mergekit GTA (theoretical) |
|---|---|---|
| Peak RSS | **105.6 MB** (1.06 × S) | **1.2 – 1.9 GB** (12 S – 19 S) |
| Stacked tensor | Never materialized | `k × S` (200 MB for k=2) |
| Per-delta temporaries | 1 delta at a time (100 MB) | All k deltas resident (200 MB) |
| Accumulator bookkeeping | acc + l1 + c = 2.5 S | stacked + weighted + mask ≈ 3 S |

For a 9 B-class model (S ≈ 1.8 GB per layer), GoatMerge peaks at
≈ 2–4 GB per layer; mergekit peaks at ≈ 22–34 GB.

### Speed

GoatMerge streams each delta exactly once per pass and accumulates in place.
There is no `torch.stack` allocation, no `stacked · weights` full-tensor
multiply, and no second pass over all k deltas for the mask. The only
second pass is the per-element `divisor` (consensus only), which is a
single `add_` per delta.

| Operation | GoatMerge | mergekit GTA |
|---|---|---|
| Load k deltas | k × (1 load) | k × (1 load) + 1 stack |
| Weighted sum | k × (in-place `add_`) | 1 × (full `stacked · weights`) |
| Mask (consensus) | in-place identity | 1 × (full mask multiply) |
| Divisor | k × (in-place `add_`) | 1 × (full `weights · mask`) |

The in-place accumulation avoids the O(k·S) temporary allocation that
`torch.stack` + `stacked · weights` requires, and the consensus identity
`(acc + M·l1)/2` replaces a full-tensor mask multiply with two in-place
`add_`/`mul_` operations.

### Numerical parity

GoatMerge matches mergekit GTA within **rtol = 2e-2, atol = 1e-2** on
bf16 tensors. The weighted product `δᵢ · αᵢ` is a bf16·bf16 tensor
multiply (matching the reference `stacked · weights`), not a Python-float
scalar `add_` — whose internal product precision diverges at near-tie
elements.

## Installation

```bash
pip install -e .
```

Requires Python ≥ 3.10, PyTorch, and `safetensors`.

## CLI

```bash
# Merge two TV dirs into a base model
goatmerge merge \
  --base /path/to/base_model \
  --tv /path/to/tv1 --weight 0.5 \
  --tv /path/to/tv2 --weight 0.7 \
  --out /path/to/output \
  --consensus sum \
  --normalize

# Extract task vectors from a model pair
goatmerge extract \
  --base /path/to/base \
  --model /path/to/finetuned \
  --out /path/to/tv_dir
```

### YAML recipe

Pass a YAML file with `-c` instead of typing every flag on the command line:

```bash
goatmerge merge -c recipe.yaml
```

**Minimal recipe** (the only required fields):

```yaml
base: /path/to/base_model
out: /path/to/merged_output
tv:
  - dir: /path/to/tv1
    weight: 0.7
```

That's all you need. Everything else is optional and falls back to its
default if omitted.

**Full recipe** (every field, annotated):

```yaml
# --- Required ---
base: /path/to/base_model        # base model directory (HF-sharded safetensors)
out: /path/to/merged_output      # where the merged model is written

# --- Task vectors (at least one) ---
tv:
  - dir: /path/to/tv1            # task-vector directory
    weight: 0.7                  # merge weight for this TV
  - dir: /path/to/tv2
    weight: 0.3

# --- Source models (alternative to tv; at least one of tv/model) ---
# model:
#   - dir: /path/to/source_model
#     weight: 0.5

# --- Merge method ---
merge_method: gta                # gta | linear | mixture | slerp | ties  (default: gta)

# --- Consensus (masked-sum) ---
consensus: sum                   # none | sum | count  (default: none)

# --- Sparsification ---
density: 1.0                     # 0 = skip, 1.0 = keep all (default: 1.0)
method: null                     # null = no sparsify; l1 | l2 | gamma | topk
n: 64                            # top-k count (default: 64)
m: 256                           # block size (default: 256)
gamma: 0.0                       # gamma threshold (default: 0.0)
epsilon: 0.0                     # epsilon floor (default: 0.0)
rescale: true                    # rescale norm after sparsify (default: true)

# --- Normalization & scaling ---
normalize: true                  # divide by per-element divisor (default: true)
lambda: 1.0                      # scale factor on the mixed tensor (default: 1.0)

# --- Chunked mode (for very large tensors) ---
chunk_elements: null             # split tensor into chunks of this many elements

# --- Fingerprint ---
skip_fingerprint_check: false    # skip base fingerprint verification
```

CLI flags override the corresponding YAML values, so you can set a recipe
as a baseline and tweak one value on the command line without editing the
file.

### Options

| Flag | Default | Description |
|---|---|---|
| `--base` | (required) | Base model directory (HF-sharded safetensors) |
| `--tv` / `--model` | (≥1) | Task-vector dir(s) or source model dir(s) |
| `--weight` | 1.0 | Merge weight per entry |
| `--out` | (required) | Output directory |
| `--merge-method` | `gta` | Merge algorithm: `gta` \| `linear` \| `mixture` \| `slerp` \| `ties` |
| `--consensus` | `none` | `none` \| `sum` \| `count` |
| `--normalize` | true | Divide by per-element divisor |
| `--lambda` | 1.0 | Scale factor on the mixed tensor |
| `--density` | 1.0 | Sparsify density (0 = skip) |
| `--method` | (none) | Sparsify method: `l1` \| `l2` \| `gamma` \| `topk` |
| `--n` | 64 | Top-k count for sparsify |
| `--m` | 256 | Block size for sparsify |
| `--gamma` | 0.0 | Gamma threshold |
| `--epsilon` | 0.0 | Epsilon floor |
| `--rescale` | true | Rescale norm after sparsify |
| `--chunk-elements` | null | Split tensor into chunks of this size (chunked mode) |

## Architecture

### Streaming kernel

Per tensor, the engine streams each delta exactly once per pass:

```
acc  = Σᵢ αᵢ · δᵢ        (in place, base dtype)
l1   = Σᵢ |αᵢ · δᵢ|     (in place, base dtype)
c    = Σᵢ sign(αᵢ·δᵢ)   (int8, count method only)
```

The weighted product `δᵢ · αᵢ` is a **bf16·bf16 tensor multiply** (matching
the reference `stacked · weights`), not a Python-float scalar `add_` — whose
internal product precision diverges at near-tie elements and flips the
per-element majority sign.

### Consensus identity

```
mixed = (acc + M · l1) / 2,   M = +1 if (acc|c) ≥ 0 else −1
```

Zeros contribute 0 in both forms, matching the reference mask which drops
sign-0 elements. Only `divisor` (per-element weight sum over sign-matching
TVs) requires a second streamed pass.

### Sparsify

Applied **before** consensus, per delta. Uses `torch.topk` (not `argsort`)
and widens bf16/fp16 to f32 for topk on CPU.

### I/O

HF-sharded safetensors layout (`model.safetensors.index.json` +
`model-XXXXX-of-NNNNN.safetensors`). Single shard → `model.safetensors`.
Only one tensor is resident at a time; `get_tensor` returns a view that
shares storage with the shard's cached copy — the caller must `.clone()`
before mutating.

## Peak RAM

Measured on a synthetic 100 MB bf16 tensor (2 TVs, consensus=sum):

```
S (largest tensor):  100.0 MB
Merge peak RSS:      105.6 MB
Peak / S:            1.06  (target: 5–7)
```

The streaming kernel is more memory-efficient than the 5–7 S worst-case
budget because tensors are freed between passes (`l1` before `divisor`,
each `delta` after accumulation).

### Chunked mode (large tensors)

For tensors larger than `chunk_elements`, the merge is split into flat
chunks of `chunk_elements` elements. Each chunk is loaded from disk,
accumulated independently, and written into a pre-allocated output buffer
via slice assignment. Peak RAM drops to **O(S) + O(chunk)** instead of
O(6–7 S):

- `base` is a lazy slice (no full O(S) allocation)
- Per-chunk: `base_chunk`, `delta_chunk`, `acc`/`l1`/`c` are O(chunk)
- `out_flat` is O(S) (unavoidable — it IS the result)

Set `chunk_elements` in the YAML recipe or `--chunk-elements` on the CLI.
Slerp does not support chunked mode (global norm dependency) and falls
back to the non-chunked path with a warning.

## Testing

```bash
python -m pytest tests/ -v
```

61 tests cover:
- Streaming merge (no-consensus, consensus sum/count)
- Sparsify (l1, l2, gamma, top-k)
- Fingerprint verification
- I/O (sharded, single-shard, submatrix truncation)
- Metadata envelope
- Numerical parity vs mergekit GTA (rtol=2e-2, atol=1e-2)
- Chunked merge parity + sparsify validity

## Files

```
goatmerge/
  __init__.py      # package
  cli.py            # CLI entry point
  consensus.py      # ConsensusAccumulator (streaming kernel)
  extract.py        # TV extraction from model pairs
  fingerprint.py    # Fingerprint verification
  hf.py             # HF model-dir helpers
  inspect.py        # Model inspection
  io.py              # ShardReader, TensorWriter, ShardedTensorIndex
  kernels.py         # Per-method merge kernels
  merge.py            # merge_model, merge_tensor (incl. chunked path)
  merge_method.py    # MergeMethod enum + build_kernel dispatch
  metadata.py         # Metadata envelope
  sparsify.py         # Sparsify kernels + chunked variants
tests/
  test_consensus_merge.py   # Parity + merge tests
  test_chunked_merge.py     # Chunked merge parity + validity
  test_fingerprint.py
  test_io.py
  test_kernels.py           # Kernel unit tests
  test_metadata.py
  test_sparsify.py
  measure_peak_ram.py       # Peak-RAM measurement
```
