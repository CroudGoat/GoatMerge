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

Instead of passing every flag on the command line, write a YAML recipe file
and pass it with `-c`:

```bash
goatmerge merge -c recipe.yaml
```

A recipe file supplies `base`, `out`, `tv`/`model` entries, and all tuning
parameters. Any flag you pass on the command line overrides the corresponding
YAML value.

```yaml
# recipe.yaml
base: /path/to/base_model
out: /path/to/merged_output

tv:
  - dir: /path/to/tv1
    weight: 0.7
  - dir: /path/to/tv2
    weight: 0.3

# model:
#   - dir: /path/to/source_model
#     weight: 0.5

consensus: sum
density: 1.0
method: null        # null = no sparsification
n: 64
m: 256
gamma: 0.0
epsilon: 0.0
rescale: true
normalize: true
lambda: 1.0
chunk_elements: null
skip_fingerprint_check: false
```

Only the fields you want to set need to appear; omitted fields fall back to
their defaults. `tv` and `model` entries accept either a bare `dir:weight`
string or a `dir` + `weight` mapping.

### Options

| Flag | Default | Description |
|---|---|---|
| `--base` | (required) | Base model directory (HF-sharded safetensors) |
| `--tv` / `--model` | (≥1) | Task-vector dir(s) or source model dir(s) |
| `--weight` | 1.0 | Merge weight per entry |
| `--out` | (required) | Output directory |
| `--consensus` | `none` | `none` \| `sum` \| `count` |
| `--normalize` | true | Divide by per-element divisor |
| `--lambda` | 1.0 | Scale factor on the mixed tensor |
| `--density` | 1.0 | Sparsify density (0 = skip) |
| `--method` | (none) | Sparsify method: `l1` \| `l2` \| `gamma` |
| `--n` | 64 | Top-k count for sparsify |
| `--m` | 256 | Block size for sparsify |
| `--gamma` | 0.0 | Gamma threshold |
| `--epsilon` | 0.0 | Epsilon floor |
| `--rescale` | true | Rescale norm after sparsify |

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

## Testing

```bash
python -m pytest tests/ -v
```

43 tests cover:
- Streaming merge (no-consensus, consensus sum/count)
- Sparsify (l1, l2, gamma, top-k)
- Fingerprint verification
- I/O (sharded, single-shard, submatrix truncation)
- Metadata envelope
- Numerical parity vs mergekit GTA (rtol=2e-2, atol=1e-2)

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
  merge.py            # merge_model, merge_tensor
  metadata.py         # Metadata envelope
  sparsify.py         # Sparsify kernels
tests/
  test_consensus_merge.py   # Parity + merge tests
  test_fingerprint.py
  test_io.py
  test_metadata.py
  test_sparsify.py
  measure_peak_ram.py       # Peak-RAM measurement
```
