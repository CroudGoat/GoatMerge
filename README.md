# GoatMerge

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
