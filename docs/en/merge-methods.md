# Merge methods

Selected with `--merge-method` (`MergeMethod`: `gta` | `linear` | `mixture` |
`slerp` | `ties`; dispatched by `merge_method.build_kernel()` with lazy imports
to avoid cycles).

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

## Consensus (`--consensus`)

`ConsensusMethod`: `none` | `sum` | `count`. Applies on top of the kernel
(typically `gta`).

- `none` — no sign filtering.
- `sum` — majority = sign of the weighted sum (`acc >= 0`).
- `count` — majority = sign of the per-element vote count (`c >= 0`).

Mechanically, consensus computes `mixed = (acc + M·l1)/2` with
`M = ±1` from the majority, then `result = base + mixed / divisor`, where
`divisor` is the per-element weight sum of sign-matching TVs
(`divisor[divisor == 0] = 1`). See [streaming-kernel.md](streaming-kernel.md).

## Scale and normalization

- `--lambda` (`MergeSettings.lambda_`): scale applied to the mixed delta.
- `--no-normalize`: disable divisor normalization (`normalize=False`; then no
  sign cache is taken at all).
- `--no-rescale`: disable norm re-normalization after sparsification
  (see [sparsification.md](sparsification.md)).

## Method notes

- `slerp` depends on a global norm, so under `--chunk-elements` it falls back
  to the non-chunked path with a warning.
- `ties` and `gta`+consensus both prune by sign agreement, but their averaging
  denominators differ; see `goatmerge/kernels.py` (`TiesKernel`) and
  `goatmerge/consensus.py` (`GtaKernel`).
- Adding a `MergeKernel`: put the class in `kernels.py` or `consensus.py`,
  extend `MergeMethod` and `build_kernel()`, and export it from
  `goatmerge/__init__.py` (`__all__`).
