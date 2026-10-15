# Sparsification

Set with `--method`; disabled when `--density 1.0`. Sparsification is applied
to each delta **before** the consensus, exactly once per delta
(see invariant #3 in [architecture.md](architecture.md)).

| `method` | Definition | Flags |
|---|---|---|
| `magnitude` | Keep the top `density` fraction by absolute value | — |
| `magnitude_outliers` | Drop the top `gamma` and the bottom, keep the middle | `--gamma` |
| `random` | Keep each element with probability `density` (DARE-style) | — |
| `della_magprune` | Keep probability depends on the within-row magnitude rank | `--epsilon` |
| `bs` | Keep the top `n` of every `m`-element block | `--n`, `--m` |

Implementation (`goatmerge/sparsify.py`):

- `magnitude` / `magnitude_outliers` use `torch.topk` (k indices only) instead
  of a full `argsort`; global thresholds use streaming exact selection
  (`_select_threshold`, O(1) RAM, bit-identical to sorting).
- All masks are `uint8` (1 byte/element); application and rescale are in place.
- `bernoulli` draws in place (`bernoulli_`, no `full_like` probability tensor).
- Rescale sums are widened to float32 (`_norm`), which is what lets the chunked
  path reproduce the non-chunked result bit-identically.
- Mask computation is **always chunked** with a default budget
  `_DEFAULT_SPARSE_CHUNK_ELEMENTS` (8M elements) even when `chunk_elements` is
  unset. Deterministic methods are bit-identical; stochastic methods are
  deterministic for the same seed (torch CPU RNG is shape-dependent, so an
  explicit chunking may differ bitwise from a whole-tensor draw, but both
  paths agree with each other).

## Deterministic masks

Stochastic methods (`random`, `della_magprune`) are seeded deterministically
from `(tensor name, source index, chunk)` via
`sparsify.merge_generator(key, source_index, chunk)`, so repeated runs — and
repeated passes over the same source — always produce the same mask.

## Global vs chunked masks

- Global methods (`magnitude`, `magnitude_outliers`) scan the whole tensor once
  to fix thresholds/tie budgets (`GlobalMasker(mask_params)`), then apply the
  same mask per chunk. Both paths use the same mask.
- Chunk Merge equivalence across methods, consensus modes, and 1–3-D tensors is
  enforced by `tests/test_chunk_merge.py`; per-method correctness by
  `tests/test_sparsify.py` and `tests/test_select_threshold.py`.
