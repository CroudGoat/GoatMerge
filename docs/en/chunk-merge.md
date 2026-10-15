# Chunk Merge

**Chunk Merge** is the chunked merge path. With `--chunk-elements N`
(or `chunk_elements` in YAML), each tensor larger than N elements is processed
in row chunks (dim 0 rows = product of trailing dims): load from disk,
accumulate, write into a pre-allocated output buffer.

- Output is allocated **once** as `full = torch.empty(base_shape)`; each chunk
  result fills `full[start:end]`.
- Peak RAM becomes **O(S) + O(chunk)** (S, the output tensor itself, cannot be
  avoided).
- Global sparsification (`magnitude`, `magnitude_outliers`) scans the tensor
  once to obtain thresholds and tie budgets in O(1) memory, then applies them
  per chunk via `GlobalMasker(mask_params)` — both paths use the same mask.
- The consensus divisor needs **no re-read** (sign cache), so each tensor is
  read exactly once. Stochastic masks are seeded per chunk
  (`merge_generator(key, source, chunk)`), so all passes agree.
- The result is **identical** to the standard path (same kernels, same masks).
- `slerp` depends on a global norm and falls back to the non-chunked path with
  a warning.
- `extract --chunk-elements` uses the same row-chunk scheme (single `full`
  buffer, slice assignment). Only shape-matching tensors take the chunked path;
  mismatches (submatrix / mismatch) fall back to the non-chunked path under the
  shared `load_delta` rules. Shape decisions are centralized in `shape_fit`.

Row reads (`io._read_rows`) allocate exactly one buffer per `readinto` and
return a tensor owning that buffer — no `bytes → bytearray → clone` triple
copy, no aliasing between reads.

Equivalence is enforced by `tests/test_chunk_merge.py` (methods × consensus ×
1–3-D tensors, sparsify validity) and `tests/test_consensus_merge.py`
(chunked-extract equality and fallback).
