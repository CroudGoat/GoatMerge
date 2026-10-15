# I/O — sharded safetensors

Standard HF-sharded safetensors layout is read and written as-is
(`model.safetensors.index.json` + `model-XXXXX-of-NNNNN.safetensors`, or a
single `model.safetensors`). Only one tensor is resident at a time.

Module: `goatmerge/io.py` — `ShardedTensorIndex`, `ShardReader`,
`TensorWriter`, `load_delta`, `load_delta_chunk`, `copy_support_files`,
`dtype_from_safetensors`, `safetensors_dtype_str`, `shape_fit`,
`advise_drop_caches`.

## Reading

- Shard headers are parsed once per shard and cached in the `ShardReader`.
- Row ranges are read by computing the byte offset from the safetensors header
  `data_offsets` — no mmap, no full-tensor load.
- `_read_rows` performs **one** `readinto` into a single freshly allocated
  buffer and returns a tensor owning that buffer. Each read owns an independent
  buffer, so there is no aliasing between reads
  (`tests/test_io.py::test_reader_clone_semantics` enforces this).
- Never mutate an mmap view in place. `load_delta(..., clone=False)` assumes
  the direct reader's already-owned storage; there is no mmap reader in the
  codebase. All merge loads use `clone=False`.
- Shape handling is centralized in `shape_fit` and shared by `_prepare_delta`
  and the extract arrival planner: mismatches (submatrix / mismatch) always
  fall back to the non-chunked path under the shared `load_delta` rules.

## Writing (`TensorWriter`)

Arrival-order streaming writer, pre-committed from the manifest (names, dtypes,
shapes → shard split, offsets, headers):

- Appends bytes as tensors arrive. No staging dict, no `save_file` buffer.
- Off-plan arrivals or size mismatches raise (never silently corrupt).
- Incomplete shards are deleted (no broken files left behind).
- Headers are canonical sorted order and deterministic (byte-equality with
  `save_file` is *not* claimed since its key order is implementation-defined).

## `--drop-caches` (opt-in)

`POSIX_FADV_DONTNEED` is issued for just-read ranges and just-written shards.
Because each tensor is read exactly once (divisor re-read was eliminated via
the sign cache), there is no re-read penalty in the current design.

The writer side does not `fdatasync` (does not wait for dirty pages), so
residual GBs from writeback are possible — a speed-first decision
(see `io._close_shard` comment). Measured on a 9.3 GB model run:
96 s / +17.2 GB cache without the flag vs **45 s / +2.7 GB** with it, same
2.3 GB process peak.

Covered by `tests/test_io.py` (sharded / single-shard / row slices, writer
naming and sharding, drop-caches advise).
