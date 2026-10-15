# Architecture & invariants

## Modules

| File | Responsibility |
|---|---|
| `merge.py` | `merge_model` / `merge_tensor` / `MergeSettings` / `ModelEntry`. Per-tensor merge logic, Chunk Merge branch. |
| `merge_method.py` | `MergeMethod` enum, `MergeKernel` base class, `build_kernel()` dispatch (lazy per-kernel imports to avoid cycles). |
| `consensus.py` | `ConsensusAccumulator` / `ConsensusMethod` / `GtaKernel`. Streaming accumulation kernel; `acc`/`l1`/`c`/`nz` updated in place. Signs cached 2-bit-packed (`_pack_signs` / `_unpack_match`); divisor assembled without re-read. |
| `kernels.py` | Non-GTA kernels: `LinearKernel` / `MixtureKernel` / `SlerpKernel` / `TiesKernel`. |
| `sparsify.py` | Sparsification kernels (`magnitude`, `random`, `magnitude_outliers`, `della_magprune`, `bs`). Streaming exact selection for global thresholds (`_select_threshold`, O(1) RAM, bit-identical to sort), `uint8` masks, chunked / global masks. `bernoulli` draws in place. Default chunking via `_DEFAULT_SPARSE_CHUNK_ELEMENTS` (8M). |
| `io.py` | `ShardedTensorIndex` / `ShardReader` / `TensorWriter` / `load_delta` / `load_delta_chunk` / `copy_support_files` / `dtype_from_safetensors` / `safetensors_dtype_str` / `shape_fit` / `advise_drop_caches`. Sharded safetensors reads/writes. Single-buffer `readinto` row reads; opt-in `POSIX_FADV_DONTNEED`. Pre-committed arrival-order `TensorWriter` (no staging). |
| `extract.py` | `extract_task_vector` — `T = W_source - W_base`. In-place subtraction (2 S) reusing `load_delta(kind="model")`; row-chunked extraction with `chunk_elements` at O(S) + O(chunk). Bit-identical to the standard path. |
| `fingerprint.py` | Base fingerprint (`compute_base_fingerprint` / `model_manifest` / `tensor_content_hash`) and verification. Anchor tensor hashed with row-chunked streaming (O(chunk)). |
| `metadata.py` | `goatmerge.json` envelope (`build_merged_metadata` / `write_metadata` / `load_metadata` / `validate_metadata`). |
| `inspect.py` | `inspect_dir` and `verify_against_base`. |
| `hf.py` | `resolve_model_dir` / `is_repo_id` — local paths and HF cache resolution. |
| `cli.py` | CLI entry point. `extract` / `merge` / `inspect`, YAML recipes (`-c`). `extract` takes `--chunk-elements`; both take `--drop-caches` (`merge` also via YAML `drop_caches`). |
| `__init__.py` | Public API re-exports. |

### Enum values (easy to mistype in signatures)

- `MergeMethod`: `gta` | `linear` | `mixture` | `slerp` | `ties`
- `ConsensusMethod`: `none` | `sum` | `count`
- `SparsificationMethod`: `magnitude` | `random` | `magnitude_outliers` | `della_magprune` | `bs`
- `RescaleNorm`: `l1` | `l2` | `linf`
- `MergeSettings` fields: `merge_method, density, method, n, m, gamma, epsilon, rescale, normalize, lambda_, consensus, chunk_elements, drop_caches`
- `ModelEntry`: `(dir: Path, kind: "tv" | "model", weight: float)`

## Data flow (per tensor)

1. Stream each delta once: `load_delta(..., clone=False)` → `sparsify_delta`
   (before consensus, in place) → bf16·bf16 weighting → kernel `accumulate`.
2. `finish(base)`: consensus identity `(acc + M·l1)/2`, divisor from the sign
   cache, `base + mixed / divisor`, `lambda_` scaling, dtype round-trip.
3. Chunk Merge: same kernels per row chunk into a pre-allocated `full` buffer.
4. `TensorWriter` appends the finished tensor bytes in arrival order.

## Invariants (do not break)

1. **Weighted product is a bf16·bf16 tensor multiply.**
   `delta.mul_(torch.tensor(alpha, dtype=delta.dtype))`. A Python-scalar
   `add_(alpha=...)` diverges in internal product precision from the reference
   (`stacked * weights`) and flips majority signs at near-tie elements.
2. **Consensus identity** `mixed = (acc + M·l1)/2` (`M = sign(majority)`).
   Never re-derive the masked sum in a second pass; assemble `divisor` from the
   cached 2-bit signs (`_pack_signs`, k·S/4). No delta re-read. With
   `normalize=False` no sign cache is taken.
3. **Sparsify before consensus**, exactly once per delta (no re-application for
   the divisor — it no longer exists).
4. **Deterministic masks** — stochastic masks (`random` / `della_magprune`) use
   `merge_generator(key, source_index, chunk)` = `(tensor name, source, chunk)`.
5. **Rescale sums widen to float32** (`sparsify._norm`) — the chunked path
   reproduces the non-chunked path bit-identically only because of this.
6. **Chunk Merge equals the standard path.**
7. **Row reads return owned storage** — `_read_rows` returns a tensor owning its
   `readinto` buffer; reads never alias. Never mutate an mmap view in place.
   `load_delta(..., clone=False)` assumes the direct reader's owned storage.
   All merge loads use `clone=False`.
8. **Extraction equality** — row-chunked / in-place subtraction is
   bit-identical to `s.to(dtype) - b`. Shape mismatches (submatrix / mismatch)
   always fall back to the non-chunked path; shape decisions live in
   `shape_fit`.
9. **`--drop-caches` is opt-in** — `POSIX_FADV_DONTNEED` for just-read ranges
   and just-written shards. Each tensor is read exactly once, so there is no
   re-read penalty. The writer never `fdatasync`s (speed-first).
10. **Arrival-order writer** — `TensorWriter` pre-commits sharding/offsets/headers
    from the manifest, appends bytes on arrival, holds no staging or `save_file`
    buffer. Off-plan arrivals / size mismatches raise. Incomplete shards are
    deleted. Headers are canonical sorted order and deterministic.
11. **Default sparsify chunking** — even without `chunk_elements`, mask
    computation splits at `_DEFAULT_SPARSE_CHUNK_ELEMENTS` (8M). Deterministic
    methods are bit-identical; stochastic ones are deterministic for the same
    seed (torch CPU RNG is shape-dependent, so whole-tensor draws may differ
    bitwise — both paths still agree with each other).

Tests guarding these: `tests/test_consensus_merge.py` (parity),
`tests/test_chunk_merge.py` (chunk equivalence),
`tests/test_io.py::test_reader_clone_semantics` (read ownership).
