"""Task-vector extraction: T = W_source - W_base, saved once as safetensors.

The TV is stored in the base dtype and carries the base fingerprint, so
later merges stream only the deltas (no re-reading of the source model).

Extraction follows the merge engine's memory contract: the source tensor is
subtracted from the base **in place** and both operands are dropped as soon
as the delta is staged, so only one tensor pair is resident at a time
(peak 2 S, S = tensor bytes). With ``chunk_elements`` set, tensors above
that budget are extracted row by row into a single pre-allocated output
buffer, which lowers the peak to O(S) + O(chunk) — the same trade-off as
the Chunk Merge path in ``merge.py``.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional

import torch

from .fingerprint import compute_base_fingerprint
from .io import (
    ShardReader,
    ShardedTensorIndex,
    TensorWriter,
    dtype_from_safetensors,
    load_delta,
    shape_fit,
)
from .metadata import build_metadata, write_metadata
from .sparsify import iter_chunk_ranges

logger = logging.getLogger("goatmerge.extract")


def _numel(shape) -> int:
    n = 1
    for s in shape:
        n *= s
    return n


def _row_chunk_plan(shape, chunk_elements: int) -> int:
    """Rows per chunk so that one row-chunk stays under ``chunk_elements``.

    A row is everything after dim 0 (the product of the trailing dims), which
    is exactly what ``ShardReader`` can read as a contiguous row range.
    """
    n_cols = _numel(shape[1:]) or 1
    return max(1, chunk_elements // n_cols)


def _extract_row_chunked(
    base_slice,
    source_slice,
    base_dtype: torch.dtype,
    chunk_elements: int,
) -> torch.Tensor:
    """Extract ``T = W_source - W_base`` row by row into one output buffer.

    Only valid when base and source have the same shape (the row ranges then
    line up on both shards). The output buffer is allocated once and each
    chunk's rows are written into it, so the peak is the buffer plus one base
    row-chunk and one source row-chunk — never a full source tensor.
    """
    shape = tuple(base_slice.get_shape())
    rows_per_chunk = _row_chunk_plan(shape, chunk_elements)

    full = torch.empty(shape, dtype=base_dtype, device="cpu")
    for start, end in iter_chunk_ranges(shape[0], rows_per_chunk):
        base_chunk = base_slice[start:end]
        # ``.to`` is a no-op (same object) when the dtype already matches, so
        # the subtraction stays in place and no extra copy is made.
        source_chunk = source_slice[start:end].to(base_dtype)
        source_chunk.sub_(base_chunk)
        full[start:end] = source_chunk
        del base_chunk, source_chunk
    return full


def extract_task_vector(
    base_dir: Path,
    source_dir: Path,
    out_dir: Path,
    chunk_elements: Optional[int] = None,
    drop_caches: bool = False,
) -> dict:
    """Extract ``T = W_source - W_base`` into ``out_dir``.

    ``chunk_elements`` enables row-chunked extraction for tensors above that
    many elements (peak O(S) + O(chunk) instead of 2 S per tensor). A tensor
    only qualifies when base and source have the same shape, so row ranges
    read from the two shards stay aligned; submatrix / mismatch cases always
    take the whole-tensor path, where the shared :func:`load_delta` shape
    rules apply.

    ``drop_caches`` releases each file range (and each written shard) from
    the page cache right after use. Every input byte is read exactly once
    here, so unlike the merge consensus path there is no re-read penalty —
    only the fingerprint anchor is read twice.

    Returns a summary dict: tensor names, skipped tensors, base fingerprint.
    """
    base_dir = Path(base_dir)
    source_dir = Path(source_dir)
    out_dir = Path(out_dir)

    logger.info("loading base index: %s", base_dir)
    base_index = ShardedTensorIndex.from_dir(base_dir)
    logger.info("loading source index: %s", source_dir)
    src_index = ShardedTensorIndex.from_dir(source_dir)
    base_reader = ShardReader(base_index, drop_caches=drop_caches)
    src_reader = ShardReader(src_index, drop_caches=drop_caches)
    # The writer plans its shards upfront. Arrival uses the same shape rule
    # as the extraction loop below (see shape_fit), so every planned tensor
    # arrives exactly once: missing or size-incompatible source tensors are
    # excluded here, matching the loop's skips.
    manifest = []
    for key in base_index.keys():
        if key not in src_index.tensor_paths:
            continue
        b_sl = base_reader.get_slice(key)
        s_sl = src_reader.get_slice(key)
        if shape_fit(tuple(b_sl.get_shape()), tuple(s_sl.get_shape())) is None:
            continue
        manifest.append((key, b_sl.get_dtype(), tuple(b_sl.get_shape())))
    writer = TensorWriter(str(out_dir), manifest, drop_caches=drop_caches)
    readers = {base_dir: base_reader, source_dir: src_reader}

    tensor_names: List[str] = []
    skipped: List[str] = []

    keys = base_index.keys()
    logger.info("extracting %d tensors -> %s", len(keys), out_dir)
    try:
        for i, key in enumerate(keys):
            logger.info("tensor %d/%d: %s", i + 1, len(keys), key)
            if key not in src_index.tensor_paths:
                logger.warning("skipping %s (missing in source)", key)
                skipped.append(key)
                continue

            base_slice = base_reader.get_slice(key)
            source_slice = src_reader.get_slice(key)
            base_shape = tuple(base_slice.get_shape())
            source_shape = tuple(source_slice.get_shape())

            # Row-chunked path: same shape, splittable into >1 row, and worth
            # splitting at all. Everything else falls through to the
            # whole-tensor path below.
            if (
                chunk_elements is not None
                and base_shape == source_shape
                and base_shape[0] > 1
                and _numel(base_shape) > chunk_elements
            ):
                base_dtype = dtype_from_safetensors(base_slice.get_dtype())
                delta = _extract_row_chunked(
                    base_slice, source_slice, base_dtype, chunk_elements
                )
                writer.save_tensor(key, delta)
                tensor_names.append(key)
                del delta
                continue

            # Whole-tensor path. The direct readers own their storage, so the
            # defensive clone before the in-place subtraction can be skipped.
            base = base_reader.get_tensor(key)
            delta = load_delta(
                base, source_dir, kind="model", key=key, readers=readers, clone=False
            )
            del base
            if delta is None:
                skipped.append(key)
                continue
            writer.save_tensor(key, delta)
            tensor_names.append(key)
            del delta
    finally:
        base_reader.close()
        src_reader.close()
        # Flush the writer: the last shard file is closed and renamed here
        # (a TensorWriter is not a context manager in this call).
        writer.finalize()

    if not tensor_names:
        raise ValueError("no tensors extracted (all skipped?)")

    logger.info("computing base fingerprint of %s", base_dir)
    fingerprint = compute_base_fingerprint(base_dir, drop_caches=drop_caches)
    meta = build_metadata(
        base_model=str(base_dir),
        source_model=str(source_dir),
        base_fingerprint=fingerprint,
    )
    write_metadata(out_dir, meta)
    logger.info(
        "done: %d extracted, %d skipped -> %s",
        len(tensor_names),
        len(skipped),
        out_dir,
    )

    return {
        "tensor_names": tensor_names,
        "skipped_tensors": skipped,
        "base_fingerprint": fingerprint,
    }
