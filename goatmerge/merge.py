"""Streaming task-arithmetic merge engine (no ``torch.stack``).

Per tensor the engine streams each task vector (or source-model tensor)
exactly once per pass:

    acc  = sum_i alpha_i * delta_i          (in place, base dtype)
    l1   = sum_i |alpha_i * delta_i|       (in place, base dtype)
    c    = sum_i sign(alpha_i * delta_i)   (int8, count consensus only)

Peak resident memory is base + acc + l1 + c + 1 delta + sparsify temp
(~5-7 S, S = tensor bytes) instead of mergekit's (4k+4)S..(6k+7)S.

Parity with mergekit GTA:
  - no consensus:  result = base + lambda * acc / sum(alpha_i)   (normalize)
                   (bit-exact: same sequential add order as the reference)
  - consensus:     mixed = (acc + M*l1)/2  (exact masked-sum identity),
                   divisor recomputed in a second streamed pass.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import torch

from .consensus import ConsensusMethod
from .fingerprint import compute_base_fingerprint
from .io import (
    ShardReader,
    ShardedTensorIndex,
    TensorWriter,
    copy_support_files,
    load_delta,
    load_delta_chunk,
)
from .merge_method import MergeMethod, build_kernel
from .sparsify import (
    GlobalMasker,
    SparsificationMethod,
    global_mask_params,
    iter_chunk_ranges,
    merge_generator,
    sparsify_delta,
)

logger = logging.getLogger("goatmerge.merge")


@dataclass(frozen=True)
class MergeSettings:
    merge_method: MergeMethod = MergeMethod.gta
    density: float = 1.0
    method: Optional[SparsificationMethod] = None
    n: int = 64
    m: int = 256
    gamma: float = 0.0
    epsilon: float = 0.0
    rescale: bool = True
    normalize: bool = True
    lambda_: float = 1.0
    consensus: ConsensusMethod = ConsensusMethod.none
    chunk_elements: Optional[int] = None


@dataclass(frozen=True)
class ModelEntry:
    """One fine-tuned source: a task-vector dir (kind='tv') or a model dir
    (kind='model'), with its merge weight."""

    dir: Path
    kind: str  # "tv" | "model"
    weight: float


def merge_tensor(
    base: torch.Tensor,
    entries: List[ModelEntry],
    key: str,
    readers: dict,
    settings: MergeSettings,
    base_reader=None,
    writer=None,
) -> torch.Tensor:
    """Merge one tensor via the per-method streaming kernel.

    Streams each weighted delta once (sparsified), accumulates into the
    method's kernel, and finalizes. The method is selected by
    ``settings.merge_method`` (see :func:`build_kernel`).

    In chunked mode (``settings.chunk_elements`` set and ``base.numel()``
    exceeds it), the tensor is processed in flat chunks: each chunk is
    loaded from disk, accumulated independently, and written into a
    pre-allocated output buffer via slice assignment. Peak RAM is
    O(S) + O(chunk) instead of O(6-7 S).

    When ``base`` is ``None`` (chunked mode), ``base_reader`` must be
    provided and base chunks are loaded from disk via ``get_slice``.

    Stochastic sparsification (``random`` / ``della_magprune``) is seeded
    from ``(key, source index)`` so every source gets the same mask in the
    consensus second pass as it did in the first.
    """
    if not entries:
        if base is not None:
            return base
        return None

    # Chunked mode requires an explicit row budget; without one there is
    # nothing to chunk by, so fall back to a full-tensor merge.
    if base is None and settings.chunk_elements is None:
        if base_reader is None:
            raise ValueError(
                "merge_tensor requires `base` or `base_reader` when "
                "chunk_elements is not set"
            )
        base = base_reader.get_tensor(key).clone()

    # Determine if we're in chunked mode.
    # If base is None, we're in chunked mode by definition.
    if base is None:
        chunked = True
    else:
        chunked = (
            settings.chunk_elements is not None
            and base.numel() > settings.chunk_elements
        )
    if not chunked:
        # --- Standard path (unchanged) ---
        kernel = build_kernel(base, settings)
        for i, entry in enumerate(entries):
            delta = load_delta(base, entry.dir, entry.kind, key, readers)
            if delta is None:
                continue
            sparsify_delta(
                delta,
                settings.method,
                density=settings.density,
                n=settings.n,
                m=settings.m,
                gamma=settings.gamma,
                epsilon=settings.epsilon,
                rescale=settings.rescale,
                chunk_elements=settings.chunk_elements,
                generator=merge_generator(key, i),
            )
            kernel.accumulate(delta, entry.weight)
            del delta
        return kernel.finish(entries, readers, key)

    # --- Chunked path ---
    if settings.merge_method == MergeMethod.slerp:
        logger.warning(
            "slerp does not support chunked mode; falling back to full tensor"
        )
        # Fall back to the standard path (base is already loaded)
        if base is None:
            base = base_reader.get_tensor(key).clone()
        kernel = build_kernel(base, settings)
        for i, entry in enumerate(entries):
            delta = load_delta(base, entry.dir, entry.kind, key, readers)
            if delta is None:
                continue
            sparsify_delta(
                delta,
                settings.method,
                density=settings.density,
                n=settings.n,
                m=settings.m,
                gamma=settings.gamma,
                epsilon=settings.epsilon,
                rescale=settings.rescale,
                chunk_elements=settings.chunk_elements,
                generator=merge_generator(key, i),
            )
            kernel.accumulate(delta, entry.weight)
            del delta
        return kernel.finish(entries, readers, key)

    # Determine shape/dtype. If base is None (chunked mode), get it from the reader.
    if base is None:
        # Get shape without loading the full tensor.
        slice_obj = base_reader.get_slice(key)
        base_shape = tuple(slice_obj.get_shape())
        # Map dtype string to torch dtype
        dtype_str = slice_obj.get_dtype()
        dtype_map = {'BF16': torch.bfloat16, 'F16': torch.float16, 'F32': torch.float32, 'F64': torch.float64, 'I32': torch.int32, 'I64': torch.int64}
        base_dtype = dtype_map.get(dtype_str, torch.bfloat16)
    else:
        base_shape = base.shape
        base_dtype = base.dtype

    # Row-based chunking: for a (R, C1, ..., Ck) tensor, chunk by rows so that
    # base[start:end] and reader.get_slice(key)[start:end] are contiguous
    # row ranges (no full-tensor load). A "row" is everything after dim 0, so
    # n_cols is the product of the trailing dims (not just shape[1]).
    n_rows = base_shape[0]
    n_cols = 1
    for s in base_shape[1:]:
        n_cols *= s
    rows_per_chunk = max(1, settings.chunk_elements // n_cols)

    # Pre-allocate the full output tensor once (O(S)) and fill each chunk
    # into it; write once at the end. Peak RAM is O(S) (the full tensor)
    # plus O(chunk) for the per-chunk accumulators.
    full = torch.empty(base_shape, dtype=base_dtype, device="cpu")
    ranges = list(iter_chunk_ranges(n_rows, rows_per_chunk))

    def _base_rows(s0: int, e0: int) -> torch.Tensor:
        if base is None:
            return base_reader.get_slice(key)[s0:e0].clone()
        return base[s0:e0].clone()

    # Global-mask sparsification (magnitude / magnitude_outliers) is decided
    # from the whole tensor, so its parameters are scanned once per source
    # (O(1) RAM, straight from disk) and then applied chunk by chunk — in the
    # accumulation pass *and* in the consensus divisor pass, so both use the
    # very same mask.
    mask_params: List[dict] = []
    for entry in entries:
        reader = readers.get(entry.dir) or readers[Path(entry.dir)]
        if (
            settings.method in (SparsificationMethod.magnitude, SparsificationMethod.magnitude_outliers)
            and settings.density < 1
            and key in reader.index.tensor_paths
        ):
            def _chunks(e=entry):
                for s0, e0 in ranges:
                    chunk = load_delta_chunk(
                        _base_rows(s0, e0), e.dir, e.kind, key, readers, s0, e0
                    )
                    if chunk is None:
                        return
                    yield chunk.reshape(-1)

            mask_params.append(
                global_mask_params(
                    settings.method,
                    n_rows * n_cols,
                    settings.density,
                    settings.gamma,
                    settings.rescale,
                    _chunks,
                )
            )
        else:
            mask_params.append({"kind": "local"})
    # one masker for the accumulation pass, one for the divisor pass: the two
    # passes interleave chunk by chunk, so each needs its own running budgets
    masker = GlobalMasker(mask_params)
    masker_divisor = GlobalMasker(mask_params)

    for chunk_idx, (start, end) in enumerate(ranges):
        # base chunk: load from disk (no full-tensor load).
        base_chunk = _base_rows(start, end)
        kernel = build_kernel(base_chunk, settings)
        masker.begin_pass(chunk_idx)
        for i, entry in enumerate(entries):
            delta_chunk = load_delta_chunk(
                base_chunk, entry.dir, entry.kind, key, readers, start, end
            )
            if delta_chunk is None:
                continue
            if not masker.apply(i, delta_chunk):
                # per-chunk mask (random / della / bs): identical in both
                # passes thanks to the deterministic per-chunk generator
                sparsify_delta(
                    delta_chunk,
                    settings.method,
                    density=settings.density,
                    n=settings.n,
                    m=settings.m,
                    gamma=settings.gamma,
                    epsilon=settings.epsilon,
                    rescale=settings.rescale,
                    chunk_elements=None,  # already a small chunk
                    generator=merge_generator(key, i, chunk_idx),
                )
            kernel.accumulate(delta_chunk, entry.weight)
            del delta_chunk
        # the consensus divisor pass re-reads only this chunk's rows
        result_chunk = kernel.finish(
            entries, readers, key, start=start, end=end, chunk=chunk_idx, masker=masker_divisor
        )
        full[start:end] = result_chunk
        del result_chunk, base_chunk, kernel
    # Return the full merged tensor (the caller writes it), matching the
    # non-chunked path's contract. Peak RAM is O(S) (the full tensor) plus
    # O(chunk) for the per-chunk accumulators.
    return full


def merge_model(
    base_dir: Path,
    entries: List[ModelEntry],
    settings: MergeSettings,
    out_dir: Path,
) -> dict:
    """Merge a whole model directory, streaming tensor by tensor.

    Returns a summary dict (tensor count, skipped tensors, elapsed).
    """
    base_index = ShardedTensorIndex.from_dir(Path(base_dir))
    tensor_names = base_index.keys()
    if not tensor_names:
        raise ValueError(f"No tensors found in {base_dir}")

    readers: dict = {}
    readers[Path(base_dir)] = ShardReader(base_index)
    for entry in entries:
        readers[Path(entry.dir)] = ShardReader(ShardedTensorIndex.from_dir(Path(entry.dir)))

    skipped: List[str] = []
    # flush_after_each=False: tensors accumulate in the current shard and are
    # grouped into proper HF shards (max_shard_size) at finalize(). Per-tensor
    # flushing would emit one shard per tensor (wrong layout).
    writer = TensorWriter(str(out_dir))
    try:
        base_reader = readers[Path(base_dir)]
        for i, key in enumerate(tensor_names):
            # In chunked mode, don't load the full base tensor.
            # Instead, pass base=None and let merge_tensor load chunks from disk.
            chunked = (
                settings.chunk_elements is not None
            )
            if chunked:
                base = None
            else:
                base = base_reader.get_tensor(key).clone()
            result = merge_tensor(base, entries, key, readers, settings, base_reader, writer=writer)
            if result is not None:
                writer.save_tensor(key, result)
            logger.info("tensor %d/%d: %s", i + 1, len(tensor_names), key)
    finally:
        for r in readers.values():
            r.close()
        # Flush the writer: staged tensors are only written to disk here.
        writer.finalize()

    # Merge only rewrites weights: carry the base model's config / tokenizer
    # files over so the output is directly loadable by transformers.
    copied = copy_support_files(base_dir, out_dir)

    return {
        "num_tensors": len(tensor_names),
        "tensor_names": tensor_names,
        "skipped_tensors": skipped,
        "base_fingerprint": compute_base_fingerprint(Path(base_dir)),
        "support_files": copied,
    }
