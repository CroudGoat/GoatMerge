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
from .io import ShardReader, ShardedTensorIndex, TensorWriter, load_delta, load_delta_chunk
from .merge_method import MergeMethod, build_kernel
from .sparsify import SparsificationMethod, iter_chunk_ranges, sparsify_delta

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
    """
    if not entries:
        return base

    chunked = (
        settings.chunk_elements is not None
        and base.numel() > settings.chunk_elements
    )
    if not chunked:
        # --- Standard path (unchanged) ---
        kernel = build_kernel(base, settings)
        for entry in entries:
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
        kernel = build_kernel(base, settings)
        for entry in entries:
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
            )
            kernel.accumulate(delta, entry.weight)
            del delta
        return kernel.finish(entries, readers, key)

    base_shape = base.shape
    base_dtype = base.dtype
    n_total = base.numel()

    # Pre-allocate the output buffer (O(S) — the result itself).
    out_flat = torch.empty(n_total, dtype=base_dtype, device="cpu")

    for start, end in iter_chunk_ranges(n_total, settings.chunk_elements):
        # Load the base chunk from disk (O(chunk)).
        base_chunk = base[start:end].to(base_dtype)
        kernel = build_kernel(base_chunk, settings)
        for entry in entries:
            delta_chunk = load_delta_chunk(
                base_chunk, entry.dir, entry.kind, key, readers, start, end
            )
            if delta_chunk is None:
                continue
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
            )
            kernel.accumulate(delta_chunk, entry.weight)
            del delta_chunk
        result_chunk = kernel.finish(entries, readers, key)
        out_flat[start:end] = result_chunk
        del result_chunk, base_chunk, kernel

    return out_flat.reshape(base_shape)


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
    writer = TensorWriter(str(out_dir))
    try:
        base_reader = readers[Path(base_dir)]
        for i, key in enumerate(tensor_names):
            # In chunked mode, use a lazy slice (no full O(S) allocation).
            # In non-chunked mode, load the full tensor as before.
            if settings.chunk_elements is not None:
                base = base_reader.get_slice(key)
            else:
                base = base_reader.get_tensor(key).clone()
            result = merge_tensor(base, entries, key, readers, settings)
            writer.save_tensor(key, result)
            logger.info("tensor %d/%d: %s", i + 1, len(tensor_names), key)
    finally:
        for r in readers.values():
            r.close()
        # Flush the writer: staged tensors are only written to disk here.
        writer.finalize()

    return {
        "num_tensors": len(tensor_names),
        "tensor_names": tensor_names,
        "skipped_tensors": skipped,
    }
