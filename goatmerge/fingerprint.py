"""Fingerprinting for task-vector / base-model integrity verification.

A *base fingerprint* is a content hash that identifies a concrete base model
(not just its architecture). A task vector stores the fingerprint of the base
it was extracted against; at merge time the fingerprint of the provided base
is recomputed and compared. A mismatch means the task vector was built
against a different base and must be rejected.

The fingerprint is cheap: it hashes the tensor manifest (name, shape, dtype)
plus the raw bytes of one anchor tensor — O(1 tensor) of I/O. That I/O is also
bounded: the anchor tensor is hashed **row by row** straight from the shard, so
its bf16 → float32 widening and byte materialization never touch more than one
chunk at a time (ISSUE.md 3.5).
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import List, Tuple

import torch

from .io import ShardReader, ShardedTensorIndex
from .sparsify import iter_chunk_ranges

# Elements per hashed row-range. bfloat16/float16 are widened to float32 before
# hashing, so this bounds the transient to ~4 bytes/element (bf16 -> f32 copy
# plus the byte buffer feeding the hasher).
_HASH_CHUNK_ELEMENTS = 8_000_000


def _update_hash(hasher: "hashlib._Hash", tensor: torch.Tensor) -> None:
    """Feed a contiguous tensor's values into ``hasher``, widening bf16/fp16.

    The widening is lossless (bf16/fp16 values are exactly representable in
    float32), so the digest matches the hash of an equivalent float32 tensor.
    """
    chunk = tensor
    if chunk.dtype in (torch.float16, torch.bfloat16):
        chunk = chunk.to(torch.float32)
    hasher.update(chunk.contiguous().numpy().tobytes())


def tensor_content_hash(tensor, chunk_elements: int = _HASH_CHUNK_ELEMENTS) -> str:
    """SHA-256 over a tensor's values, in a dtype numpy can represent.

    BF16/FP16 have no numpy dtype, so they are widened to float32 (lossless)
    before hashing. The resulting bytes uniquely determine the original values,
    so the hash is a valid content fingerprint.

    The hash is taken over row ranges, so the widening and the byte buffer stay
    bounded by ``chunk_elements``; chunk boundaries never split an element and
    the concatenation order is the tensor's natural order, so the digest is
    identical to hashing the whole tensor at once.
    """
    hasher = hashlib.sha256()
    flat = tensor.reshape(-1)
    for start, end in iter_chunk_ranges(flat.numel(), chunk_elements):
        _update_hash(hasher, flat[start:end])
    return hasher.hexdigest()


def _hash_tensor_rows(
    reader: ShardReader,
    key: str,
    chunk_elements: int = _HASH_CHUNK_ELEMENTS,
) -> str:
    """SHA-256 over ``key`` read row by row from a shard (no full load).

    Rows concatenate to the tensor's natural order, so this digest equals
    :func:`tensor_content_hash` on the fully loaded tensor.
    """
    sl = reader.get_slice(key)
    shape = tuple(sl.get_shape())
    if not shape:
        # 0-d tensors have no row axis to chunk by; the single read is O(1).
        return tensor_content_hash(reader.get_tensor(key))

    n_cols = 1
    for s in shape[1:]:
        n_cols *= s
    rows_per_chunk = max(1, chunk_elements // n_cols)

    hasher = hashlib.sha256()
    for start, end in iter_chunk_ranges(shape[0], rows_per_chunk):
        _update_hash(hasher, sl[start:end])
    return hasher.hexdigest()


def model_manifest(index: ShardedTensorIndex) -> List[Tuple[str, Tuple[int, ...], str]]:
    """(name, shape, dtype) for every tensor, using only shard metadata."""
    reader = ShardReader(index)
    try:
        manifest = []
        for key in index.keys():
            sl = reader.get_slice(key)
            manifest.append((key, tuple(sl.get_shape()), sl.get_dtype()))
        return manifest
    finally:
        reader.close()


def compute_base_fingerprint(
    model_dir: Path,
    chunk_elements: int = _HASH_CHUNK_ELEMENTS,
    drop_caches: bool = False,
) -> str:
    """Fingerprint a model directory: manifest + anchor-tensor content hash.

    The anchor is the first tensor in sorted name order, so the result is
    deterministic for a given model directory. The anchor is streamed from the
    shard in row chunks (``chunk_elements``), so the fingerprint costs
    O(chunk) resident memory even for a multi-gigabyte anchor tensor.
    ``drop_caches`` additionally releases the anchor rows from the page cache
    as they are hashed.
    """
    index = ShardedTensorIndex.from_dir(Path(model_dir))
    keys = index.keys()
    if not keys:
        raise ValueError(f"No tensors found in {model_dir}")

    manifest = model_manifest(index)
    reader = ShardReader(index, drop_caches=drop_caches)
    try:
        anchor_hash = _hash_tensor_rows(reader, keys[0], chunk_elements)
    finally:
        reader.close()

    payload = json.dumps(manifest, sort_keys=True) + anchor_hash
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def tensor_names_and_shapes(model_dir: Path) -> dict:
    """Map of tensor name -> (shape, dtype) for a model directory."""
    index = ShardedTensorIndex.from_dir(Path(model_dir))
    manifest = model_manifest(index)
    return {name: (shape, dtype) for (name, shape, dtype) in manifest}
