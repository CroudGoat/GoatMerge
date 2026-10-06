"""Fingerprinting for task-vector / base-model integrity verification.

A *base fingerprint* is a content hash that identifies a concrete base model
(not just its architecture). A task vector stores the fingerprint of the base
it was extracted against; at merge time the fingerprint of the provided base
is recomputed and compared. A mismatch means the task vector was built
against a different base and must be rejected.

The fingerprint is cheap: it hashes the tensor manifest (name, shape, dtype)
plus the raw bytes of one anchor tensor — O(1 tensor) of I/O.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import List, Tuple

import torch

from .io import ShardReader, ShardedTensorIndex


def tensor_content_hash(tensor) -> str:
    """SHA-256 over a tensor's values, in a dtype numpy can represent.

    BF16/FP16 have no numpy dtype, so they are widened to float32 (lossless)
    before hashing. The resulting bytes uniquely determine the original values,
    so the hash is a valid content fingerprint.
    """
    arr = tensor.detach().contiguous()
    if arr.dtype in (torch.float16, torch.bfloat16):
        arr = arr.to(torch.float32)
    arr = arr.numpy()
    return hashlib.sha256(arr.tobytes()).hexdigest()


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


def compute_base_fingerprint(model_dir: Path) -> str:
    """Fingerprint a model directory: manifest + anchor-tensor content hash.

    The anchor is the first tensor in sorted name order, so the result is
    deterministic for a given model directory.
    """
    index = ShardedTensorIndex.from_dir(Path(model_dir))
    keys = index.keys()
    if not keys:
        raise ValueError(f"No tensors found in {model_dir}")

    manifest = model_manifest(index)
    reader = ShardReader(index)
    try:
        anchor = reader.get_tensor(keys[0])
    finally:
        reader.close()

    anchor_hash = tensor_content_hash(anchor)
    payload = json.dumps(manifest, sort_keys=True) + anchor_hash
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def tensor_names_and_shapes(model_dir: Path) -> dict:
    """Map of tensor name -> (shape, dtype) for a model directory."""
    index = ShardedTensorIndex.from_dir(Path(model_dir))
    manifest = model_manifest(index)
    return {name: (shape, dtype) for (name, shape, dtype) in manifest}
