"""Tests for content hashing and base-model fingerprints."""

import os

import safetensors.torch
import torch

from goatmerge.fingerprint import (
    compute_base_fingerprint,
    model_manifest,
    tensor_content_hash,
)
from goatmerge.io import ShardedTensorIndex


def _write_single_file(path: str, tensors: dict[str, torch.Tensor]) -> None:
    os.makedirs(path, exist_ok=True)
    safetensors.torch.save_file(tensors, os.path.join(path, "model.safetensors"))


def test_tensor_content_hash_deterministic():
    t = torch.randn(64)
    assert tensor_content_hash(t) == tensor_content_hash(t.clone())
    t2 = t.clone()
    t2[0] += 1.0
    assert tensor_content_hash(t) != tensor_content_hash(t2)


def test_tensor_content_hash_bf16_widening_is_lossless():
    t = torch.randn(128, dtype=torch.bfloat16)
    # bf16 -> f32 is lossless, so the widened hash must match the f32 hash
    assert tensor_content_hash(t) == tensor_content_hash(t.float())
    # and differ from a different bf16 tensor
    t2 = t.clone()
    t2[5] += 0.01
    assert tensor_content_hash(t) != tensor_content_hash(t2)


def test_manifest_shapes_and_dtypes(tmp_path):
    d = str(tmp_path)
    _write_single_file(d, {"w": torch.randn(8, 4, dtype=torch.bfloat16), "b": torch.randn(8)})
    idx = ShardedTensorIndex.from_dir(d)
    manifest = model_manifest(idx)
    got = {name: (shape, dtype) for name, shape, dtype in manifest}
    # safetensors reports dtypes as their string names (e.g. "BF16")
    assert got["w"] == ((8, 4), "BF16")
    assert got["b"] == ((8,), "F32")


def test_fingerprint_deterministic(tmp_path):
    d = str(tmp_path)
    _write_single_file(d, {"w": torch.randn(16, 16), "b": torch.randn(16)})
    f1 = compute_base_fingerprint(d)
    f2 = compute_base_fingerprint(d)
    assert f1 == f2
    assert len(f1) == 64  # sha256 hexdigest


def test_fingerprint_sensitive_to_anchor_change(tmp_path):
    d = str(tmp_path)
    _write_single_file(d, {"w": torch.randn(16, 16), "b": torch.randn(16)})
    f1 = compute_base_fingerprint(d)
    # the anchor is the first tensor in sorted name order: "b"
    _write_single_file(d, {"w": torch.randn(16, 16), "b": torch.randn(16) + 1.0})
    f2 = compute_base_fingerprint(d)
    assert f1 != f2


def test_fingerprint_differs_across_models(tmp_path):
    d1 = str(tmp_path / "m1")
    d2 = str(tmp_path / "m2")
    _write_single_file(d1, {"w": torch.randn(32, 32)})
    _write_single_file(d2, {"w": torch.randn(32, 32)})  # different values
    assert compute_base_fingerprint(d1) != compute_base_fingerprint(d2)
