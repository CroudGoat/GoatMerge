"""Tests for the config.json metadata envelope."""

import pytest

import torch

from goatmerge.metadata import (
    build_metadata,
    build_merged_metadata,
    load_metadata,
    validate_metadata,
    write_metadata,
)

NAMES = ["w1", "w2", "w3"]


def test_build_metadata_fields():
    meta = build_metadata(
        base_model="base-dir",
        source_model="src-dir",
        dtype=torch.bfloat16,
        tensor_names=NAMES,
        base_fingerprint="abc" * 21,
    )
    assert meta["model_type"] == "task_vector"
    assert meta["base_model"] == "base-dir"
    assert meta["source_model"] == "src-dir"
    assert meta["dtype"] == str(torch.bfloat16)
    assert meta["num_tensors"] == 3
    assert meta["tensor_names"] == NAMES
    assert meta["base_fingerprint"] == "abc" * 21
    validate_metadata(meta)  # must pass


def test_build_merged_metadata_fields():
    meta = build_merged_metadata(
        base_model="base-dir",
        task_vectors=[{"dir": "tv1", "kind": "tv", "weight": 0.5}],
        tensor_names=NAMES,
        dtype=None,
        merge_settings={"density": 0.5, "consensus": "sum"},
        skipped_tensors=["w9"],
    )
    assert meta["model_type"] == "merged_model"
    assert meta["dtype"] is None
    assert meta["task_vectors"] == [{"dir": "tv1", "kind": "tv", "weight": 0.5}]
    assert meta["skipped_tensors"] == ["w9"]
    validate_metadata(meta)


def test_write_load_roundtrip(tmp_path):
    meta = build_metadata(
        base_model="b",
        source_model="s",
        dtype=torch.float32,
        tensor_names=NAMES,
        base_fingerprint="f" * 64,
    )
    p = write_metadata(tmp_path, meta)
    loaded = load_metadata(tmp_path)
    assert loaded["base_fingerprint"] == meta["base_fingerprint"]
    assert loaded["tensor_names"] == NAMES
    assert loaded["model_type"] == "task_vector"


def test_load_missing_config_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_metadata(tmp_path)


def test_validate_missing_field_raises():
    meta = build_metadata(
        base_model="b",
        source_model="s",
        dtype=torch.float32,
        tensor_names=NAMES,
        base_fingerprint="f",
    )
    del meta["base_fingerprint"]
    with pytest.raises(ValueError, match="base_fingerprint"):
        validate_metadata(meta)


def test_validate_unknown_model_type():
    meta = build_metadata(
        base_model="b",
        source_model="s",
        dtype=torch.float32,
        tensor_names=NAMES,
        base_fingerprint="f",
    )
    meta["model_type"] = "banana"
    with pytest.raises(ValueError, match="Unknown model_type"):
        validate_metadata(meta)


def test_validate_num_tensors_mismatch():
    meta = build_metadata(
        base_model="b",
        source_model="s",
        dtype=torch.float32,
        tensor_names=NAMES,
        base_fingerprint="f",
    )
    meta["num_tensors"] = 2
    with pytest.raises(ValueError, match="num_tensors"):
        validate_metadata(meta)


def test_validate_empty_tensor_names():
    meta = build_metadata(
        base_model="b",
        source_model="s",
        dtype=torch.float32,
        tensor_names=[],
        base_fingerprint="f",
    )
    with pytest.raises(ValueError, match="non-empty"):
        validate_metadata(meta)
