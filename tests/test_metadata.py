"""Tests for the goatmerge.json provenance sidecar."""

import json

import pytest

from goatmerge.metadata import (
    METADATA_NAME,
    build_metadata,
    build_merged_metadata,
    load_metadata,
    validate_metadata,
    write_metadata,
)


def _tv_meta(**kw):
    base = dict(base_model="base-dir", source_model="src-dir", base_fingerprint="abc" * 21)
    base.update(kw)
    return build_metadata(**base)


def test_build_metadata_fields():
    meta = _tv_meta()
    assert meta["model_type"] == "task_vector"
    assert meta["base_model"] == "base-dir"
    assert meta["source_model"] == "src-dir"
    assert meta["base_fingerprint"] == "abc" * 21
    # the tensor manifest is derivable from the shards and is NOT recorded
    assert "tensor_names" not in meta
    assert "num_tensors" not in meta
    validate_metadata(meta)  # must pass


def test_build_merged_metadata_fields():
    meta = build_merged_metadata(
        base_model="base-dir",
        task_vectors=[{"dir": "tv1", "kind": "tv", "weight": 0.5}],
        base_fingerprint="f" * 64,
        merge_settings={"density": 0.5, "consensus": "sum"},
    )
    assert meta["model_type"] == "merged_model"
    assert meta["task_vectors"] == [{"dir": "tv1", "kind": "tv", "weight": 0.5}]
    assert meta["merge_settings"] == {"density": 0.5, "consensus": "sum"}
    assert "tensor_names" not in meta
    assert "skipped_tensors" not in meta
    validate_metadata(meta)


def test_written_under_goatmerge_json(tmp_path):
    """The sidecar must not occupy the HuggingFace config.json name."""
    write_metadata(tmp_path, _tv_meta())
    assert (tmp_path / METADATA_NAME).is_file()
    assert not (tmp_path / "config.json").exists()


def test_write_load_roundtrip(tmp_path):
    meta = _tv_meta(base_fingerprint="f" * 64)
    write_metadata(tmp_path, meta)
    loaded = load_metadata(tmp_path)
    assert loaded["base_fingerprint"] == meta["base_fingerprint"]
    assert loaded["model_type"] == "task_vector"
    assert loaded["source_model"] == "src-dir"


def test_huggingface_config_is_not_metadata(tmp_path):
    """A transformers config.json must never be mistaken for GoatMerge's."""
    (tmp_path / "config.json").write_text(json.dumps({"architectures": ["LlamaForCausalLM"], "model_type": "llama"}))
    with pytest.raises(FileNotFoundError):
        load_metadata(tmp_path)


def test_legacy_config_json_sidecar_is_read(tmp_path):
    """GoatMerge <= 0.1.0 stored its metadata in config.json; keep reading it."""
    legacy = _tv_meta()
    (tmp_path / "config.json").write_text(json.dumps(legacy))
    loaded = load_metadata(tmp_path)
    assert loaded["base_fingerprint"] == legacy["base_fingerprint"]


def test_load_missing_metadata_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_metadata(tmp_path)


def test_validate_missing_field_raises():
    meta = _tv_meta()
    del meta["base_fingerprint"]
    with pytest.raises(ValueError, match="base_fingerprint"):
        validate_metadata(meta)


def test_validate_unknown_model_type():
    meta = _tv_meta()
    meta["model_type"] = "banana"
    with pytest.raises(ValueError, match="Unknown model_type"):
        validate_metadata(meta)


def test_validate_missing_format_version():
    meta = _tv_meta()
    del meta["goatmerge_format_version"]
    with pytest.raises(ValueError, match="goatmerge_format_version"):
        validate_metadata(meta)
