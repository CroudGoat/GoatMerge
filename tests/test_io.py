"""Tests for the sharded-safetensors I/O layer (index, reader, writer)."""

import json
import os

import pytest
import safetensors.torch
import torch

from goatmerge.io import Progress, ShardReader, ShardedTensorIndex, TensorWriter


def _write_single_file(path: str, tensors: dict[str, torch.Tensor]) -> None:
    os.makedirs(path, exist_ok=True)
    safetensors.torch.save_file(tensors, os.path.join(path, "model.safetensors"))


def _write_shards(path: str, shards: list[dict[str, torch.Tensor]]) -> None:
    os.makedirs(path, exist_ok=True)
    weight_map = {}
    for i, shard in enumerate(shards):
        name = f"model-{i + 1:05d}-of-{len(shards):05d}.safetensors"
        safetensors.torch.save_file(shard, os.path.join(path, name))
        for k in shard:
            weight_map[k] = name
    with open(os.path.join(path, "model.safetensors.index.json"), "w") as f:
        json.dump({"metadata": {"total_size": 0}, "weight_map": weight_map}, f)


def test_index_from_single_file(tmp_path):
    d = str(tmp_path)
    _write_single_file(d, {"w": torch.randn(4, 4), "b": torch.randn(4)})
    idx = ShardedTensorIndex.from_dir(d)
    assert idx.keys() == ["b", "w"]
    assert set(idx.tensor_paths.values()) == {"model.safetensors"}
    assert len(idx.shards) == 1
    assert set(idx.shards[0].contained_keys) == {"w", "b"}


def test_index_from_index_json(tmp_path):
    d = str(tmp_path)
    _write_shards(
        d,
        [
            {"a": torch.randn(8), "b": torch.randn(8)},
            {"c": torch.randn(8)},
        ],
    )
    idx = ShardedTensorIndex.from_dir(d)
    assert idx.keys() == ["a", "b", "c"]
    assert idx.shard_filename("a") == "model-00001-of-00002.safetensors"
    assert idx.shard_filename("c") == "model-00002-of-00002.safetensors"
    assert [len(s.contained_keys) for s in idx.shards] == [2, 1]


def test_index_glob_fallback(tmp_path):
    d = str(tmp_path)
    _write_shards(
        d,
        [
            {"a": torch.randn(4)},
            {"b": torch.randn(4)},
        ],
    )
    # remove the index file -> glob fallback must rebuild the index
    os.remove(os.path.join(d, "model.safetensors.index.json"))
    idx = ShardedTensorIndex.from_dir(d)
    assert idx.keys() == ["a", "b"]
    assert {v for v in idx.tensor_paths.values()} == {
        "model-00001-of-00002.safetensors",
        "model-00002-of-00002.safetensors",
    }


def test_reader_clone_semantics(tmp_path):
    """Direct-read reader: each ``get_tensor`` returns a fresh, independent
    tensor (no shared in-memory cache), so mutations never leak across reads.
    """
    d = str(tmp_path)
    _write_single_file(d, {"w": torch.randn(16)})
    idx = ShardedTensorIndex.from_dir(d)
    with ShardReader(idx) as r:
        t1 = r.get_tensor("w")
        t2 = r.get_tensor("w")
        # independent storage: distinct data_ptr, identical values
        assert t1.data_ptr() != t2.data_ptr()
        assert torch.equal(t1, t2)
        orig = t1.clone()  # capture the original before mutating
        t1.add_(1.0)  # mutate t1's own storage
        t3 = r.get_tensor("w")
        # a fresh read is the original (mutation did not persist)
        assert torch.allclose(t3, orig)
        assert not torch.allclose(t3, t1)
        # a fresh reader re-reads the original bytes from disk
    with ShardReader(idx) as r2:
        t4 = r2.get_tensor("w")
        assert torch.allclose(t4, orig)  # unchanged original
        t5 = t4.clone()
        t5.add_(2.0)
        assert torch.allclose(r2.get_tensor("w"), t4)  # clone protects storage


def test_writer_single_shard(tmp_path):
    out = str(tmp_path / "out")
    a = torch.randn(8, 8)
    b = torch.randn(8)
    with TensorWriter(out) as w:
        w.save_tensor("a", a)
        w.save_tensor("b", b)
    assert os.path.exists(os.path.join(out, "model.safetensors"))
    assert not os.path.exists(os.path.join(out, "model.safetensors.index.json"))
    with safetensors.torch.safe_open(os.path.join(out, "model.safetensors"), framework="pt") as st:
        assert set(st.keys()) == {"a", "b"}
        assert torch.equal(st.get_slice("a")[:], a)
        assert torch.equal(st.get_slice("b")[:], b)


def test_writer_multi_shard_renames(tmp_path):
    out = str(tmp_path / "out")
    with TensorWriter(out, max_shard_size=1000) as w:
        # 3 tensors of 400 B (f32) each: a+b fit in one 1000-B shard, c overflows
        w.save_tensor("a", torch.randn(100))
        w.save_tensor("b", torch.randn(100))
        w.save_tensor("c", torch.randn(100))
    files = sorted(os.listdir(out))
    assert "model-00001-of-00002.safetensors" in files
    assert "model-00002-of-00002.safetensors" in files
    assert "model.safetensors.index.json" in files
    with open(os.path.join(out, "model.safetensors.index.json")) as f:
        wm = json.load(f)["weight_map"]
    assert set(wm) == {"a", "b", "c"}
    assert all(name.endswith(".safetensors") for name in wm.values())
    # every key must be loadable from its mapped shard
    with ShardReader(ShardedTensorIndex.from_dir(out)) as r:
        for k in ("a", "b", "c"):
            assert r.get_tensor(k).numel() == 100


def test_progress_reporting(tmp_path):
    p = Progress(total_tensors=10)
    p.report(5)  # must not raise
    assert p.total_tensors == 10
    assert p.elapsed >= 0
