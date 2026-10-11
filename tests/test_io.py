"""Tests for the sharded-safetensors I/O layer (index, reader, writer)."""

import json
import os

import pytest
import safetensors.torch
import torch

from goatmerge.io import (
    Progress,
    ShardReader,
    ShardedTensorIndex,
    TensorWriter,
    load_delta,
    safetensors_dtype_str,
)


def _manifest(tensors: dict[str, torch.Tensor]) -> list:
    return [(k, safetensors_dtype_str(v.dtype), tuple(v.shape)) for k, v in tensors.items()]


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
    tensors = {"a": torch.randn(8, 8), "b": torch.randn(8)}
    with TensorWriter(out, _manifest(tensors)) as w:
        for k in sorted(tensors):
            w.save_tensor(k, tensors[k])
    assert os.path.exists(os.path.join(out, "model.safetensors"))
    assert not os.path.exists(os.path.join(out, "model.safetensors.index.json"))
    with safetensors.torch.safe_open(os.path.join(out, "model.safetensors"), framework="pt") as st:
        assert set(st.keys()) == {"a", "b"}
        assert torch.equal(st.get_slice("a")[:], tensors["a"])
        assert torch.equal(st.get_slice("b")[:], tensors["b"])


def test_writer_multi_shard_renames(tmp_path):
    out = str(tmp_path / "out")
    tensors = {"a": torch.randn(100), "b": torch.randn(100), "c": torch.randn(100)}
    with TensorWriter(out, _manifest(tensors), max_shard_size=1000) as w:
        # 3 tensors of 400 B (f32) each: a+b fit in one 1000-B shard, c overflows
        for k in sorted(tensors):
            w.save_tensor(k, tensors[k])
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


def test_drop_caches_advised_on_row_read(tmp_path, monkeypatch):
    """drop_caches=True: each row read advises DONTNEED on its file range."""
    calls = []
    monkeypatch.setattr(
        os, "posix_fadvise",
        lambda fd, offset, length, advice: calls.append((fd, offset, length, advice)),
        raising=False,
    )
    d = str(tmp_path)
    _write_single_file(d, {"w": torch.randn(16, 8, dtype=torch.bfloat16)})
    idx = ShardedTensorIndex.from_dir(d)
    with ShardReader(idx, drop_caches=True) as r:
        t = r.get_tensor("w")
        assert t.shape == (16, 8)
        chunk = r.get_slice("w")[4:8]
        assert chunk.shape == (4, 8)
    assert calls, "expected at least one fadvise call"
    for fd, offset, length, advice in calls:
        assert advice == os.POSIX_FADV_DONTNEED
        assert offset > 0 and length > 0
    # values are unaffected by the advice
    with ShardReader(idx) as r2:
        assert torch.equal(r2.get_tensor("w"), t)


def test_drop_caches_off_by_default(tmp_path, monkeypatch):
    """drop_caches=False (default): no fadvise call is issued."""
    calls = []
    monkeypatch.setattr(
        os, "posix_fadvise",
        lambda fd, offset, length, advice: calls.append((fd, offset, length, advice)),
        raising=False,
    )
    d = str(tmp_path)
    _write_single_file(d, {"w": torch.randn(16, 8)})
    idx = ShardedTensorIndex.from_dir(d)
    with ShardReader(idx) as r:
        r.get_tensor("w")
    assert calls == []


def test_writer_drop_caches_after_shard(tmp_path, monkeypatch):
    """drop_caches=True: a written shard is advised DONTNEED after writing."""
    calls = []
    monkeypatch.setattr(
        os, "posix_fadvise",
        lambda fd, offset, length, advice: calls.append((fd, offset, length, advice)),
        raising=False,
    )
    out = str(tmp_path / "out")
    with TensorWriter(out, _manifest({"a": torch.randn(8, 8)}), drop_caches=True) as w:
        w.save_tensor("a", torch.randn(8, 8))
    assert calls, "expected an fadvise call for the written shard"
    assert all(c[3] == os.POSIX_FADV_DONTNEED for c in calls)
    # the shard itself is intact and loadable
    with ShardReader(ShardedTensorIndex.from_dir(out)) as r:
        assert r.get_tensor("a").shape == (8, 8)


def test_load_delta_no_clone_independent(tmp_path):
    """clone=False reuses the reader's private storage: correct values, and
    in-place mutation never leaks into later reads (each read owns a fresh
    buffer, so no cross-read aliasing)."""
    d = str(tmp_path)
    base_t = torch.randn(8, 8, dtype=torch.bfloat16)
    src_t = base_t + torch.randn(8, 8, dtype=torch.bfloat16) * 0.1
    _write_single_file(d + "/base", {"w": base_t})
    _write_single_file(d + "/src", {"w": src_t})
    base_idx = ShardedTensorIndex.from_dir(d + "/base")
    src_idx = ShardedTensorIndex.from_dir(d + "/src")
    with ShardReader(base_idx) as br, ShardReader(src_idx) as sr:
        readers = {d + "/src": sr}
        base = br.get_tensor("w")
        # kind="tv": stored TV returned as-is (no extra copy)
        tv = load_delta(base, d + "/src", kind="tv", key="w", readers=readers, clone=False)
        assert torch.equal(tv, src_t.to(torch.bfloat16))
        tv.add_(1.0)  # mutate: must not affect later reads
        tv2 = load_delta(base, d + "/src", kind="tv", key="w", readers=readers, clone=False)
        assert torch.equal(tv2, src_t.to(torch.bfloat16))
        # kind="model": delta = source - base, computed in place
        dm = load_delta(base, d + "/src", kind="model", key="w", readers=readers, clone=False)
        assert torch.equal(dm, src_t.to(torch.bfloat16).sub(base))


def _read_all_tensors(path: str) -> dict:
    out = {}
    with safetensors.torch.safe_open(path, framework="pt") as st:
        for k in st.keys():
            out[k] = st.get_tensor(k)
    return out


def test_writer_matches_save_file_values_and_layout(tmp_path):
    """Streaming output carries the same values/layout as save_file.

    Header key order is canonical-sorted (save_file's own order is an
    opaque implementation detail), so equality is per-tensor bytes plus
    file layout — dtypes/shapes vary (incl. bf16 and odd sizes) to prove
    the offset math.
    """
    torch.manual_seed(0)
    tensors = {
        "z_last": torch.randn(7, dtype=torch.bfloat16),
        "m_mid": torch.randn(4, 5),
        "a_first": torch.randn(3, dtype=torch.float64),
        "b_int": torch.randint(-5, 5, (6,)).to(torch.int32),
    }
    ref_path = str(tmp_path / "ref.safetensors")
    safetensors.torch.save_file(tensors, ref_path, metadata={"format": "pt"})
    ref = _read_all_tensors(ref_path)

    out = str(tmp_path / "out")
    with TensorWriter(out, _manifest(tensors)) as w:
        for k in sorted(tensors):
            w.save_tensor(k, tensors[k])
    assert os.path.exists(os.path.join(out, "model.safetensors"))
    got = _read_all_tensors(os.path.join(out, "model.safetensors"))
    assert set(got) == set(ref)
    for k in ref:
        assert got[k].dtype == ref[k].dtype, k
        assert torch.equal(got[k], ref[k]), k

    # determinism: a second run is byte-identical to the first
    out2 = str(tmp_path / "out2")
    with TensorWriter(out2, _manifest(tensors)) as w:
        for k in sorted(tensors):
            w.save_tensor(k, tensors[k])
    with open(os.path.join(out, "model.safetensors"), "rb") as f:
        first = f.read()
    with open(os.path.join(out2, "model.safetensors"), "rb") as f:
        second = f.read()
    assert first == second


def test_writer_multi_shard_values(tmp_path):
    """Same per-tensor equality across a shard boundary."""
    torch.manual_seed(1)
    tensors = {
        "a": torch.randn(100),
        "b": torch.randn(100, dtype=torch.bfloat16),
        "c": torch.randn(50, 4),
    }
    out = str(tmp_path / "out")
    # 400 + 200 + 800 B with a 1000-B budget: {a, b} | {c}
    with TensorWriter(out, _manifest(tensors), max_shard_size=1000) as w:
        for k in sorted(tensors):
            w.save_tensor(k, tensors[k])
    got = {}
    for shard_file in (
        "model-00001-of-00002.safetensors",
        "model-00002-of-00002.safetensors",
    ):
        got.update(_read_all_tensors(os.path.join(out, shard_file)))
    assert set(got) == set(tensors)
    for k in tensors:
        assert torch.equal(got[k], tensors[k].to(got[k].dtype)), k


def test_writer_rejects_out_of_order_arrival(tmp_path):
    tensors = {"a": torch.randn(4), "b": torch.randn(4)}
    out = str(tmp_path / "out")
    with TensorWriter(out, _manifest(tensors)) as w:
        with pytest.raises(RuntimeError):
            w.save_tensor("b", tensors["b"])


def test_writer_drops_incomplete_shard(tmp_path):
    """A shard whose members never all arrive is removed, not renamed."""
    tensors = {"a": torch.randn(100), "b": torch.randn(100)}
    out = str(tmp_path / "out")
    with TensorWriter(out, _manifest(tensors), max_shard_size=1000) as w:
        # 400 B each, 1000-B budget: both land in one shard; save only "a"
        w.save_tensor("a", tensors["a"])
    leftovers = [f for f in os.listdir(out) if f.endswith(".safetensors")]
    assert leftovers == [], leftovers
