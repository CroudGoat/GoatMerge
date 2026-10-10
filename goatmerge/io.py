"""Streaming shard-aware I/O for task vectors and merged models.

Self-contained (no mergekit dependency) but byte-compatible with the
HF sharded-safetensors layout used by mergekit and taskvector:
  model.safetensors.index.json
  model-00001-of-0000N.safetensors

Only one tensor is resident at a time. ``get_tensor(key)`` loads a single
tensor; the caller must ``.clone()`` before mutating it in place, because the
returned tensor shares storage with the shard's cached copy.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import struct
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Tuple

import safetensors
import safetensors.torch
import torch

logger = logging.getLogger("goatmerge.io")


# --------------------------------------------------------------------------- #
# Sharded index (identical layout to mergekit.io.lazy_tensor_loader).
# --------------------------------------------------------------------------- #
@dataclass
class ShardInfo:
    filename: str
    contained_keys: List[str]


@dataclass
class ShardedTensorIndex:
    """Index of a safetensors model split across ``model-XXXXX-of-YYYYY`` shards."""

    base_path: str
    is_safetensors: bool
    tensor_paths: Dict[str, str]  # key -> shard filename
    shards: List[ShardInfo]

    @classmethod
    def from_dir(cls, base_path: Path) -> "ShardedTensorIndex":
        """Build an index from a directory containing safetensors weights.

        Prefers ``model.safetensors.index.json`` (HF sharded format), then a
        single ``model.safetensors`` file, then a bare scan of ``*.safetensors``.
        """
        base_path = Path(base_path)
        index_path = base_path / "model.safetensors.index.json"
        if index_path.exists():
            with index_path.open("r", encoding="utf-8") as f:
                weight_map = json.load(f).get("weight_map", {})
            return cls.from_weight_map(str(base_path), weight_map)

        single = base_path / "model.safetensors"
        if single.exists():
            with safetensors.safe_open(str(single), framework="pt") as st:
                keys = list(st.keys())
            return cls(
                base_path=str(base_path),
                is_safetensors=True,
                tensor_paths={k: single.name for k in keys},
                shards=[ShardInfo(single.name, list(keys))],
            )

        # Fallback: scan every safetensors file in the directory.
        shards_by_key: Dict[str, str] = {}
        shard_keys: Dict[str, List[str]] = {}
        for sf in sorted(base_path.glob("*.safetensors")):
            with safetensors.safe_open(str(sf), framework="pt") as st:
                keys = list(st.keys())
            shard_keys[sf.name] = keys
            for k in keys:
                shards_by_key[k] = sf.name
        return cls(
            base_path=str(base_path),
            is_safetensors=True,
            tensor_paths=shards_by_key,
            shards=[
                ShardInfo(filename=fn, contained_keys=list(keys))
                for fn, keys in sorted(shard_keys.items())
            ],
        )

    @classmethod
    def from_weight_map(cls, base_path: str, weight_map: Mapping[str, str]) -> "ShardedTensorIndex":
        tensor_paths = dict(weight_map)
        shard_names = sorted({v for v in weight_map.values() if v})
        shards = [
            ShardInfo(
                filename=fn,
                contained_keys=[k for k, v in weight_map.items() if v == fn],
            )
            for fn in shard_names
        ]
        return cls(
            base_path=base_path,
            is_safetensors=True,
            tensor_paths=tensor_paths,
            shards=shards,
        )

    def shard_filename(self, key: str) -> str:
        return self.tensor_paths[key]

    def keys(self) -> List[str]:
        """All tensor names known to this index (sorted for determinism)."""
        return sorted(self.tensor_paths.keys())


class ShardReader:
    """Lazy streaming reader over a :class:`ShardedTensorIndex`.

    A shard is opened once and reused until its keys are exhausted, then
    closed. The only tensors resident at any moment are those returned by
    ``get_tensor`` (one per call).
    """

    def __init__(self, index: ShardedTensorIndex, use_mmap: bool = False):
        self.index = index
        self._handles: Dict[str, "safetensors.torch.safe_open"] = {}
        self._use_mmap = use_mmap
        # shard filename -> (parsed safetensors header, header byte length)
        self._header_cache: Dict[str, Tuple[dict, int]] = {}

    def _shard_header(self, shard_name: str) -> Tuple[dict, int]:
        """Cached ``(parsed safetensors header, header byte length)``.

        The chunked merge path reads one row range at a time; re-opening and
        re-parsing the (potentially large) shard header for every row read
        dominated its cost, so the parsed header is cached here.
        """
        cached = self._header_cache.get(shard_name)
        if cached is not None:
            return cached
        path = Path(self.index.base_path) / shard_name
        with open(str(path), "rb") as f:
            header_len = struct.unpack("<Q", f.read(8))[0]
            f.seek(8)
            header_json = f.read(header_len)
        meta = json.loads(header_json)
        self._header_cache[shard_name] = (meta, header_len)
        return meta, header_len

    def _read_direct(self, key: str) -> torch.Tensor:
        """Read a tensor directly from disk without mmap."""
        return self._read_rows(key, 0, None)

    def _read_rows(self, key: str, row_start: int, row_end: Optional[int]) -> torch.Tensor:
        """Read specific rows of a tensor directly from disk.

        For a tensor with shape (R, C1, C2, ..., Ck) stored in row-major
        order, row ``i`` starts at byte offset ``i * (C1*C2*...*Ck) *
        element_size`` within the tensor data. We seek to the correct
        offset and read only the needed bytes.
        """
        shard_name = self.index.shard_filename(key)
        meta, header_len = self._shard_header(shard_name)
        path = Path(self.index.base_path) / shard_name
        with open(str(path), "rb") as f:
            entry = meta[key]
            dtype_str = entry["dtype"]
            shape = tuple(entry["shape"])
            data_start, data_end = entry["data_offsets"]
            # data_offsets are relative to the data-section start (8 + header_len),
            # not absolute file positions. Add the base to get the absolute offset.
            base = 8 + header_len
            dtype_map = {
                "BF16": torch.bfloat16,
                "F16": torch.float16,
                "F32": torch.float32,
                "F64": torch.float64,
                "I32": torch.int32,
                "I64": torch.int64,
            }
            dtype = dtype_map.get(dtype_str, torch.bfloat16)
            # Compute the byte offset for the row range.
            # For a (R, C1, C2, ..., Ck) tensor, row i starts at byte
            # i * (C1*C2*...*Ck) * element_size within the tensor data.
            n_rows = shape[0]
            cols = 1
            for s in shape[1:]:
                cols *= s
            elem_size = torch.tensor([], dtype=dtype).element_size()
            row_start_byte = base + data_start + row_start * cols * elem_size
            if row_end is None:
                row_end = n_rows
            row_end_byte = base + data_start + row_end * cols * elem_size
            f.seek(row_start_byte)
            raw = f.read(row_end_byte - row_start_byte)
            # Use a writable bytearray to avoid the non-writable buffer warning.
            buf = bytearray(raw)
            t = torch.frombuffer(buf, dtype=dtype).reshape(
                (row_end - row_start,) + shape[1:]
            )
            return t.clone()

    def get_slice(self, key: str):
        """Return a reusable slice object for ``key`` (opened on first use)."""
        if self._use_mmap:
            shard_name = self.index.shard_filename(key)
            if shard_name not in self._handles:
                path = Path(self.index.base_path) / shard_name
                self._handles[shard_name] = safetensors.torch.safe_open(
                    str(path), framework="pt", device="cpu"
                )
            return self._handles[shard_name].get_slice(key)
        else:
            # Direct read: return a wrapper that supports [start:end] slicing
            return _DirectSlice(self, key)

    def get_tensor(self, key: str) -> torch.Tensor:
        """Load one full tensor into memory.

        NOTE: the returned tensor shares storage with the shard's cached copy,
        so callers that mutate it in place must first ``.clone()`` it.
        """
        if self._use_mmap:
            return self.get_slice(key)[:]
        else:
            return self._read_direct(key)

    def keys(self) -> List[str]:
        return self.index.keys()

    def close(self):
        handles = list(self._handles.values())
        self._handles.clear()
        for h in handles:
            h.__exit__(None, None, None)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


class _DirectSlice:
    """Wrapper that supports [start:end] row slicing via direct disk reads.

    Provides the same interface as ``safe_open(...).get_slice(key)``:
    ``get_shape()`` and ``get_dtype()`` for shape/dtype inspection, and
    ``[start:end]`` for row slicing. Shape/dtype are read from the safetensors
    header (no full-tensor load).
    """

    def __init__(self, reader: "ShardReader", key: str):
        self._reader = reader
        self._key = key
        self._shape: Optional[tuple] = None
        self._dtype_str: Optional[str] = None

    def _read_header_entry(self) -> dict:
        """Read the safetensors header entry for this key (no tensor data)."""
        shard_name = self._reader.index.shard_filename(self._key)
        meta, _ = self._reader._shard_header(shard_name)
        return meta[self._key]

    def get_shape(self) -> list:
        if self._shape is None:
            entry = self._read_header_entry()
            self._shape = tuple(entry["shape"])
            self._dtype_str = entry["dtype"]
        return list(self._shape)

    def get_dtype(self) -> str:
        if self._dtype_str is None:
            self.get_shape()  # triggers the header read
        return self._dtype_str

    def __getitem__(self, idx):
        # idx is a slice object (start, end)
        if isinstance(idx, slice):
            start, end = idx.start, idx.stop
            if start is None:
                start = 0
            if end is None:
                end = self._shape[0] if self._shape else None
            result = self._reader._read_rows(self._key, start, end)
            return result
        raise TypeError("only slice indexing is supported")


def _prepare_delta(
    t: torch.Tensor,
    base: torch.Tensor,
    entry_dir,
    kind: str,
    key: str,
) -> Optional[torch.Tensor]:
    """Shape-check ``t`` against ``base`` and convert it to a delta.

    ``entry_dir`` / ``key`` are only used for the log messages. Returns
    ``None`` when the tensor is size-incompatible (caller skips it).
    """
    if t.shape != base.shape:
        # embed-style submatrix: truncate when the TV is a superset grid
        if (
            base.dim() >= 2
            and t.dim() >= 2
            and t.shape[0] >= base.shape[0]
            and t.shape[1] >= base.shape[1]
            and t.shape[2:] == base.shape[2:]
        ):
            logger.warning("using submatrix of %s:%s", entry_dir, key)
            t = t[: base.shape[0], : base.shape[1], *base.shape[2:]]
        else:
            logger.warning("skipping %s:%s due to size mismatch", entry_dir, key)
            return None
    if kind == "tv":
        return t.to(base.dtype).clone()
    # model mode: delta = W_i - base (in place after clone)
    t = t.to(base.dtype).clone()
    t.sub_(base)
    return t


def load_delta(base: torch.Tensor, entry_dir, kind: str, key: str, readers: dict):
    """Load ``delta = W_i - base`` (``kind="model"``) or the stored TV (``kind="tv"``).

    Returns ``None`` if the tensor is missing or size-incompatible. The
    returned tensor owns its storage (cloned). ``entry_dir`` is a ``Path`` and
    ``kind`` is ``"tv"`` or ``"model"`` (no ``ModelEntry`` import, so this stays
    import-cycle-free).
    """
    reader = readers[entry_dir]
    if key not in reader.index.tensor_paths:
        logger.warning("skipping %s:%s (tensor missing)", entry_dir, key)
        return None
    t = reader.get_tensor(key)
    return _prepare_delta(t, base, entry_dir, kind, key)


def load_delta_chunk(
    base_chunk: torch.Tensor,
    entry_dir,
    kind: str,
    key: str,
    readers: dict,
    start: int,
    end: int,
) -> Optional[torch.Tensor]:
    """Load rows ``[start:end]`` of the tensor as a delta chunk.

    Row ranges are read straight from the shard (no full-tensor load) and go
    through the same shape checks as :func:`load_delta`, so a missing or
    size-incompatible tensor is skipped rather than corrupting the merge.
    Returns ``None`` when the tensor must be skipped.
    """
    reader = readers[entry_dir]
    if key not in reader.index.tensor_paths:
        logger.warning("skipping %s:%s (tensor missing)", entry_dir, key)
        return None
    t = reader.get_slice(key)[start:end]
    return _prepare_delta(t, base_chunk, entry_dir, kind, key)


# --------------------------------------------------------------------------- #
# Output writer: emits the exact HF sharded-safetensors layout.
# --------------------------------------------------------------------------- #
# Copied from the base model into a merged output so the result is directly
# loadable by transformers. Weight files are never touched.
SUPPORT_FILES = (
    "config.json",
    "generation_config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "added_tokens.json",
    "vocab.json",
    "merges.txt",
    "vocab.txt",
    "chat_template.jinja",
    "preprocessor_config.json",
)


def copy_support_files(base_dir, out_dir) -> List[str]:
    """Copy the base model's non-weight config files into ``out_dir``.

    Merging only rewrites weights, so the architecture config, generation
    config, and tokenizer files are carried over unchanged. Missing files are
    skipped. Returns the list of copied file names.
    """
    base_dir = Path(base_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    copied: List[str] = []
    for name in SUPPORT_FILES:
        src = base_dir / name
        if not src.is_file():
            continue
        shutil.copyfile(src, out_dir / name)
        copied.append(name)
    return copied


class TensorWriter:
    """Writes a model/task-vector as HF sharded safetensors.

    Produces ``model.safetensors`` (single shard) or
    ``model-00001-of-0000N.safetensors`` + ``model.safetensors.index.json``
    (multiple shards), byte-compatible with mergekit's ``TensorWriter``.
    """

    def __init__(
        self,
        out_path: str,
        max_shard_size: int = 1_000_000_000,
        safe_serialization: bool = True,
        flush_after_each: bool = False,
    ) -> None:
        os.makedirs(out_path, exist_ok=True)
        self.out_path = out_path
        self.max_shard_size = max_shard_size
        self.safe_serialization = safe_serialization
        self.flush_after_each = flush_after_each

        self.shards_written = 0
        self.weight_map: Dict[str, str] = {}
        self.current_shard: Dict[str, torch.Tensor] = {}
        self.current_shard_size = 0
        self.total_size = 0

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.finalize()

    def save_tensor(self, name: str, tensor: torch.Tensor, clone: bool = False) -> None:
        if not tensor.is_contiguous():
            tensor = tensor.contiguous()
        if clone:
            tensor = tensor.clone()

        tensor_size = tensor.numel() * tensor.element_size()
        if (
            self.current_shard
            and self.max_shard_size > 0
            and self.current_shard_size + tensor_size > self.max_shard_size
        ):
            self._flush_current_shard()

        self.current_shard[name] = tensor
        self.current_shard_size += tensor_size

        if self.flush_after_each:
            self._flush_current_shard()

    def _flush_current_shard(self) -> None:
        if not self.current_shard:
            return
        shard_data = self.current_shard
        shard_index = self.shards_written
        self.total_size += self.current_shard_size
        self.current_shard = {}
        self.current_shard_size = 0
        self.shards_written += 1

        prefix, extension = self._name_components()
        shard_name = f"{prefix}-{shard_index + 1}.{extension}"
        shard_path = os.path.join(self.out_path, shard_name)
        for key in shard_data:
            self.weight_map[key] = shard_name
        self._write_shard(shard_data, shard_path)

    def _write_shard(self, shard_data: Dict[str, torch.Tensor], shard_path: str) -> None:
        if self.safe_serialization:
            safetensors.torch.save_file(
                shard_data, shard_path, metadata={"format": "pt"}
            )
        else:
            torch.save(shard_data, shard_path)

    def finalize(self) -> None:
        self._flush_current_shard()

        prefix, extension = self._name_components()
        total_shards = self.shards_written

        # Standardize shard names to the Hugging Face format.
        name_remap: Dict[str, str] = {}
        if total_shards == 1:
            name_remap[f"{prefix}-1.{extension}"] = f"{prefix}.{extension}"
        else:
            for idx in range(total_shards):
                old_name = f"{prefix}-{idx + 1}.{extension}"
                new_name = f"{prefix}-{idx + 1:05d}-of-{total_shards:05d}.{extension}"
                name_remap[old_name] = new_name

        for old_name, new_name in name_remap.items():
            old_path = os.path.join(self.out_path, old_name)
            new_path = os.path.join(self.out_path, new_name)
            if old_path != new_path and os.path.exists(old_path):
                os.rename(old_path, new_path)

        if total_shards > 1:
            for key in self.weight_map:
                self.weight_map[key] = name_remap.get(self.weight_map[key], self.weight_map[key])
            index_filename = f"{prefix}.{extension}.index.json"
            index_path = os.path.join(self.out_path, index_filename)
            with open(index_path, "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "metadata": {
                            "total_size": self.total_size,
                            "goatmerge_version": "0.1.0",
                        },
                        "weight_map": self.weight_map,
                    },
                    f,
                    indent=2,
                )

    def _name_components(self):
        basename = "model" if self.safe_serialization else "pytorch_model"
        extension = "safetensors" if self.safe_serialization else "bin"
        return basename, extension


# --------------------------------------------------------------------------- #
# Progress accounting.
# --------------------------------------------------------------------------- #
@dataclass
class Progress:
    total_tensors: int
    start_time: float = field(default_factory=time.time)

    @property
    def elapsed(self) -> float:
        return time.time() - self.start_time

    def report(self, current: int) -> None:
        rate = self.total_tensors / self.elapsed if self.elapsed > 0 else 0.0
        logger.info(
            "tensor %d/%d | %.1fs elapsed | ~%.1f tensors/s",
            current,
            self.total_tensors,
            self.elapsed,
            rate,
        )
