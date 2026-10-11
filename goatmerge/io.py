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


def advise_drop_caches(fd: int, offset: int, length: int) -> None:
    """Advise the kernel that a just-read file range will not be re-read.

    Uses ``POSIX_FADV_DONTNEED`` so the pages backing the range are released
    from the page cache instead of lingering until memory pressure. This keeps
    a streaming, single-pass workload (extract, consensus-free merge) from
    filling system memory with file cache. No-op on platforms without
    ``os.posix_fadvise``. Callers must only use it for data they will not
    re-read soon: a consensus divisor second pass, for example, re-reads the
    same deltas and would go back to disk.
    """
    if length <= 0:
        return
    fadvise = getattr(os, "posix_fadvise", None)
    dontneed = getattr(os, "POSIX_FADV_DONTNEED", None)
    if fadvise is None or dontneed is None:
        return
    try:
        fadvise(fd, offset, length, dontneed)
    except OSError:
        pass


# Canonical safetensors dtype table: string name -> (torch dtype, bytes/element).
# Single source of truth for dtype mapping in both directions.
_SAFETENSORS_DTYPES = {
    "BF16": (torch.bfloat16, 2),
    "F16": (torch.float16, 2),
    "F32": (torch.float32, 4),
    "F64": (torch.float64, 8),
    "I8": (torch.int8, 1),
    "U8": (torch.uint8, 1),
    "I16": (torch.int16, 2),
    "I32": (torch.int32, 4),
    "I64": (torch.int64, 8),
    "BOOL": (torch.bool, 1),
}
_TORCH_TO_SAFETENSORS = {v[0]: k for k, v in _SAFETENSORS_DTYPES.items()}


def dtype_from_safetensors(dtype_str: str) -> torch.dtype:
    """Map a safetensors dtype string (``"BF16"``, ``"F32"``, ...) to a torch dtype.

    Single source of truth shared by the row reader and the chunked paths.
    Unknown strings fall back to bfloat16, the dtype these engines are tuned
    for; every dtype safetensors can store is mapped explicitly.
    """
    entry = _SAFETENSORS_DTYPES.get(dtype_str)
    return entry[0] if entry is not None else torch.bfloat16


def safetensors_dtype_str(dtype: torch.dtype) -> str:
    """Map a torch dtype to its safetensors string name (inverse of above)."""
    try:
        return _TORCH_TO_SAFETENSORS[dtype]
    except KeyError:
        raise ValueError(f"dtype {dtype} has no safetensors representation")


def safetensors_elem_size(dtype_str: str) -> int:
    """Bytes per element for a safetensors dtype string (no tensor needed)."""
    entry = _SAFETENSORS_DTYPES.get(dtype_str)
    if entry is None:
        raise ValueError(f"unknown safetensors dtype: {dtype_str!r}")
    return entry[1]


def shape_fit(base_shape, src_shape) -> Optional[str]:
    """How a source tensor fits a base tensor: ``"exact"``, ``"submatrix"``.

    ``"submatrix"`` is the embed-style truncation (source is a superset grid
    in dims 0-1 with identical trailing dims). Returns ``None`` when the
    tensor must be skipped. Shared by the delta shape checks and the writer
    arrival planner so the two can never disagree.
    """
    base_shape = tuple(base_shape)
    src_shape = tuple(src_shape)
    if base_shape == src_shape:
        return "exact"
    if (
        len(base_shape) >= 2
        and len(src_shape) >= 2
        and src_shape[0] >= base_shape[0]
        and src_shape[1] >= base_shape[1]
        and src_shape[2:] == base_shape[2:]
    ):
        return "submatrix"
    return None


class ShardReader:
    """Lazy streaming reader over a :class:`ShardedTensorIndex`.

    A shard is opened once and reused until its keys are exhausted, then
    closed. The only tensors resident at any moment are those returned by
    ``get_tensor`` (one per call).

    ``drop_caches`` asks the kernel to release each file range right after
    it is read (see :func:`advise_drop_caches`), so bulk one-shot reads do
    not fill the page cache. Only enable it for data that will not be
    re-read soon.
    """

    def __init__(self, index: ShardedTensorIndex, use_mmap: bool = False, drop_caches: bool = False):
        self.index = index
        self._handles: Dict[str, "safetensors.torch.safe_open"] = {}
        self._use_mmap = use_mmap
        self._drop_caches = drop_caches
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
            dtype = dtype_from_safetensors(dtype_str)
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
            n_elems = row_end - row_start
            shape_out = (n_elems,) + shape[1:]
            n_bytes = row_end_byte - row_start_byte
            if n_bytes == 0:
                return torch.empty(shape_out, dtype=dtype)
            # One allocation per read: the bytes land directly in the
            # bytearray that backs the returned tensor, so a read costs a single
            # buffer (instead of bytes -> bytearray -> clone) and still returns
            # storage the caller exclusively owns.
            buf = bytearray(n_bytes)
            mv = memoryview(buf)
            with open(str(path), "rb") as f:
                f.seek(row_start_byte)
                read = 0
                while read < n_bytes:
                    got = f.readinto(mv[read:])
                    if got == 0:
                        raise RuntimeError(
                            f"short read for {key} in {path} "
                            f"(wanted {n_bytes} bytes, got {read})"
                        )
                    read += got
                if self._drop_caches:
                    advise_drop_caches(f.fileno(), row_start_byte, n_bytes)
            return torch.frombuffer(buf, dtype=dtype).reshape(shape_out)

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
    clone: bool = True,
) -> Optional[torch.Tensor]:
    """Shape-check ``t`` against ``base`` and convert it to a delta.

    ``entry_dir`` / ``key`` are only used for the log messages. Returns
    ``None`` when the tensor is size-incompatible (caller skips it).

    ``clone=False`` skips the defensive copy before the in-place subtraction.
    Only safe when the reader already owns ``t``'s storage: the direct-read
    path clones on every read, while an mmap-backed ``get_tensor`` returns a
    view into the mapped shard, which must never be mutated in place.
    """
    if t.shape != base.shape:
        # embed-style submatrix: truncate when the TV is a superset grid
        fit = shape_fit(tuple(base.shape), tuple(t.shape))
        if fit == "submatrix":
            logger.warning("using submatrix of %s:%s", entry_dir, key)
            t = t[: base.shape[0], : base.shape[1], *base.shape[2:]]
        else:
            logger.warning("skipping %s:%s due to size mismatch", entry_dir, key)
            return None
    if kind == "tv":
        # ``t.to`` is a no-op (same object) when the dtype already matches.
        t = t.to(base.dtype)
        return t if not clone else t.clone()
    # model mode: delta = W_i - base. ``t.to`` is a no-op (same object) when
    # the dtype already matches, so the subtraction stays in place.
    t = t.to(base.dtype)
    if clone:
        t = t.clone()
    t.sub_(base)
    return t


def load_delta(base: torch.Tensor, entry_dir, kind: str, key: str, readers: dict, clone: bool = True):
    """Load ``delta = W_i - base`` (``kind="model"``) or the stored TV (``kind="tv"``).

    Returns ``None`` if the tensor is missing or size-incompatible. The
    returned tensor owns its storage (cloned). ``entry_dir`` is a ``Path`` and
    ``kind`` is ``"tv"`` or ``"model"`` (no ``ModelEntry`` import, so this stays
    import-cycle-free). ``clone=False`` saves one full-tensor copy for callers
    whose reader already returns owned storage (see :func:`_prepare_delta`).
    """
    reader = readers[entry_dir]
    if key not in reader.index.tensor_paths:
        logger.warning("skipping %s:%s (tensor missing)", entry_dir, key)
        return None
    t = reader.get_tensor(key)
    return _prepare_delta(t, base, entry_dir, kind, key, clone=clone)


def load_delta_chunk(
    base_chunk: torch.Tensor,
    entry_dir,
    kind: str,
    key: str,
    readers: dict,
    start: int,
    end: int,
    clone: bool = True,
) -> Optional[torch.Tensor]:
    """Load rows ``[start:end]`` of the tensor as a delta chunk.

    Row ranges are read straight from the shard (no full-tensor load) and go
    through the same shape checks as :func:`load_delta`, so a missing or
    size-incompatible tensor is skipped rather than corrupting the merge.
    Returns ``None`` when the tensor must be skipped. ``clone=False`` saves
    one row-chunk copy for callers whose reader already returns owned storage
    (see :func:`_prepare_delta`).
    """
    reader = readers[entry_dir]
    if key not in reader.index.tensor_paths:
        logger.warning("skipping %s:%s (tensor missing)", entry_dir, key)
        return None
    t = reader.get_slice(key)[start:end]
    return _prepare_delta(t, base_chunk, entry_dir, kind, key, clone=clone)


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


# Bounded transient per tensor write: tensor bytes are appended to the shard
# file in slices of at most this many bytes (one memcpy, freed at once).
_WRITE_SLICE_BYTES = 64 * 1024 * 1024


def _safetensors_header_bytes(entries) -> bytes:
    """Serialize one shard's header in canonical form.

    ``entries``: ``(name, dtype_str, shape, start, nbytes)``. Keys sorted by
    name, compact JSON separators, space-padded to an 8-byte multiple — the
    same framing ``safetensors.torch.save_file`` uses. (``save_file`` itself
    emits header entries in an opaque, implementation-defined order, so
    byte-equality with it is not a goal; values, offsets, layout, and
    determinism are. Canonical sorted order is strictly more reproducible.)
    """
    header = {"__metadata__": {"format": "pt"}}
    for name, dtype_str, shape, start, nbytes in sorted(entries, key=lambda e: e[0]):
        header[name] = {
            "dtype": dtype_str,
            "shape": list(shape),
            "data_offsets": [start, start + nbytes],
        }
    raw = json.dumps(header, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return raw + b" " * (-len(raw) % 8)


class TensorWriter:
    """Writes a model/task-vector as HF sharded safetensors, streaming.

    Produces ``model.safetensors`` (single shard) or
    ``model-00001-of-0000N.safetensors`` + ``model.safetensors.index.json``
    (multiple shards): the same HF layout mergekit's ``TensorWriter``
    produces, with identical per-tensor values and offsets and a canonical
    sorted-name header order (deterministic across runs).

    Unlike a staging writer, tensor bytes are appended straight to the shard
    file as each tensor arrives: output costs one tensor plus a bounded
    write slice, never a staged shard nor a serialization buffer.

    ``manifest`` lists ``(name, dtype_str, shape)`` for every tensor that
    will arrive. Shard grouping follows the same greedy ``max_shard_size``
    rule as the classic writer. Every planned tensor must arrive exactly
    once, in manifest (sorted-name) order — merge passes all base tensors,
    extract passes its arrival-filtered subset (see :func:`shape_fit`).
    Violations raise loudly instead of corrupting offsets.

    ``drop_caches`` releases each completed shard from the page cache
    (see :func:`advise_drop_caches`); the output is never re-read in-run,
    so this is always safe to enable alongside the flag.
    """

    def __init__(
        self,
        out_path: str,
        manifest,
        max_shard_size: int = 1_000_000_000,
        drop_caches: bool = False,
    ) -> None:
        os.makedirs(out_path, exist_ok=True)
        self.out_path = out_path
        self.max_shard_size = max_shard_size
        self.drop_caches = drop_caches

        ordered = sorted(manifest, key=lambda m: m[0])
        # Greedy shard assignment on planned byte sizes — the same rule the
        # staging writer applied to arriving tensors, so the layout is
        # unchanged (arrival always follows manifest order in our callers).
        self._shards: List[dict] = []
        cur: list = []
        cur_size = 0
        for name, dtype_str, shape in ordered:
            nbytes = 1
            for s in shape:
                nbytes *= s
            nbytes *= safetensors_elem_size(dtype_str)
            if max_shard_size > 0 and cur and cur_size + nbytes > max_shard_size:
                self._shards.append(self._new_shard(cur))
                cur, cur_size = [], 0
            cur.append((name, dtype_str, tuple(shape), nbytes))
            cur_size += nbytes
        if cur:
            self._shards.append(self._new_shard(cur))

        self._expected = [m[0] for sh in self._shards for m in sh["members"]]
        self._by_name = {
            m[0]: (si, mi)
            for si, sh in enumerate(self._shards)
            for mi, m in enumerate(sh["members"])
        }
        self._arrived = 0
        self.shards_written = 0
        self.weight_map: Dict[str, str] = {}
        self.total_size = 0

    def _new_shard(self, members: list) -> dict:
        """Precompute one shard's offsets, header bytes, and temp path."""
        offset = 0
        entries = []
        for name, dtype_str, shape, nbytes in members:
            entries.append((name, dtype_str, shape, offset, nbytes))
            offset += nbytes
        idx = len(self._shards)
        return {
            "members": members,
            "header": _safetensors_header_bytes(entries),
            "path": os.path.join(self.out_path, f"model-{idx + 1}.safetensors"),
            "fh": None,
            "arrived": 0,
        }

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.finalize()

    def save_tensor(self, name: str, tensor: torch.Tensor) -> None:
        if self._arrived >= len(self._expected) or name != self._expected[self._arrived]:
            raise RuntimeError(
                f"TensorWriter: unexpected arrival {name!r} "
                "(manifest order violated; offsets would corrupt)"
            )
        if not tensor.is_contiguous():
            tensor = tensor.contiguous()

        si, mi = self._by_name[name]
        shard = self._shards[si]
        nbytes = tensor.numel() * tensor.element_size()
        planned = shard["members"][mi][3]
        if nbytes != planned:
            raise RuntimeError(
                f"TensorWriter: size mismatch for {name!r} "
                f"(got {nbytes} bytes, planned {planned})"
            )
        if shard["fh"] is None:
            fh = open(shard["path"], "wb")
            fh.write(struct.pack("<Q", len(shard["header"])))
            fh.write(shard["header"])
            shard["fh"] = fh
        # Append raw bytes in bounded slices: one memcpy at a time, viewing
        # the tensor as uint8 (no numpy bf16 issue, no full-tensor copy).
        u8 = tensor.view(torch.uint8).reshape(-1)
        for off in range(0, nbytes, _WRITE_SLICE_BYTES):
            shard["fh"].write(u8[off : off + _WRITE_SLICE_BYTES].numpy().tobytes())
        shard["arrived"] += 1
        self._arrived += 1
        self.total_size += nbytes
        if shard["arrived"] == len(shard["members"]):
            self._close_shard(si)

    def _close_shard(self, si: int) -> None:
        """Close a completed shard file. Weight map is rebuilt in finalize."""
        shard = self._shards[si]
        if shard["fh"] is not None:
            shard["fh"].close()
            shard["fh"] = None
        if self.drop_caches:
            try:
                with open(shard["path"], "rb") as f:
                    advise_drop_caches(f.fileno(), 0, os.path.getsize(shard["path"]))
            except OSError:
                pass
        self.shards_written += 1

    def finalize(self) -> None:
        # Close any still-open shard files. A shard whose planned members did
        # not all arrive (caller bug or aborted run) is removed instead of
        # renamed: a gapped file would be corrupt, while absence is loud
        # downstream. In normal runs every shard is already complete here.
        completed: list = []
        for shard in self._shards:
            if shard["fh"] is not None:
                shard["fh"].close()
                shard["fh"] = None
            if shard["arrived"] == len(shard["members"]):
                completed.append(shard)
            else:
                logger.warning(
                    "TensorWriter: dropping incomplete shard %s (%d/%d tensors arrived)",
                    shard["path"],
                    shard["arrived"],
                    len(shard["members"]),
                )
                try:
                    os.remove(shard["path"])
                except OSError:
                    pass

        # Standardize shard names to the Hugging Face format.
        total_shards = len(completed)
        name_remap: Dict[str, str] = {}
        if total_shards == 1:
            name_remap[os.path.basename(completed[0]["path"])] = "model.safetensors"
        else:
            for idx, shard in enumerate(completed):
                old_name = os.path.basename(shard["path"])
                new_name = f"model-{idx + 1:05d}-of-{total_shards:05d}.safetensors"
                name_remap[old_name] = new_name

        self.weight_map = {}
        for shard in completed:
            for name, _, _, _ in shard["members"]:
                self.weight_map[name] = name_remap[os.path.basename(shard["path"])]

        for old_name, new_name in name_remap.items():
            old_path = os.path.join(self.out_path, old_name)
            new_path = os.path.join(self.out_path, new_name)
            if old_path != new_path and os.path.exists(old_path):
                os.rename(old_path, new_path)

        if total_shards > 1:
            index_path = os.path.join(self.out_path, "model.safetensors.index.json")
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
