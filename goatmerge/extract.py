"""Task-vector extraction: T = W_source - W_base, saved once as safetensors.

The TV is stored in the base dtype and carries the base fingerprint, so
later merges stream only the deltas (no re-reading of the source model).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List

from .fingerprint import compute_base_fingerprint
from .io import ShardReader, ShardedTensorIndex, TensorWriter
from .metadata import build_metadata, write_metadata

logger = logging.getLogger("goatmerge.extract")


def extract_task_vector(
    base_dir: Path,
    source_dir: Path,
    out_dir: Path,
) -> dict:
    """Extract ``T = W_source - W_base`` into ``out_dir``.

    Returns a summary dict: tensor names, skipped tensors, base fingerprint.
    """
    base_dir = Path(base_dir)
    source_dir = Path(source_dir)
    out_dir = Path(out_dir)

    base_index = ShardedTensorIndex.from_dir(base_dir)
    src_index = ShardedTensorIndex.from_dir(source_dir)
    base_reader = ShardReader(base_index)
    src_reader = ShardReader(src_index)
    writer = TensorWriter(str(out_dir))

    tensor_names: List[str] = []
    skipped: List[str] = []

    try:
        for key in base_index.keys():
            if key not in src_index.tensor_paths:
                logger.warning("skipping %s (missing in source)", key)
                skipped.append(key)
                continue
            b = base_reader.get_tensor(key)
            s = src_reader.get_tensor(key)
            if s.shape != b.shape:
                if (
                    b.dim() >= 2
                    and s.dim() >= 2
                    and s.shape[0] >= b.shape[0]
                    and s.shape[1] >= b.shape[1]
                    and s.shape[2:] == b.shape[2:]
                ):
                    logger.warning("using submatrix of %s", key)
                    s = s[: b.shape[0], : b.shape[1], *b.shape[2:]]
                else:
                    logger.warning("skipping %s due to size mismatch", key)
                    skipped.append(key)
                    continue
            delta = s.to(b.dtype) - b
            writer.save_tensor(key, delta)
            tensor_names.append(key)
    finally:
        base_reader.close()
        src_reader.close()
        # Flush the writer: tensors staged in current_shard are only written
        # to disk here (a TensorWriter is not a context manager in this call).
        writer.finalize()

    if not tensor_names:
        raise ValueError("no tensors extracted (all skipped?)")

    fingerprint = compute_base_fingerprint(base_dir)
    meta = build_metadata(
        base_model=str(base_dir),
        source_model=str(source_dir),
        base_fingerprint=fingerprint,
    )
    write_metadata(out_dir, meta)

    return {
        "tensor_names": tensor_names,
        "skipped_tensors": skipped,
        "base_fingerprint": fingerprint,
    }
