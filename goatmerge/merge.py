"""Streaming task-arithmetic merge engine (no ``torch.stack``).

Per tensor the engine streams each task vector (or source-model tensor)
exactly once per pass:

    acc  = sum_i alpha_i * delta_i          (in place, base dtype)
    l1   = sum_i |alpha_i * delta_i|       (in place, base dtype)
    c    = sum_i sign(alpha_i * delta_i)   (int8, count consensus only)

Peak resident memory is base + acc + l1 + c + 1 delta + sparsify temp
(~5-7 S, S = tensor bytes) instead of mergekit's (4k+4)S..(6k+7)S.

Parity with mergekit GTA:
  - no consensus:  result = base + lambda * acc / sum(alpha_i)   (normalize)
                   (bit-exact: same sequential add order as the reference)
  - consensus:     mixed = (acc + M*l1)/2  (exact masked-sum identity),
                   divisor recomputed in a second streamed pass.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import torch

from .consensus import ConsensusAccumulator, ConsensusMethod
from .io import ShardReader, ShardedTensorIndex, TensorWriter
from .sparsify import RescaleNorm, SparsificationMethod, sparsify_inplace

logger = logging.getLogger("goatmerge.merge")


@dataclass(frozen=True)
class MergeSettings:
    density: float = 1.0
    method: Optional[SparsificationMethod] = None
    n: int = 64
    m: int = 256
    gamma: float = 0.0
    epsilon: float = 0.0
    rescale: bool = True
    normalize: bool = True
    lambda_: float = 1.0
    consensus: ConsensusMethod = ConsensusMethod.none
    chunk_elements: Optional[int] = None


@dataclass(frozen=True)
class ModelEntry:
    """One fine-tuned source: a task-vector dir (kind='tv') or a model dir
    (kind='model'), with its merge weight."""

    dir: Path
    kind: str  # "tv" | "model"
    weight: float


def _sparsify(delta: torch.Tensor, s: MergeSettings) -> None:
    if s.method is None:
        return
    rescale_norm = RescaleNorm.l1 if s.rescale else None
    sparsify_inplace(
        delta,
        density=s.density,
        method=s.method,
        n=s.n,
        m=s.m,
        gamma=s.gamma,
        epsilon=s.epsilon,
        rescale_norm=rescale_norm,
        chunk_elements=s.chunk_elements,
    )


def load_delta(
    base: torch.Tensor,
    entry: ModelEntry,
    key: str,
    readers: dict,
) -> Optional[torch.Tensor]:
    """Load ``delta = W_i - base`` for one entry, or the stored TV.

    Returns None if the tensor is missing or size-incompatible (the caller
    records the skip). The returned tensor owns its storage (cloned).
    """
    reader = readers[entry.dir]
    if key not in reader.index.tensor_paths:
        logger.warning("skipping %s:%s (tensor missing)", entry.dir, key)
        return None
    t = reader.get_tensor(key)
    if t.shape != base.shape:
        # embed-style submatrix: truncate when the TV is a superset grid
        if (
            base.dim() >= 2
            and t.dim() >= 2
            and t.shape[0] >= base.shape[0]
            and t.shape[1] >= base.shape[1]
            and t.shape[2:] == base.shape[2:]
        ):
            logger.warning("using submatrix of %s:%s", entry.dir, key)
            t = t[: base.shape[0], : base.shape[1], *base.shape[2:]]
        else:
            logger.warning("skipping %s:%s due to size mismatch", entry.dir, key)
            return None
    if entry.kind == "tv":
        return t.to(base.dtype).clone()
    # model mode: delta = W_i - base (in place after clone)
    t = t.to(base.dtype).clone()
    t.sub_(base)
    return t


def merge_tensor(
    base: torch.Tensor,
    entries: List[ModelEntry],
    key: str,
    readers: dict,
    settings: MergeSettings,
) -> torch.Tensor:
    """Merge one tensor: base + (consensus-masked) weighted sum of deltas."""
    if not entries:
        return base

    acc = ConsensusAccumulator(base, settings.consensus)

    # ---- pass 1: stream deltas, accumulate acc / l1 / c ----
    for entry in entries:
        delta = load_delta(base, entry, key, readers)
        if delta is None:
            continue
        _sparsify(delta, settings)
        acc.accumulate(delta, entry.weight)
        del delta

    # ---- no consensus: result = base + lambda * acc / sum(alpha_i) ----
    if settings.consensus == ConsensusMethod.none:
        mixed = acc.acc
        if settings.normalize:
            wsum = sum(e.weight for e in entries)
            if abs(wsum) < 1e-8:
                wsum = 1.0
            mixed.div_(wsum)
        if settings.lambda_ != 1.0:
            mixed.mul_(settings.lambda_)
        return (base + mixed).to(base.dtype)

    # ---- consensus: mixed = (acc + M*l1)/2 in place ----
    mixed = acc.masked_sum_inplace()
    del acc.l1  # consumed by the identity; free it

    # ---- pass 2: stream deltas again for the per-element divisor ----
    divisor = torch.zeros_like(base)
    for entry in entries:
        delta = load_delta(base, entry, key, readers)
        if delta is None:
            continue
        _sparsify(delta, settings)
        acc.divisor_accumulate(divisor, delta, entry.weight)
        del delta

    divisor[divisor == 0] = 1
    if settings.normalize:
        mixed.div_(divisor)
    if settings.lambda_ != 1.0:
        mixed.mul_(settings.lambda_)
    return (base + mixed).to(base.dtype)


def merge_model(
    base_dir: Path,
    entries: List[ModelEntry],
    settings: MergeSettings,
    out_dir: Path,
) -> dict:
    """Merge a whole model directory, streaming tensor by tensor.

    Returns a summary dict (tensor count, skipped tensors, elapsed).
    """
    base_index = ShardedTensorIndex.from_dir(Path(base_dir))
    tensor_names = base_index.keys()
    if not tensor_names:
        raise ValueError(f"No tensors found in {base_dir}")

    readers: dict = {}
    readers[Path(base_dir)] = ShardReader(base_index)
    for entry in entries:
        readers[Path(entry.dir)] = ShardReader(ShardedTensorIndex.from_dir(Path(entry.dir)))

    skipped: List[str] = []
    writer = TensorWriter(str(out_dir))
    try:
        base_reader = readers[Path(base_dir)]
        for i, key in enumerate(tensor_names):
            base = base_reader.get_tensor(key).clone()
            result = merge_tensor(base, entries, key, readers, settings)
            writer.save_tensor(key, result)
            logger.info("tensor %d/%d: %s", i + 1, len(tensor_names), key)
    finally:
        for r in readers.values():
            r.close()
        # Flush the writer: staged tensors are only written to disk here.
        writer.finalize()

    return {
        "num_tensors": len(tensor_names),
        "tensor_names": tensor_names,
        "skipped_tensors": skipped,
    }
