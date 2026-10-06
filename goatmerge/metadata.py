"""Task-vector / merged-model metadata envelope (``config.json``).

A task vector directory is laid out like a small HF model directory:

    config.json
    model.safetensors.index.json
    model-00001-of-0000N.safetensors

``config.json`` carries everything needed to validate and reason about the
vector: which base it was extracted against, the source model, the dtype,
the tensor manifest, and the base fingerprint used for integrity checks.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch

GOATMERGE_FORMAT_VERSION = "1.0.0"
CONFIG_NAME = "config.json"


def build_metadata(
    *,
    base_model: str,
    source_model: str,
    dtype: torch.dtype,
    tensor_names: List[str],
    base_fingerprint: str,
    architecture: Optional[dict] = None,
    created_utc: Optional[float] = None,
) -> Dict[str, Any]:
    """Assemble the ``config.json`` payload for a task vector."""
    return {
        "goatmerge_format_version": GOATMERGE_FORMAT_VERSION,
        "model_type": "task_vector",
        "framework": "torch",
        "base_model": base_model,
        "source_model": source_model,
        "dtype": str(dtype),
        "torch_dtype": str(dtype),
        "num_tensors": len(tensor_names),
        "tensor_names": list(tensor_names),
        "base_fingerprint": base_fingerprint,
        "architecture": architecture or {},
        "created_utc": created_utc if created_utc is not None else time.time(),
        "description": "Task vector: W - W_base relative to a base model.",
    }


def build_merged_metadata(
    *,
    base_model: str,
    task_vectors: List[Dict[str, Any]],
    tensor_names: List[str],
    dtype: Optional[torch.dtype] = None,
    merge_settings: Dict[str, Any] | None = None,
    skipped_tensors: List[str] | None = None,
    created_utc: Optional[float] = None,
) -> Dict[str, Any]:
    """Assemble the ``config.json`` payload for a merged model."""
    return {
        "goatmerge_format_version": GOATMERGE_FORMAT_VERSION,
        "model_type": "merged_model",
        "framework": "torch",
        "base_model": base_model,
        "task_vectors": task_vectors,
        "dtype": str(dtype) if dtype is not None else None,
        "torch_dtype": str(dtype) if dtype is not None else None,
        "num_tensors": len(tensor_names),
        "tensor_names": list(tensor_names),
        "merge_settings": merge_settings or {},
        "skipped_tensors": skipped_tensors or [],
        "created_utc": created_utc if created_utc is not None else time.time(),
        "description": "Merged model: W_base + sum_i alpha_i * T_i.",
    }


def load_metadata(directory: Path) -> Dict[str, Any]:
    cfg = Path(directory) / CONFIG_NAME
    if not cfg.exists():
        raise FileNotFoundError(
            f"No {CONFIG_NAME} in {directory} (expected a task vector or merged model)"
        )
    with cfg.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_metadata(directory: Path, metadata: Dict[str, Any]) -> Path:
    cfg = Path(directory) / CONFIG_NAME
    with cfg.open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    return cfg


def validate_metadata(meta: Dict[str, Any]) -> None:
    """Raise if the metadata envelope is malformed / missing required fields."""
    required = [
        "goatmerge_format_version",
        "model_type",
        "base_model",
        "dtype",
        "num_tensors",
        "tensor_names",
    ]
    if meta.get("model_type") == "task_vector":
        required += ["source_model", "base_fingerprint"]
    missing = [k for k in required if k not in meta]
    if missing:
        raise ValueError(f"Metadata missing fields: {missing}")
    if meta["model_type"] not in ("task_vector", "merged_model"):
        raise ValueError(f"Unknown model_type: {meta['model_type']!r}")
    if not isinstance(meta["tensor_names"], list) or not meta["tensor_names"]:
        raise ValueError("tensor_names must be a non-empty list")
    if meta["num_tensors"] != len(meta["tensor_names"]):
        raise ValueError(
            f"num_tensors ({meta['num_tensors']}) != len(tensor_names) ({len(meta['tensor_names'])})"
        )
