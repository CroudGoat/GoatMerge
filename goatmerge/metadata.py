"""GoatMerge provenance envelope (``goatmerge.json``).

GoatMerge writes a small self-describing sidecar next to the safetensors it
produces:

    <out>/goatmerge.json          GoatMerge metadata
    <out>/model.safetensors       merged weights
    <out>/config.json             copied from the base model (merged models)

``goatmerge.json`` deliberately does **not** use the name ``config.json``:
that name belongs to the HuggingFace model config, and a merged model must
stay directly loadable by ``transformers``. The sidecar records only what is
needed to reason about the artifact afterwards — which base it came from, and
(for task vectors) the base fingerprint used by the merge-time verification.

Merge *recipes* live in YAML (``goatmerge merge -c recipe.yaml``); this file
records what a directory actually contains, not how to reproduce it.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

GOATMERGE_FORMAT_VERSION = "1.1.0"
METADATA_NAME = "goatmerge.json"

# Name used by GoatMerge <= 0.1.0. Its GoatMerge metadata lives under this
# name, which collides with the HuggingFace model config, so it is only read
# (never written) and only when the content is recognizably GoatMerge's.
LEGACY_METADATA_NAME = "config.json"


def build_metadata(
    *,
    base_model: str,
    source_model: str,
    base_fingerprint: str,
    created_utc: Optional[float] = None,
) -> Dict[str, Any]:
    """Assemble the ``goatmerge.json`` payload for a task vector.

    Only the identity of the base it was extracted against is recorded — the
    tensor manifest is derivable from the shards themselves.
    """
    return {
        "goatmerge_format_version": GOATMERGE_FORMAT_VERSION,
        "model_type": "task_vector",
        "base_model": base_model,
        "source_model": source_model,
        "base_fingerprint": base_fingerprint,
        "created_utc": created_utc if created_utc is not None else _now(),
        "description": "Task vector: W - W_base, relative to the base model above.",
    }


def build_merged_metadata(
    *,
    base_model: str,
    task_vectors: List[Dict[str, Any]],
    base_fingerprint: Optional[str] = None,
    merge_settings: Optional[Dict[str, Any]] = None,
    created_utc: Optional[float] = None,
) -> Dict[str, Any]:
    """Assemble the ``goatmerge.json`` payload for a merged model."""
    return {
        "goatmerge_format_version": GOATMERGE_FORMAT_VERSION,
        "model_type": "merged_model",
        "base_model": base_model,
        "base_fingerprint": base_fingerprint,
        "task_vectors": task_vectors,
        "merge_settings": merge_settings or {},
        "created_utc": created_utc if created_utc is not None else _now(),
        "description": "Merged model: W_base + sum_i alpha_i * T_i.",
    }


def _now() -> float:
    import time

    return time.time()


def load_metadata(directory: Path) -> Dict[str, Any]:
    """Load the ``goatmerge.json`` sidecar of a task vector / merged model.

    A ``config.json`` written by GoatMerge <= 0.1.0 is still accepted (its
    content carries ``goatmerge_format_version``), so task vectors extracted
    by an older version keep working. A HuggingFace ``config.json`` is *not*
    metadata and is rejected.
    """
    directory = Path(directory)
    primary = directory / METADATA_NAME
    if primary.exists():
        with primary.open("r", encoding="utf-8") as f:
            return json.load(f)

    legacy = directory / LEGACY_METADATA_NAME
    if legacy.exists():
        with legacy.open("r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict) and "goatmerge_format_version" in data:
            return data

    raise FileNotFoundError(
        f"No {METADATA_NAME} in {directory} (expected a GoatMerge task vector "
        f"or merged model)"
    )


def write_metadata(directory: Path, metadata: Dict[str, Any]) -> Path:
    """Write the ``goatmerge.json`` sidecar into ``directory``."""
    directory = Path(directory)
    cfg = directory / METADATA_NAME
    with cfg.open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    return cfg


def validate_metadata(meta: Dict[str, Any]) -> None:
    """Raise if the metadata envelope is malformed / missing required fields."""
    if meta.get("model_type") not in ("task_vector", "merged_model"):
        raise ValueError(f"Unknown model_type: {meta.get('model_type')!r}")
    if not meta.get("goatmerge_format_version"):
        raise ValueError("Metadata missing fields: ['goatmerge_format_version']")
    if not meta.get("base_model"):
        raise ValueError("Metadata missing fields: ['base_model']")
    if meta["model_type"] == "task_vector":
        for field in ("source_model", "base_fingerprint"):
            if not meta.get(field):
                raise ValueError(f"Metadata missing fields: ['{field}']")
    if not isinstance(meta.get("task_vectors"), list) and meta["model_type"] == "merged_model":
        raise ValueError("task_vectors must be a list")
