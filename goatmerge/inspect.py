"""Inspection and verification of task vectors / merged models."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List

from .fingerprint import compute_base_fingerprint, model_manifest
from .io import ShardedTensorIndex
from .metadata import load_metadata, validate_metadata

logger = logging.getLogger("goatmerge.inspect")


def inspect_dir(path: Path) -> Dict[str, Any]:
    """Return the GoatMerge sidecar + tensor manifest of a TV / merged model."""
    path = Path(path)
    meta = load_metadata(path)
    validate_metadata(meta)
    index = ShardedTensorIndex.from_dir(path)
    manifest = model_manifest(index)
    return {
        "path": str(path),
        "metadata": meta,
        "manifest": {name: (shape, dtype) for (name, shape, dtype) in manifest},
    }


def verify_against_base(base_dir: Path, tv_dirs: List[Path]) -> List[str]:
    """Check each TV's stored base fingerprint against ``base_dir``.

    Returns the list of TV dirs whose fingerprint does not match (empty
    list = all consistent).
    """
    base_fp = compute_base_fingerprint(Path(base_dir))
    mismatches: List[str] = []
    for tv in tv_dirs:
        meta = load_metadata(Path(tv))
        stored = meta.get("base_fingerprint")
        if stored != base_fp:
            mismatches.append(str(tv))
            logger.warning(
                "fingerprint mismatch for %s (stored %s... vs base %s...)",
                tv,
                stored[:12] if stored else "<none>",
                base_fp[:12],
            )
    return mismatches
