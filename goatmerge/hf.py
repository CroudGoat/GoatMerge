"""Hugging Face Hub cache resolution for model directories.

``resolve_model_dir`` accepts either a local directory (used as-is) or a
Hugging Face repo id such as ``qwen/qwen3.5-4b``. A repo id is resolved
against the local HF Hub cache *without any network access*: it locates the
most recently updated snapshot of the cached repo and returns that directory.

The cache layout is the standard one used by ``huggingface_hub``:

    $HF_HUB_CACHE/models--<org>--<name>/snapshots/<commit-sha>/...

so a repo id like ``qwen/qwen3.5-4b`` maps to the folder
``models--qwen--qwen3.5-4b`` (``/`` -> ``--``).
"""

from __future__ import annotations

import os
import re
from pathlib import Path

_REPO_ID_RE = re.compile(r"^[A-Za-z0-9._-]+/[A-Za-z0-9._-]+$")


def is_repo_id(s: str) -> bool:
    """True if ``s`` looks like a Hugging Face repo id (``org/name``)."""
    return bool(_REPO_ID_RE.match(s))


def hf_hub_cache_root() -> Path:
    """The HF Hub cache root: ``$HF_HUB_CACHE`` or ``$HUGGINGFACE_HUB_CACHE``,
    defaulting to ``~/.cache/huggingface/hub``."""
    env = os.environ.get("HF_HUB_CACHE") or os.environ.get("HUGGINGFACE_HUB_CACHE")
    if env:
        return Path(os.environ.expandvars(os.environ.expanduser(env)))
    return Path.home() / ".cache" / "huggingface" / "hub"


def repo_folder_name(repo_id: str) -> str:
    """Serialize a repo id to its cache folder name (huggingface_hub's rule):
    ``models--<org>--<name>``, i.e. ``/`` -> ``--``."""
    return "--".join(["models", *repo_id.split("/")])


def resolve_model_dir(s: str) -> Path:
    """Resolve a model reference to a directory.

    * If ``s`` is an existing local directory, it is returned unchanged.
    * If ``s`` is a Hugging Face repo id, the most recent snapshot of that
      repo in the local HF cache is returned.
    * Otherwise a ``FileNotFoundError`` is raised with an actionable message.
    """
    p = Path(s)
    if p.is_dir():
        return p

    if is_repo_id(s):
        root = hf_hub_cache_root()
        snapshots = root / repo_folder_name(s) / "snapshots"
        candidates = [d for d in snapshots.glob("*") if d.is_dir()]
        if not candidates:
            raise FileNotFoundError(
                f"HF repo '{s}' is not in the local HF cache "
                f"({root / repo_folder_name(s)}). "
                f"Download it first, e.g.: huggingface-cli download {s}"
            )
        # Most recently updated snapshot wins (handles multiple revisions).
        return max(candidates, key=lambda d: d.stat().st_mtime)

    raise FileNotFoundError(
        f"Model directory not found: {s} "
        f"(neither an existing local directory nor a cached HF repo id)"
    )
