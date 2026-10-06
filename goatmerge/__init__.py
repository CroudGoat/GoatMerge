"""GoatMerge: streaming task-arithmetic merge engine.

Full-scratch implementation of the improvements proposed in ISSUE.md:
no ``torch.stack``, single-pass sign consensus, topk-based sparsification,
uint8 masks, BS (n:m) pruning, chunked global pruning, HF-sharded I/O,
fingerprint verification, metadata envelopes, and a standalone CLI.
"""

from .consensus import ConsensusAccumulator, ConsensusMethod
from .extract import extract_task_vector
from .fingerprint import (
    compute_base_fingerprint,
    model_manifest,
    tensor_content_hash,
    tensor_names_and_shapes,
)
from .hf import is_repo_id, resolve_model_dir
from .inspect import inspect_dir, verify_against_base
from .io import ShardReader, ShardedTensorIndex, TensorWriter
from .merge import ModelEntry, MergeSettings, merge_model, merge_tensor
from .metadata import (
    build_metadata,
    build_merged_metadata,
    load_metadata,
    validate_metadata,
    write_metadata,
)
from .sparsify import (
    RescaleNorm,
    SparsificationMethod,
    bs_mask,
    bs_mask_chunked,
    magnitude_mask,
    magnitude_mask_chunked,
    magnitude_threshold,
    sparsify_inplace,
)

__all__ = [
    "ConsensusAccumulator",
    "ConsensusMethod",
    "ModelEntry",
    "MergeSettings",
    "RescaleNorm",
    "SparsificationMethod",
    "ShardReader",
    "ShardedTensorIndex",
    "TensorWriter",
    "build_metadata",
    "build_merged_metadata",
    "compute_base_fingerprint",
    "extract_task_vector",
    "inspect_dir",
    "load_metadata",
    "magnitude_mask",
    "magnitude_mask_chunked",
    "magnitude_threshold",
    "merge_model",
    "merge_tensor",
    "model_manifest",
    "sparsify_inplace",
    "tensor_content_hash",
    "tensor_names_and_shapes",
    "verify_against_base",
    "write_metadata",
]

__version__ = "0.1.0"
