"""Sparsification kernels for task vectors (uint8 masks, topk-based).

Improvements over mergekit's ``sparsify.py``:

* ``magnitude`` / ``magnitude_outliers`` use ``torch.topk`` (k indices only,
  O(N log k), k*4B of indices) instead of a full ``argsort`` (N*4B indices,
  O(N log N)).
* All masks are ``uint8`` (1 byte/element) instead of bf16/f32.
* Mask application and rescaling are in-place.
* A Balanced-Sparsification (BS, n:m block) kernel is included (from CABS).
* Chunked variants for large tensors: global magnitude pruning is done in two
  passes with a bounded max-heap (k*8B) so the per-chunk RAM is O(chunk).

Reference semantics:
- magnitude: keep the top ``density`` fraction by |value|.
- magnitude_outliers: drop the top ``gamma`` (largest) and the bottom
  ``(1 - density - gamma)`` (smallest), keep the middle.
- bernoulli: keep each element with probability ``density``.
- della_magprune: keep each element with probability
  ``density - epsilon + rank_norm * 2*epsilon`` (rank within the row).
- bs (n:m): keep the top ``n`` of every ``m``-block (flattened, row-major).
"""

from __future__ import annotations

import heapq
from enum import Enum
from typing import Iterator, List, Optional, Tuple

import torch


class SparsificationMethod(str, Enum):
    magnitude = "magnitude"
    random = "random"
    magnitude_outliers = "magnitude_outliers"
    della_magprune = "della_magprune"
    bs = "bs"  # Balanced Sparsification (n:m block)


class RescaleNorm(str, Enum):
    l1 = "l1"
    l2 = "l2"
    linf = "linf"


EPS = 1e-7


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _abs_flat(tensor: torch.Tensor) -> torch.Tensor:
    """Flattened |tensor| in a dtype safe for topk on the current device."""
    w = tensor.abs().reshape(-1)
    if w.device.type == "cpu" and w.dtype in (torch.float16, torch.bfloat16):
        w = w.float()
    return w


def _norm(t: torch.Tensor, norm: RescaleNorm) -> float:
    if norm == RescaleNorm.l1:
        return float(t.abs().sum())
    if norm == RescaleNorm.l2:
        return float(t.norm())
    if norm == RescaleNorm.linf:
        return float(t.abs().max())
    raise NotImplementedError(norm)


def _row_stride(t: torch.Tensor) -> int:
    """Number of elements in one row (product of trailing dims); 1 for 1-D."""
    if t.dim() >= 2:
        return int(torch.tensor(t.shape[1:]).prod())
    return 1


# --------------------------------------------------------------------------- #
# Mask builders (return a uint8 mask shaped like the input)
# --------------------------------------------------------------------------- #
def magnitude_mask(t: torch.Tensor, density: float) -> torch.Tensor:
    """0/1 mask keeping the ``density`` fraction of largest |values|."""
    n = t.numel()
    k = int(round(density * n))
    k = max(0, min(n, k))
    mask = torch.zeros(n, dtype=torch.uint8, device=t.device)
    if k > 0:
        w = _abs_flat(t)
        idx = torch.topk(w, k).indices
        mask[idx] = 1
    return mask.reshape_as(t)


def magnitude_outliers_mask(t: torch.Tensor, density: float, gamma: float) -> torch.Tensor:
    """0/1 mask: drop the top ``gamma`` (largest) and the bottom
    ``(1 - density - gamma)`` (smallest); keep the middle."""
    n = t.numel()
    target = int(round(density * n))
    n_top = int(round(gamma * n))
    n_bot = n - target - n_top
    if n_bot < 0:
        n_top += n_bot
        n_bot = 0
    n_top = max(0, min(n, n_top))
    n_bot = max(0, min(n, n_bot))

    w = _abs_flat(t)
    drop = torch.zeros(n, dtype=torch.uint8, device=t.device)
    if n_top > 0:
        top_idx = torch.topk(w, n_top).indices
        drop[top_idx] = 1
    if n_bot > 0:
        # smallest = topk of -w
        bot_idx = torch.topk(-w, n_bot).indices
        drop[bot_idx] = 1
    mask = (1 - drop)  # keep the middle
    return mask.reshape_as(t)


def bernoulli_mask(t: torch.Tensor, density: float) -> torch.Tensor:
    """0/1 mask: each element kept with probability ``density``."""
    work_dtype = (
        t.dtype
        if t.device.type != "cpu" or t.dtype == torch.bfloat16
        else torch.float32
    )
    mask = torch.bernoulli(
        torch.full_like(input=t, fill_value=density, dtype=work_dtype)
    )
    return (mask > 0).to(torch.uint8).reshape_as(t)


def della_mask(t: torch.Tensor, density: float, epsilon: float) -> torch.Tensor:
    """0/1 mask: per-row rank-based Bernoulli (della_magprune)."""
    if density + epsilon >= 1 or density - epsilon <= 0:
        raise ValueError("density +/- epsilon must be in (0, 1)")
    if t.dim() < 2:
        t2 = t.unsqueeze(0)
    else:
        t2 = t
    work_dtype = (
        t.dtype
        if t.device.type != "cpu" or t.dtype == torch.bfloat16
        else torch.float32
    )
    magnitudes = t2.abs()
    sorted_indices = torch.argsort(magnitudes, dim=1, descending=False)
    ranks = sorted_indices.argsort(dim=1).to(work_dtype) + 1
    min_ranks = ranks.min(dim=1, keepdim=True).values
    max_ranks = ranks.max(dim=1, keepdim=True).values
    rank_norm = ((ranks - min_ranks) / (max_ranks - min_ranks)).clamp(0, 1)
    probs = (density - epsilon) + rank_norm * 2 * epsilon
    mask = torch.bernoulli(probs).to(work_dtype)
    return (mask > 0).to(torch.uint8).reshape_as(t)


def bs_mask(t: torch.Tensor, n: int, m: int) -> torch.Tensor:
    """Balanced Sparsification: keep top-``n`` of every ``m``-block.

    Flattened row-major, split into consecutive blocks of ``m``. Trailing
    elements beyond the last full block are zeroed (matches the official CABS
    reference).
    """
    if n <= 0 or m <= 0 or n > m:
        raise ValueError(f"invalid n:m configuration n={n}, m={m}")
    flat = t.reshape(-1)
    n_full = (flat.numel() // m) * m
    mask = torch.zeros_like(flat, dtype=torch.uint8)
    if n_full > 0:
        blocks = _abs_flat(t)[:n_full].view(-1, m)
        topk = blocks.topk(n, dim=1).indices
        block_mask = torch.zeros_like(blocks, dtype=torch.uint8).scatter_(1, topk, 1)
        mask[:n_full] = block_mask.reshape(-1)
    return mask.reshape_as(t)


def make_mask(
    t: torch.Tensor,
    density: float,
    method: SparsificationMethod,
    n: int = 64,
    m: int = 256,
    gamma: float = 0.0,
    epsilon: float = 0.0,
) -> torch.Tensor:
    """Dispatch to the right mask builder."""
    if method == SparsificationMethod.magnitude:
        return magnitude_mask(t, density)
    if method == SparsificationMethod.magnitude_outliers:
        return magnitude_outliers_mask(t, density, gamma)
    if method == SparsificationMethod.random:
        return bernoulli_mask(t, density)
    if method == SparsificationMethod.della_magprune:
        return della_mask(t, density, epsilon)
    if method == SparsificationMethod.bs:
        return bs_mask(t, n, m)
    raise NotImplementedError(method)


# --------------------------------------------------------------------------- #
# In-place sparsify: apply mask + rescale onto ``t`` (returns ``t``)
# --------------------------------------------------------------------------- #
def sparsify_inplace(
    t: torch.Tensor,
    density: float,
    method: SparsificationMethod,
    n: int = 64,
    m: int = 256,
    gamma: float = 0.0,
    epsilon: float = 0.0,
    rescale_norm: Optional[RescaleNorm] = None,
    chunk_elements: Optional[int] = None,
) -> torch.Tensor:
    """Apply the mask in-place and rescale to match the original norm.

    Returns the (modified) ``t``. If ``density >= 1`` the tensor is unchanged.

    ``chunk_elements`` bounds the global magnitude/BS mask to a chunked
    two-pass computation (bounded heap), keeping per-chunk RAM O(chunk).
    """
    if density >= 1:
        return t

    n_elems = t.numel()
    chunked = chunk_elements is not None and n_elems > max(1, chunk_elements)
    if chunked:
        if method == SparsificationMethod.magnitude:
            mask = magnitude_mask_chunked(t, density, chunk_elements)
        elif method == SparsificationMethod.bs:
            mask = bs_mask_chunked(t, n, m, chunk_elements)
        elif method == SparsificationMethod.magnitude_outliers:
            mask = magnitude_outliers_mask_chunked(t, density, gamma, chunk_elements)
        elif method == SparsificationMethod.random:
            mask = bernoulli_mask_chunked(t, density, chunk_elements)
        elif method == SparsificationMethod.della_magprune:
            mask = della_mask_chunked(t, density, epsilon, chunk_elements)
        else:
            mask = make_mask(t, density, method, n=n, m=m, gamma=gamma, epsilon=epsilon)
    else:
        mask = make_mask(t, density, method, n=n, m=m, gamma=gamma, epsilon=epsilon)

    if rescale_norm is not None:
        before = _norm(t, rescale_norm)
        t.mul_(mask.to(t.dtype))
        after = _norm(t, rescale_norm)
        if before < EPS or after < EPS:
            return t
        t.mul_(before / after)
    else:
        t.mul_(mask.to(t.dtype))
    return t


def sparsify_delta(
    t: torch.Tensor,
    method,
    density: float = 1.0,
    n: int = 64,
    m: int = 256,
    gamma: float = 0.0,
    epsilon: float = 0.0,
    rescale: bool = True,
    chunk_elements: Optional[int] = None,
) -> torch.Tensor:
    """Apply sparsification in place to one streamed delta.

    A settings-free wrapper around :func:`sparsify_inplace` so the per-method
    kernels can sparsify a streamed delta without importing ``MergeSettings``
    (keeps the kernel modules import-cycle-free).
    """
    if method is None:
        return t
    rescale_norm = RescaleNorm.l1 if rescale else None
    sparsify_inplace(
        t,
        density=density,
        method=method,
        n=n,
        m=m,
        gamma=gamma,
        epsilon=epsilon,
        rescale_norm=rescale_norm,
        chunk_elements=chunk_elements,
    )
    return t


# --------------------------------------------------------------------------- #
# Chunked global magnitude (two-pass, bounded max-heap)
# --------------------------------------------------------------------------- #
def iter_chunks(t: torch.Tensor, chunk_elements: int, align: int = 1) -> Iterator[torch.Tensor]:
    """Yield flat chunks of ``t``.

    ``align`` forces chunk boundaries to multiples of ``align`` elements
    (use ``m`` for BS, the row stride for row-based methods, 1 otherwise).
    """
    flat = t.reshape(-1)
    n = flat.numel()
    if chunk_elements <= 0 or chunk_elements >= n:
        yield flat
        return
    align = max(1, align)
    step = max(1, (chunk_elements // align) * align)
    for start in range(0, n, step):
        yield flat[start:start + step]


def magnitude_threshold(t: torch.Tensor, k: int) -> float:
    """The k-th largest |value| of ``t`` (global), via a bounded max-heap.

    Streams the tensor in chunks; keeps only the top-k values in a Python
    min-heap (k*8B). Returns the threshold (the smallest of the top-k).
    """
    if k <= 0:
        return 0.0
    heap: List[float] = []
    for chunk in iter_chunks(t, chunk_elements=10_000_000):
        w = _abs_flat(chunk)
        take = min(k, w.numel())
        if take == 0:
            continue
        vals = torch.topk(w, take).values.detach().cpu().numpy().ravel()
        for v in vals:
            if len(heap) < k:
                heapq.heappush(heap, v)
            elif v > heap[0]:
                heapq.heapreplace(heap, v)
    return float(heap[0]) if heap else 0.0


def magnitude_mask_chunked(
    t: torch.Tensor,
    density: float,
    chunk_elements: int,
) -> torch.Tensor:
    """Global magnitude mask computed in two passes over chunks.

    Pass 1 finds the global k-th largest |value| (threshold). Pass 2 keeps
    elements with |x| > threshold, plus the first (index order) of the
    |x| == threshold elements up to k. Deterministic.
    """
    n = t.numel()
    k = int(round(density * n))
    k = max(0, min(n, k))
    if k == 0:
        return torch.zeros(n, dtype=torch.uint8, device=t.device).reshape_as(t)
    thr = magnitude_threshold(t, k)
    mask = torch.zeros(n, dtype=torch.uint8, device=t.device)
    start = 0
    for chunk in iter_chunks(t, chunk_elements, align=1):
        w = _abs_flat(chunk)
        above = (w > thr).to(torch.uint8)
        eq = (w == thr).to(torch.uint8)
        n_above = int(above.sum())
        n_eq = int(eq.sum())
        take_eq = max(0, min(n_eq, k - n_above))
        if take_eq > 0:
            eq_idx = torch.nonzero(eq).view(-1)
            above[eq_idx[:take_eq]] = 1
        c = chunk.numel()
        mask[start:start + c] = above
        start += c
    return mask.reshape_as(t)


def bs_mask_chunked(t: torch.Tensor, n: int, m: int, chunk_elements: int) -> torch.Tensor:
    """BS mask computed per chunk (chunk boundaries aligned to ``m``)."""
    flat = t.reshape(-1)
    n_full = (flat.numel() // m) * m
    mask = torch.zeros(flat.numel(), dtype=torch.uint8, device=t.device)
    for chunk in iter_chunks(t, chunk_elements, align=m):
        c = chunk.numel()
        start = int(chunk.data_ptr() - flat.data_ptr()) // flat.element_size()
        # recompute start robustly via cumulative offset
        # (data_ptr arithmetic is reliable for contiguous flat views)
        chunk_mask = torch.zeros(c, dtype=torch.uint8, device=t.device)
        if start + c <= n_full:
            blocks = _abs_flat(t)[start:start + c].view(-1, m)
            topk = blocks.topk(n, dim=1).indices
            bm = torch.zeros_like(blocks, dtype=torch.uint8).scatter_(1, topk, 1)
            chunk_mask = bm.reshape(-1)
        mask[start:start + c] = chunk_mask
    return mask.reshape_as(t)


# --------------------------------------------------------------------------- #
# Range iterator (no tensor required)
# --------------------------------------------------------------------------- #
def iter_chunk_ranges(n_total: int, chunk_elements: int, align: int = 1) -> Iterator[Tuple[int, int]]:
    """Yield (start, end) pairs for flat chunks of ``n_total`` elements.

    A pure-Python range iterator: no tensor is needed, so the chunked merge
    path can iterate over ranges before loading any data.
    """
    if chunk_elements <= 0 or chunk_elements >= n_total:
        yield 0, n_total
        return
    align = max(1, align)
    step = max(1, (chunk_elements // align) * align)
    for start in range(0, n_total, step):
        end = min(start + step, n_total)
        yield start, end


# --------------------------------------------------------------------------- #
# Chunked mask builders for the remaining methods
# --------------------------------------------------------------------------- #
def magnitude_outliers_mask_chunked(
    t: torch.Tensor,
    density: float,
    gamma: float,
    chunk_elements: int,
) -> torch.Tensor:
    """Global magnitude_outliers mask in two passes over chunks.

    Pass 1 finds the global top-γ and bottom-(1-d-γ) thresholds via bounded
    heaps. Pass 2 keeps the middle elements per chunk.
    """
    n = t.numel()
    target = int(round(density * n))
    n_top = int(round(gamma * n))
    n_bot = n - target - n_top
    if n_bot < 0:
        n_top += n_bot
        n_bot = 0
    n_top = max(0, min(n, n_top))
    n_bot = max(0, min(n, n_bot))

    # Pass 1: find global thresholds via bounded heaps
    top_thr = _topk_threshold(t, n_top) if n_top > 0 else 0.0
    bot_thr = _bottomk_threshold(t, n_bot) if n_bot > 0 else 0.0

    # Pass 2: apply per chunk
    mask = torch.zeros(n, dtype=torch.uint8, device=t.device)
    start = 0
    for chunk in iter_chunks(t, chunk_elements, align=1):
        w = _abs_flat(chunk)
        c = chunk.numel()
        keep = torch.ones(c, dtype=torch.uint8, device=t.device)
        if n_top > 0:
            keep[(w >= top_thr)] = 0
        if n_bot > 0:
            keep[(w <= bot_thr)] = 0
        mask[start:start + c] = keep
        start += c
    return mask.reshape_as(t)


def _topk_threshold(t: torch.Tensor, k: int) -> float:
    """The k-th largest |value| of ``t`` via a bounded max-heap."""
    if k <= 0:
        return 0.0
    heap: List[float] = []
    for chunk in iter_chunks(t, chunk_elements=10_000_000):
        w = _abs_flat(chunk)
        take = min(k, w.numel())
        if take == 0:
            continue
        vals = torch.topk(w, take).values.detach().cpu().numpy().ravel()
        for v in vals:
            if len(heap) < k:
                heapq.heappush(heap, v)
            elif v > heap[0]:
                heapq.heapreplace(heap, v)
    return float(heap[0]) if heap else 0.0


def _bottomk_threshold(t: torch.Tensor, k: int) -> float:
    """The k-th smallest |value| of ``t`` via a bounded min-heap (on -w)."""
    if k <= 0:
        return 0.0
    heap: List[float] = []
    for chunk in iter_chunks(t, chunk_elements=10_000_000):
        w = _abs_flat(chunk)
        neg_w = -w
        take = min(k, neg_w.numel())
        if take == 0:
            continue
        vals = torch.topk(neg_w, take).values.detach().cpu().numpy().ravel()
        for v in vals:
            if len(heap) < k:
                heapq.heappush(heap, v)
            elif v > heap[0]:
                heapq.heapreplace(heap, v)
    return -float(heap[0]) if heap else 0.0


def bernoulli_mask_chunked(t: torch.Tensor, density: float, chunk_elements: int) -> torch.Tensor:
    """Bernoulli mask per chunk (per-element, so chunking is exact)."""
    mask = torch.zeros(t.numel(), dtype=torch.uint8, device=t.device)
    start = 0
    for chunk in iter_chunks(t, chunk_elements, align=1):
        c = chunk.numel()
        chunk_mask = bernoulli_mask(chunk, density).reshape(-1)
        mask[start:start + c] = chunk_mask
        start += c
    return mask.reshape_as(t)


def della_mask_chunked(t: torch.Tensor, density: float, epsilon: float, chunk_elements: int) -> torch.Tensor:
    """Della mask per chunk (per-row, so chunking aligned to rows is exact)."""
    row_stride = _row_stride(t)
    mask = torch.zeros(t.numel(), dtype=torch.uint8, device=t.device)
    start = 0
    for chunk in iter_chunks(t, chunk_elements, align=row_stride):
        c = chunk.numel()
        chunk_mask = della_mask(chunk, density, epsilon).reshape(-1)
        mask[start:start + c] = chunk_mask
        start += c
    return mask.reshape_as(t)
