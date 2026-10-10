"""Sparsification kernels for task vectors (uint8 masks, topk-based).

Improvements over mergekit's ``sparsify.py``:

* ``magnitude`` / ``magnitude_outliers`` use ``torch.topk`` (k indices only,
  O(N log k), k*4B of indices) instead of a full ``argsort`` (N*4B indices,
  O(N log N)).
* All masks are ``uint8`` (1 byte/element) instead of bf16/f32.
* Mask application and rescaling are in-place.
* A Balanced-Sparsification (BS, n:m block) kernel is included (from CABS).
* Chunked variants for large tensors: global magnitude pruning is done in two
  passes with a bounded top-k candidate buffer (k values, torch memory) so the
  per-chunk RAM is O(chunk) and no per-element Python loop is needed.

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

import zlib
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

# Flat chunk used when scanning a tensor for a global top-k/bottom-k value.
# Bounded so the scan never materializes more than a chunk of |values| at once.
_SCAN_CHUNK_ELEMENTS = 10_000_000


def merge_generator(key: str, index: int, chunk: int = 0) -> torch.Generator:
    """Deterministic CPU generator for a (tensor, source, chunk) triple.

    Seeded from ``(key, index, chunk)`` so that a stochastic mask
    (``random`` / ``della_magprune``) is *identical* for the same source and
    chunk in every pass of a merge. The consensus divisor is computed in a
    second streamed pass that re-reads and re-sparsifies each delta; without a
    shared seed the two masks differ and the divisor no longer matches the
    masked numerator.
    """
    g = torch.Generator(device="cpu")
    seed = (zlib.crc32(key.encode("utf-8")) * 1_000_003 + index * 7919 + chunk) % (2**63 - 1)
    g.manual_seed(seed)
    return g


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
    # |values| are widened to float32 before summing so the rescale factor is
    # independent of the accumulation order — which is what lets the chunked
    # merge path reproduce the whole-tensor (non-chunked) result exactly.
    w = _abs_flat(t)
    if norm == RescaleNorm.l1:
        return float(w.sum())
    if norm == RescaleNorm.l2:
        return float(w.norm())
    if norm == RescaleNorm.linf:
        return float(w.max())
    raise NotImplementedError(norm)


def _row_stride(t: torch.Tensor) -> int:
    """Number of elements in one row (product of trailing dims); 1 for 1-D."""
    if t.dim() >= 2:
        return int(torch.tensor(t.shape[1:]).prod())
    return 1


# --------------------------------------------------------------------------- #
# Mask builders (return a uint8 mask shaped like the input)
# --------------------------------------------------------------------------- #
def magnitude_mask(
    t: torch.Tensor,
    density: float,
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    """0/1 mask keeping the ``density`` fraction of largest |values|.

    ``k = int(density * n)`` (truncation) to match mergekit's ``magnitude``.
    Ties at the k-th value are broken by flat index order (deterministic,
    and identical to what :func:`magnitude_mask_chunked` / the chunked merge
    path produce — ``torch.topk`` tie order is implementation-defined, so it
    is deliberately not used as the tie rule).
    """
    return magnitude_mask_chunked(t, density, max(1, t.numel()))


def magnitude_outliers_mask(
    t: torch.Tensor,
    density: float,
    gamma: float,
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    """0/1 mask: drop the top ``gamma`` (largest) and the bottom
    ``(1 - density - gamma)`` (smallest); keep the middle.

    ``k`` uses truncation (``int``) to match mergekit's
    ``magnitude_outliers``. Ties are broken by flat index order.
    """
    return magnitude_outliers_mask_chunked(t, density, gamma, max(1, t.numel()))


def bernoulli_mask(
    t: torch.Tensor,
    density: float,
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    """0/1 mask: each element kept with probability ``density``."""
    work_dtype = (
        t.dtype
        if t.device.type != "cpu" or t.dtype == torch.bfloat16
        else torch.float32
    )
    mask = torch.bernoulli(
        torch.full_like(input=t, fill_value=density, dtype=work_dtype),
        generator=generator,
    )
    return (mask > 0).to(torch.uint8).reshape_as(t)


def della_mask(
    t: torch.Tensor,
    density: float,
    epsilon: float,
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
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
    span = max_ranks - min_ranks
    # A single-element row has span 0 -> 0/0. Define rank_norm = 0 for it.
    span = torch.where(span == 0, torch.ones_like(span), span)
    rank_norm = ((ranks - min_ranks) / span).clamp(0, 1)
    probs = (density - epsilon) + rank_norm * 2 * epsilon
    mask = torch.bernoulli(probs, generator=generator).to(work_dtype)
    return (mask > 0).to(torch.uint8).reshape_as(t)


def bs_mask(t: torch.Tensor, n: int, m: int) -> torch.Tensor:
    """Balanced Sparsification: keep top-``n`` of every ``m``-block.

    Flattened row-major, split into consecutive blocks of ``m``. Trailing
    elements beyond the last full block are zeroed (matches the official CABS
    reference). Delegates to :func:`bs_mask_chunked` so the chunked merge path
    uses exactly the same block grid.
    """
    if n <= 0 or m <= 0 or n > m:
        raise ValueError(f"invalid n:m configuration n={n}, m={m}")
    return bs_mask_chunked(t, n, m, max(1, t.numel()))


def make_mask(
    t: torch.Tensor,
    density: float,
    method: SparsificationMethod,
    n: int = 64,
    m: int = 256,
    gamma: float = 0.0,
    epsilon: float = 0.0,
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    """Dispatch to the right mask builder."""
    if method == SparsificationMethod.magnitude:
        return magnitude_mask(t, density, generator=generator)
    if method == SparsificationMethod.magnitude_outliers:
        return magnitude_outliers_mask(t, density, gamma, generator=generator)
    if method == SparsificationMethod.random:
        return bernoulli_mask(t, density, generator=generator)
    if method == SparsificationMethod.della_magprune:
        return della_mask(t, density, epsilon, generator=generator)
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
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    """Apply the mask in-place and rescale to match the original norm.

    Returns the (modified) ``t``. If ``density >= 1`` the tensor is unchanged.

    ``chunk_elements`` bounds the global magnitude/BS mask to a chunked
    two-pass computation, keeping per-chunk RAM O(chunk).

    ``generator`` seeds the stochastic masks (``random`` / ``della_magprune``)
    so the same mask is produced for the same (tensor, source) pair in every
    pass of a merge (see :func:`merge_generator`).
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
            mask = bernoulli_mask_chunked(t, density, chunk_elements, generator=generator)
        elif method == SparsificationMethod.della_magprune:
            mask = della_mask_chunked(t, density, epsilon, chunk_elements, generator=generator)
        else:
            mask = make_mask(t, density, method, n=n, m=m, gamma=gamma, epsilon=epsilon, generator=generator)
    else:
        mask = make_mask(t, density, method, n=n, m=m, gamma=gamma, epsilon=epsilon, generator=generator)

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
    generator: Optional[torch.Generator] = None,
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
        generator=generator,
    )
    return t


# --------------------------------------------------------------------------- #
# Chunked global magnitude (two-pass, bounded top-k candidate buffer)
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


def _iter_abs_chunks(source) -> Iterator[torch.Tensor]:
    """Yield flattened |values| chunks from a tensor or an iterable of chunks.

    ``source`` may be a tensor (scanned in bounded chunks) or any iterable of
    row/chunk tensors (e.g. chunks streamed from disk), which is what lets the
    global mask parameters be computed without materializing the tensor.
    """
    if isinstance(source, torch.Tensor):
        for chunk in iter_chunks(source, chunk_elements=_SCAN_CHUNK_ELEMENTS):
            yield _abs_flat(chunk)
    else:
        for chunk in source:
            yield _abs_flat(chunk)


def _topk_buffer_from(source, k: int, negate: bool = False) -> Optional[torch.Tensor]:
    """Top-``k`` |values| of a chunk source (of ``-w`` when ``negate``).

    A single candidate buffer of at most ``k`` elements is maintained while
    scanning, so the scan costs one pass with O(k) scratch (torch values, not
    Python objects) and no per-element Python loop.
    """
    buf: Optional[torch.Tensor] = None
    for w in _iter_abs_chunks(source):
        if negate:
            w = -w
        take = min(k, w.numel())
        if take <= 0:
            continue
        vals = torch.topk(w, take).values
        buf = vals if buf is None else torch.cat([buf, vals])
        if buf.numel() > k:
            buf = torch.topk(buf, k).values
    return buf


def _threshold_from(source, k: int) -> float:
    """The k-th largest |value| of a chunk source (global)."""
    if k <= 0:
        return 0.0
    buf = _topk_buffer_from(source, k)
    if buf is None or buf.numel() == 0:
        return 0.0
    return float(buf.min())


def _bottom_threshold_from(source, k: int) -> float:
    """The k-th smallest |value| of a chunk source (global)."""
    if k <= 0:
        return 0.0
    buf = _topk_buffer_from(source, k, negate=True)
    if buf is None or buf.numel() == 0:
        return 0.0
    return -float(buf.min())


def _count_from(source, op: str, thr: float) -> int:
    """Global count of ``|x| > thr`` (``op=">"``) or ``|x| < thr`` (``"<"``)."""
    total = 0
    for w in _iter_abs_chunks(source):
        if op == ">":
            total += int((w > thr).sum())
        else:
            total += int((w < thr).sum())
    return total


def magnitude_threshold(t: torch.Tensor, k: int) -> float:
    """The k-th largest |value| of ``t`` (global), streamed in bounded chunks."""
    return _threshold_from(t, k)


def _bottomk_threshold(t: torch.Tensor, k: int) -> float:
    """The k-th smallest |value| of ``t`` (global), streamed in bounded chunks."""
    return _bottom_threshold_from(t, k)


def _count_above(t: torch.Tensor, thr: float) -> int:
    """Global count of ``|x| > thr``, streamed in bounded chunks."""
    return _count_from(t, ">", thr)


def _count_below(t: torch.Tensor, thr: float) -> int:
    """Global count of ``|x| < thr``, streamed in bounded chunks."""
    return _count_from(t, "<", thr)


def magnitude_mask_chunked(
    t: torch.Tensor,
    density: float,
    chunk_elements: int,
) -> torch.Tensor:
    """Global magnitude mask computed in two passes over chunks.

    Pass 1 finds the global k-th largest |value| (threshold). Pass 2 keeps
    every element with ``|x| > threshold`` plus, in flat index order, enough
    of the ``|x| == threshold`` elements to keep exactly ``k`` globally. The
    strict/tie split is decided from *global* counts so the result keeps
    exactly ``k`` elements regardless of how ties fall across chunks.
    Deterministic.
    """
    n = t.numel()
    k = int(density * n)
    k = max(0, min(n, k))
    if k == 0:
        return torch.zeros(n, dtype=torch.uint8, device=t.device).reshape_as(t)
    thr = magnitude_threshold(t, k)
    tie_budget = k - _count_above(t, thr)
    mask = torch.zeros(n, dtype=torch.uint8, device=t.device)
    start = 0
    for chunk in iter_chunks(t, chunk_elements, align=1):
        w = _abs_flat(chunk)
        c = chunk.numel()
        above = (w > thr).to(torch.uint8)
        eq = (w == thr).to(torch.uint8)
        take_eq = max(0, min(int(eq.sum()), tie_budget))
        if take_eq > 0:
            eq_idx = torch.nonzero(eq).view(-1)
            above[eq_idx[:take_eq]] = 1
            tie_budget -= take_eq
        mask[start:start + c] = above
        start += c
    return mask.reshape_as(t)


def bs_mask_chunked(t: torch.Tensor, n: int, m: int, chunk_elements: int) -> torch.Tensor:
    """BS mask computed per chunk (chunk boundaries aligned to ``m``).

    Each chunk's ``|values|`` is computed once (from the chunk itself), so the
    cost is O(N) overall rather than O(N^2 / chunk).
    """
    flat = t.reshape(-1)
    total = flat.numel()
    n_full = (total // m) * m
    mask = torch.zeros(total, dtype=torch.uint8, device=t.device)
    start = 0
    for chunk in iter_chunks(t, chunk_elements, align=m):
        c = chunk.numel()
        # only the part of this chunk that lies inside full m-blocks is masked
        full_c = min(c, max(0, n_full - start))
        if full_c > 0:
            blocks = _abs_flat(chunk)[:full_c].view(-1, m)
            topk = blocks.topk(n, dim=1).indices
            block_mask = torch.zeros_like(blocks, dtype=torch.uint8).scatter_(1, topk, 1)
            mask[start:start + full_c] = block_mask.reshape(-1)
        start += c
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

    Pass 1 finds the global top-γ and bottom-(1-d-γ) thresholds and the
    global counts strictly beyond them. Pass 2 drops everything beyond them
    plus enough ties (flat index order) to drop exactly ``n_top`` / ``n_bot``,
    so exactly ``target`` elements survive — including tie-heavy /
    discrete-valued tensors, where a naive per-chunk threshold comparison
    drops (or keeps) far too many.
    """
    n = t.numel()
    target = int(density * n)
    n_top = int(gamma * n)
    n_bot = n - target - n_top
    if n_bot < 0:
        n_top += n_bot
        n_bot = 0
    n_top = max(0, min(n, n_top))
    n_bot = max(0, min(n, n_bot))

    top_thr = magnitude_threshold(t, n_top) if n_top > 0 else 0.0
    bot_thr = _bottomk_threshold(t, n_bot) if n_bot > 0 else 0.0
    # global strict counts first: the remaining (tie) budgets are then fixed
    top_ties = n_top - _count_above(t, top_thr) if n_top > 0 else 0
    bot_ties = n_bot - _count_below(t, bot_thr) if n_bot > 0 else 0

    mask = torch.zeros(n, dtype=torch.uint8, device=t.device)
    start = 0
    for chunk in iter_chunks(t, chunk_elements, align=1):
        w = _abs_flat(chunk)
        c = chunk.numel()
        drop = torch.zeros(c, dtype=torch.uint8, device=t.device)

        if n_top > 0:
            drop[(w > top_thr)] = 1
            eq = w == top_thr
            take = min(int(eq.sum()), max(0, top_ties))
            if take > 0:
                idx = torch.nonzero(eq).view(-1)
                drop[idx[:take]] = 1
                top_ties -= take

        if n_bot > 0:
            drop[(w < bot_thr)] = 1
            eq = (w == bot_thr) & (drop == 0)
            take = min(int(eq.sum()), max(0, bot_ties))
            if take > 0:
                idx = torch.nonzero(eq).view(-1)
                drop[idx[:take]] = 1
                bot_ties -= take

        mask[start:start + c] = 1 - drop
        start += c
    return mask.reshape_as(t)


def bernoulli_mask_chunked(
    t: torch.Tensor,
    density: float,
    chunk_elements: int,
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    """Bernoulli mask per chunk (per-element, so chunking is exact)."""
    mask = torch.zeros(t.numel(), dtype=torch.uint8, device=t.device)
    start = 0
    for chunk in iter_chunks(t, chunk_elements, align=1):
        c = chunk.numel()
        chunk_mask = bernoulli_mask(chunk, density, generator=generator).reshape(-1)
        mask[start:start + c] = chunk_mask
        start += c
    return mask.reshape_as(t)


def della_mask_chunked(
    t: torch.Tensor,
    density: float,
    epsilon: float,
    chunk_elements: int,
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    """Della mask per chunk (per-row, so chunking aligned to rows is exact).

    Chunks are at least one row wide so a row is never split.
    """
    row_stride = _row_stride(t)
    step = max(1, row_stride if chunk_elements <= row_stride else (chunk_elements // row_stride) * row_stride)
    mask = torch.zeros(t.numel(), dtype=torch.uint8, device=t.device)
    start = 0
    for chunk in iter_chunks(t, step, align=row_stride):
        c = chunk.numel()
        chunk_mask = della_mask(chunk, density, epsilon, generator=generator).reshape(-1)
        mask[start:start + c] = chunk_mask
        start += c
    return mask.reshape_as(t)


# --------------------------------------------------------------------------- #
# Streaming global mask (used by the chunked merge path)
#
# The chunked merge path streams one row range at a time, so a global mask
# cannot be materialized. These helpers compute the mask *parameters*
# (thresholds + tie budgets, and the rescale factor) with O(1) RAM by scanning
# the delta straight from disk, then apply them chunk by chunk in flat index
# order — which reproduces the non-chunked (whole-tensor) mask exactly.
# --------------------------------------------------------------------------- #
def _magnitude_params(source_factory, n_total: int, density: float, rescale: bool) -> dict:
    k = int(density * n_total)
    k = max(0, min(n_total, k))
    if k == 0:
        return {"kind": "empty"}
    thr = _threshold_from(source_factory(), k)
    tie_budget = k - _count_from(source_factory(), ">", thr)
    params = {"kind": "magnitude", "thr": thr, "ties": tie_budget}
    if rescale:
        before = after = 0.0
        for w in _iter_abs_chunks(source_factory()):
            before += float(w.sum())
            keep = w > thr
            eq = w == thr
            take = min(int(eq.sum()), max(0, tie_budget))
            if take > 0:
                idx = torch.nonzero(eq).view(-1)
                keep[idx[:take]] = True
                tie_budget -= take
            after += float(w[keep].sum())
        params["scale"] = _rescale_scale(before, after)
    return params


def _outliers_params(source_factory, n_total: int, density: float, gamma: float, rescale: bool) -> dict:
    target = int(density * n_total)
    n_top = int(gamma * n_total)
    n_bot = n_total - target - n_top
    if n_bot < 0:
        n_top += n_bot
        n_bot = 0
    n_top = max(0, min(n_total, n_top))
    n_bot = max(0, min(n_total, n_bot))
    top_thr = _threshold_from(source_factory(), n_top) if n_top > 0 else 0.0
    bot_thr = _bottom_threshold_from(source_factory(), n_bot) if n_bot > 0 else 0.0
    params = {
        "kind": "outliers",
        "has_top": n_top > 0,
        "has_bot": n_bot > 0,
        "top_thr": top_thr,
        "bot_thr": bot_thr,
        "top_ties": n_top - _count_from(source_factory(), ">", top_thr) if n_top > 0 else 0,
        "bot_ties": n_bot - _count_from(source_factory(), "<", bot_thr) if n_bot > 0 else 0,
    }
    if rescale:
        # dry-run the tie fill on a copy of the budgets to get the masked norm
        state = {"top_ties": params["top_ties"], "bot_ties": params["bot_ties"]}
        before = after = 0.0
        for w in _iter_abs_chunks(source_factory()):
            before += float(w.sum())
            after += float(w[_outlier_keep(w, params, state)].sum())
        params["scale"] = _rescale_scale(before, after)
    return params


def _rescale_scale(before: float, after: float) -> float:
    if before < EPS or after < EPS:
        return 1.0
    return before / after


def _outlier_keep(w: torch.Tensor, params: dict, state: dict) -> torch.Tensor:
    """Keep-mask of one chunk, consuming the running tie budgets in ``state``."""
    drop = torch.zeros(w.numel(), dtype=torch.bool, device=w.device)
    if params["has_top"]:
        drop[(w > params["top_thr"])] = True
        eq = w == params["top_thr"]
        ties = state.get("top_ties", params["top_ties"])
        take = min(int(eq.sum()), max(0, ties))
        if take > 0:
            idx = torch.nonzero(eq).view(-1)
            drop[idx[:take]] = True
            state["top_ties"] = ties - take
    if params["has_bot"]:
        drop[(w < params["bot_thr"])] = True
        eq = (w == params["bot_thr"]) & (~drop)
        ties = state.get("bot_ties", params["bot_ties"])
        take = min(int(eq.sum()), max(0, ties))
        if take > 0:
            idx = torch.nonzero(eq).view(-1)
            drop[idx[:take]] = True
            state["bot_ties"] = ties - take
    return ~drop


def global_mask_params(
    method,
    n_total: int,
    density: float,
    gamma: float,
    rescale: bool,
    chunk_source,
) -> dict:
    """Parameters of the *global* mask for a tensor streamed from ``chunk_source``.

    ``chunk_source`` is a callable returning a fresh iterable of that tensor's
    chunks. Returns ``{"kind": "local"}`` for methods whose mask is inherently
    per-chunk (``random``, ``della_magprune``, ``bs``), in which case the
    caller falls back to the per-chunk mask builder.
    """
    if density >= 1 or method is None:
        return {"kind": "local"}
    if method == SparsificationMethod.magnitude:
        return _magnitude_params(chunk_source, n_total, density, rescale)
    if method == SparsificationMethod.magnitude_outliers:
        return _outliers_params(chunk_source, n_total, density, gamma, rescale)
    return {"kind": "local"}


def apply_global_mask(chunk: torch.Tensor, params: dict, state: dict) -> torch.Tensor:
    """Apply pre-computed global mask parameters to one chunk, in place.

    ``state`` carries the running tie budgets across chunks (a dict per
    tensor/source). Returns ``chunk``.
    """
    kind = params.get("kind")
    if kind == "empty":
        return chunk.zero_()
    if kind == "local":
        return chunk
    w = _abs_flat(chunk)
    if kind == "magnitude":
        keep = w > params["thr"]
        eq = w == params["thr"]
        ties = state.get("ties", params["ties"])
        take = min(int(eq.sum()), max(0, ties))
        if take > 0:
            idx = torch.nonzero(eq).view(-1)
            keep[idx[:take]] = True
            state["ties"] = ties - take
    elif kind == "outliers":
        keep = _outlier_keep(w, params, state)
    else:
        return chunk
    chunk.mul_(keep.to(chunk.dtype))
    if "scale" in params:
        chunk.mul_(params["scale"])
    return chunk


class GlobalMasker:
    """Applies one source's *global* mask to every chunk, in every pass.

    The chunked merge path needs the mask of the whole tensor, but streams
    chunks; :func:`global_mask_params` computes the mask parameters once and
    this object re-applies them to each chunk. The tie budgets are reset by
    :meth:`begin_pass` so the consensus second pass reproduces pass 1's mask
    exactly.
    """

    def __init__(self, params_list: List[dict]) -> None:
        self.params_list = params_list
        self.states: List[dict] = [{} for _ in params_list]

    def begin_pass(self, chunk: int = 0) -> None:
        """Start a new pass. Called with each chunk index; the tie budgets
        are reset only at chunk 0, so they run continuously across the chunks
        of one pass (the accumulation pass and the divisor pass each restart).
        """
        if chunk == 0:
            self.states = [{} for _ in self.params_list]

    def apply(self, index: int, chunk: torch.Tensor) -> bool:
        """Apply the global mask in place; ``False`` if this source is local."""
        params = self.params_list[index]
        if params.get("kind") == "local":
            return False
        apply_global_mask(chunk.reshape(-1), params, self.states[index])
        return True
