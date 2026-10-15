# Troubleshooting & FAQ

## Errors

### `error: --base (or 'base' in config) is required` (exit 2)

`base` resolves CLI → YAML. Either pass `--base` or set `base:` in the recipe.
Same for `--out` / `out:`.

### `error: no task vectors or models given` (exit 2)

Pass at least one `--tv` / `--model`, or define `tv:` / `model:` in YAML.
Note: when CLI `--tv` is given it takes priority over the YAML list (they are
not merged).

### `error: base fingerprint mismatch for: ...` (exit 3)

A TV was extracted against a different base. Check you passed the right
`--base`, re-extract the TV against this base, or — only if you really mean
it — pass `--skip-fingerprint-check` (`skip_fingerprint_check: true`).
See [fingerprint.md](fingerprint.md).

### `bad entry '...' (expected DIR:WEIGHT)`

Entries split at the **last** colon. A bare `--tv /path/to/tv` without
`:weight` is invalid; write `--tv /path/to/tv:0.7`. Paths containing colons
still work because only the last colon separates the weight.

### Writer errors (off-plan arrival / size mismatch)

`TensorWriter` pre-commits sharding from the manifest and raises on anything
off-plan instead of silently corrupting. This usually means the manifest
(names/dtypes/shapes) does not match the tensors being fed — do not catch and
ignore it; fix the input set.

## Behavior questions

### Results differ slightly from mergekit — is that expected?

Small bf16 differences within rtol=2e-2, atol=1e-2 are the parity budget
(max|d| = 0.0625 observed). Larger or sign-flipping diffs are not — check that
the bf16·bf16 weighted product was not "simplified" to a Python-scalar
`add_(alpha=...)` (invariant #1).

### Chunk Merge result differs from the standard path — bug?

Yes, that would be a bug: they must be identical (except `slerp`, which
intentionally falls back with a warning). Report with tensor shape, method,
consensus, sparsify settings, and `chunk_elements`. See
[chunk-merge.md](chunk-merge.md).

### Stochastic sparsify gives different results run to run?

It should not for the same inputs: masks are seeded from
`(tensor name, source index, chunk)`. If you renamed tensors or reordered
sources, the seed — and hence the mask — legitimately changes.

### Out of memory on a huge tensor?

Set `--chunk-elements` (e.g. `33554432`) so tensors above N elements take the
O(S) + O(chunk) path. `slerp` still falls back to the full path. See
[memory.md](memory.md).

### System memory balloons though process RSS looks fine?

That is the page cache, not a leak. Use `--drop-caches` (`drop_caches: true`
for merge) for bulk jobs. See [io.md](io.md).

## Contributing checklist

1. Match existing enum naming and `from __future__ import annotations` style.
2. Add public APIs to `__init__.py` `__all__`.
3. Write docstrings in English (why / caveats, not trivial restatements).
4. Run the full suite: 97 passed is the baseline.
5. Re-check mergekit parity with the `test_consensus_merge.py` patterns.
6. Re-read [architecture.md](architecture.md) invariants — most merge bugs are
   invariant #1/#2/#3 violations.
