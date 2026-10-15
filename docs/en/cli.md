# CLI reference

Subcommands: `extract` / `merge` / `inspect` (`goatmerge/cli.py`).

Exit codes: `0` success, `2` missing arguments, `3` base-fingerprint mismatch.

## `merge`

| Flag | Default | Description |
|---|---|---|
| `--base` | required (or YAML `base`) | Base model directory, or a cached HF repo id |
| `--tv DIR:WEIGHT` | — | Task vector to merge (repeatable; overrides YAML) |
| `--model DIR:WEIGHT` | — | Source model to merge (repeatable; overrides YAML) |
| `--out` | required (or YAML `out`) | Output directory |
| `--merge-method` | `gta` | `gta` \| `linear` \| `mixture` \| `slerp` \| `ties` |
| `--consensus` | `none` | `none` \| `sum` \| `count` |
| `--density` | `1.0` | Sparsification density; `1.0` disables it |
| `--method` | (none) | `magnitude` \| `random` \| `magnitude_outliers` \| `della_magprune` \| `bs` |
| `--n` | `64` | Kept values per block for BS (n:m) |
| `--m` | `256` | BS block size |
| `--gamma` | `0.0` | Fraction dropped from the top by `magnitude_outliers` |
| `--epsilon` | `0.0` | Probability spread for `della_magprune` |
| `--no-rescale` | false | Disable norm re-normalization after sparsification |
| `--no-normalize` | false | Disable divisor normalization |
| `--lambda` | `1.0` | Scale applied to the mixed delta |
| `--chunk-elements` | (none) | Chunk Merge: process tensors above this many elements in row chunks |
| `--skip-fingerprint-check` | false | Skip the base fingerprint comparison |
| `--drop-caches` | false | Release each file range from the page cache after reading (`POSIX_FADV_DONTNEED`) |
| `-c, --config` | (none) | YAML recipe file (see [yaml-recipes.md](yaml-recipes.md)) |

Entry syntax is `DIR:WEIGHT`, split at the **last** colon (`cli._parse_entry`).
At least one `--tv` / `--model` (or YAML `tv` / `model`) is required.

`--drop-caches` (merge): opt-in page-cache hygiene. Each tensor is read exactly
once (the divisor is built from the sign cache, no re-read), so there is no
re-read penalty in the current design. Written shards are also dropped from
cache on close without `fdatasync` (speed-first; a few GB of residual dirty
pages are possible — see `io._close_shard`). YAML key: `drop_caches`.

## `extract`

| Flag | Default | Description |
|---|---|---|
| `--base` | required | Base model directory, or a cached HF repo id |
| `--source` | required | Fine-tuned model directory, or a cached HF repo id |
| `--out` | required | Where the task vector is written |
| `--chunk-elements` | (none) | Row-chunked extraction above this many elements (O(S) + O(chunk)) |
| `--drop-caches` | false | Release each file range from the page cache after reading |

Subtraction reuses `load_delta(kind="model")` in place (peak 2 S); the chunked
path pre-allocates one output buffer and fills it slice by slice. Shape
mismatches fall back to the non-chunked path. Results are bit-identical.

## `inspect`

| Flag | Default | Description |
|---|---|---|
| `--dir` | required | Task-vector or merged-model directory |

Prints JSON metadata / manifest / fingerprint of a TV or merged model.
See [metadata-inspect.md](metadata-inspect.md).

## Examples

```bash
# YAML baseline with one CLI tweak
goatmerge merge -c examples/merge_recipe.yaml --consensus sum

# Direct source models (no pre-extracted TVs)
goatmerge merge --base B --model S1:0.5 --model S2:0.5 --out OUT

# Very large tensors
goatmerge merge --base B --tv TV1:0.7 --tv TV2:0.3 --out OUT --chunk-elements 33554432

# Low page-cache footprint (bulk jobs)
goatmerge merge --base B --tv TV1:1.0 --out OUT --drop-caches
goatmerge extract --base B --source S --out TV --drop-caches
```
