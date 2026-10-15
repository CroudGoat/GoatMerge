# YAML recipes

Pass `-c recipe.yaml` to keep a merge configuration in a file. CLI flags
override the YAML values, so a recipe can serve as a baseline with one value
tweaked on the command line.

Resolution order: **CLI → YAML → default**. `base` / `out` resolve CLI → YAML
(required). `tv` / `model` entries come from CLI when given, else from YAML.
Each entry is a string `DIR` or `{dir, weight}` (weight defaults to 1.0).

## Minimal recipe

```yaml
base: /path/to/base_model
out: /path/to/merged_model
tv:
  - dir: /path/to/tv_jp
    weight: 0.7
```

## Full recipe

```yaml
# --- Required ---
base: /path/to/base_model          # base model (HF-sharded safetensors)
out: /path/to/merged_model         # output directory

# --- Merge inputs (at least one of tv / model) ---
tv:
  - dir: /path/to/tv_jp            # task vector
    weight: 0.7                    # merge weight
  - dir: /path/to/tv_math
    weight: 0.3
# model:
#   - dir: /path/to/source_model   # source model (delta computed on the fly)
#     weight: 0.5

# --- Merge method ---
merge_method: gta                  # gta | linear | mixture | slerp | ties

# --- Consensus ---
consensus: sum                     # none | sum | count

# --- Sparsification ---
density: 1.0                       # 1.0 keeps everything
method: null                       # null | magnitude | random | magnitude_outliers | della_magprune | bs
n: 64                              # --n
m: 256                             # --m
gamma: 0.0                         # --gamma
epsilon: 0.0                       # --epsilon
no_rescale: false                  # true disables norm re-normalization
no_normalize: false                # true disables divisor normalization
lambda: 1.0                        # --lambda

# --- Very large tensors ---
chunk_elements: null               # --chunk-elements

# --- Fingerprint check ---
skip_fingerprint_check: false      # true skips the comparison

# --- Page-cache hygiene (merge only) ---
drop_caches: false                 # true = POSIX_FADV_DONTNEED after each range
```

The shipped example is `examples/merge_recipe.yaml`. Other YAML keys map 1:1 to
the CLI option names (`lambda`, `no_rescale`, `no_normalize`,
`chunk_elements`, `skip_fingerprint_check`, `drop_caches`).

Note: `extract` has no YAML mode; all other keys above apply to `merge`.
