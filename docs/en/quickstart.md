# Quickstart

## Install

```bash
pip install -e .
```

Requirements: Python ≥ 3.10, `torch>=2.1`, `safetensors>=0.4`, `pyyaml>=6.0`.
This installs the `goatmerge` command (`goatmerge.cli:main`).

Model references may be a local directory or a HuggingFace repo id already
present in the local HF cache (see `goatmerge/hf.py`).

## 1. Extract task vectors (recommended)

Compute the difference against the base model once and store it as a task vector.
Later merges never read the source models again.

```bash
goatmerge extract \
  --base   /path/to/base_model \
  --source /path/to/finetuned_model \
  --out    /path/to/tv_jp
```

Large tensors: add `--chunk-elements 33554432` for row-chunked extraction
(O(S) + O(chunk) instead of 2 S). The result is bit-identical to the standard
path. See [chunk-merge.md](chunk-merge.md).

## 2. Merge

```bash
goatmerge merge \
  --base /path/to/base_model \
  --tv /path/to/tv_jp:0.7 \
  --tv /path/to/tv_math:0.5 \
  --out /path/to/merged_model \
  --consensus sum
```

- Weights are `--tv DIR:WEIGHT` (same syntax for `--model`).
- Use `--model` to merge source models directly; the delta is computed as
  `source − base` on the fly.
- Keep recipes in YAML: `goatmerge merge -c recipe.yaml`. CLI flags override
  YAML values. See [yaml-recipes.md](yaml-recipes.md).

## 3. Inspect the result

```bash
goatmerge inspect --dir /path/to/merged_model
```

A merged model directory is **directly loadable by `transformers`**: the base
model's `config.json` and tokenizer files are copied over automatically, since
GoatMerge only rewrites weights.

```
merged_model/
  model.safetensors           merged weights
  config.json                 copied from the base model (HF model config)
  tokenizer_config.json ...   copied when present in the base
  goatmerge.json              GoatMerge metadata
```

> Keep merge *recipes* in YAML. `goatmerge.json` is not a replacement for a
> recipe — it records what a directory was produced from.
> See [metadata-inspect.md](metadata-inspect.md).

## Fingerprint gate

Every TV carries the fingerprint of the base it was extracted against. At merge
time the fingerprint is compared with the base you passed; a mismatch refuses
the merge (exit code 3). Bypass with `--skip-fingerprint-check` only if you
know what you are doing. See [fingerprint.md](fingerprint.md).
