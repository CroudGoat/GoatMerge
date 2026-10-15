# Metadata & inspect

Every task vector and merged model carries a `goatmerge.json` sidecar recording
the base model, the task vectors, the applied settings, and a fingerprint.

Module: `goatmerge/metadata.py` — `build_merged_metadata`, `write_metadata`,
`load_metadata`, `validate_metadata`. Inspection: `goatmerge/inspect.py` —
`inspect_dir`, `verify_against_base`.

- `goatmerge.json` records **what a directory was produced from**. It is not a
  replacement for a recipe — keep merge recipes in YAML
  (`merge -c recipe.yaml`, see [yaml-recipes.md](yaml-recipes.md)).
- `goatmerge inspect --dir MODEL` prints the metadata / manifest /
  fingerprint as JSON.
- A merged model directory stays **directly loadable by `transformers`**: the
  base model's `config.json` and tokenizer files are copied over automatically
  (`io.copy_support_files`), since GoatMerge only rewrites weights.

```
merged_model/
  model.safetensors           merged weights
  config.json                 copied from the base model (HF model config)
  tokenizer_config.json ...   copied when present in the base
  goatmerge.json              GoatMerge metadata
```

Covered by `tests/test_metadata.py` and the CLI e2e tests in
`tests/test_consensus_merge.py`.
