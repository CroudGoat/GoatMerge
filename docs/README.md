# GoatMerge docs

Bilingual documentation for GoatMerge (plain Markdown, no site generator required).

- English: [`en/index.md`](en/index.md)
- 日本語: [`ja/index.md`](ja/index.md)

## Map

| Topic | en | ja |
|---|---|---|
| Overview / 概要 | [en/index.md](en/index.md) | [ja/index.md](ja/index.md) |
| Quickstart | [en/quickstart.md](en/quickstart.md) | [ja/quickstart.md](ja/quickstart.md) |
| CLI reference | [en/cli.md](en/cli.md) | [ja/cli.md](ja/cli.md) |
| YAML recipes | [en/yaml-recipes.md](en/yaml-recipes.md) | [ja/yaml-recipes.md](ja/yaml-recipes.md) |
| Merge methods | [en/merge-methods.md](en/merge-methods.md) | [ja/merge-methods.md](ja/merge-methods.md) |
| Sparsification | [en/sparsification.md](en/sparsification.md) | [ja/sparsification.md](ja/sparsification.md) |
| Streaming kernel & consensus | [en/streaming-kernel.md](en/streaming-kernel.md) | [ja/streaming-kernel.md](ja/streaming-kernel.md) |
| Chunk Merge | [en/chunk-merge.md](en/chunk-merge.md) | [ja/chunk-merge.md](ja/chunk-merge.md) |
| I/O (sharded safetensors) | [en/io.md](en/io.md) | [ja/io.md](ja/io.md) |
| Fingerprint | [en/fingerprint.md](en/fingerprint.md) | [ja/fingerprint.md](ja/fingerprint.md) |
| Metadata & inspect | [en/metadata-inspect.md](en/metadata-inspect.md) | [ja/metadata-inspect.md](ja/metadata-inspect.md) |
| Architecture & invariants | [en/architecture.md](en/architecture.md) | [ja/architecture.md](ja/architecture.md) |
| Memory model | [en/memory.md](en/memory.md) | [ja/memory.md](ja/memory.md) |
| Testing & benchmarks | [en/testing.md](en/testing.md) | [ja/testing.md](ja/testing.md) |
| Troubleshooting & FAQ | [en/troubleshooting-faq.md](en/troubleshooting-faq.md) | [ja/troubleshooting-faq.md](ja/troubleshooting-faq.md) |

Conventions:

- `S` = bytes of the largest single tensor.
- CLI flags override YAML recipe values (`merge -c recipe.yaml`).
- Exit codes: `0` success, `2` missing arguments, `3` base-fingerprint mismatch.
- Code comments and docstrings are in English; these docs mirror that terminology in both languages.
