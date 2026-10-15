# メタデータと inspect

抽出した TV とマージ結果には `goatmerge.json` が付く。ベースモデル・使用した
TV・適用した設定・フィンガープリントの記録だ。

モジュールは `goatmerge/metadata.py`：`build_merged_metadata` /
`write_metadata` / `load_metadata` / `validate_metadata`。検査は
`goatmerge/inspect.py` の `inspect_dir` / `verify_against_base`。

- `goatmerge.json` は「このディレクトリが何から作られたか」の記録だ。
  レシピの代わりじゃない。マージのレシピ（どの TV をどの重みで、どんな設定で混ぜるか）は
  YAML で管理すること（`goatmerge merge -c recipe.yaml`、
  [yaml-recipes.md](yaml-recipes.md) 参照）。
- `goatmerge inspect --dir MODEL` でメタデータ / manifest / 指紋を JSON 表示できる。
- マージ結果のディレクトリは **そのまま `transformers` で読み込める**。
  ベースモデルの `config.json` とトークナイザ関連ファイルは自動的にコピーされる
  （`io.copy_support_files`）。GoatMerge が書き換えるのは重みだけなので、
  これらの内容は変わらない。

```
merged_model/
  model.safetensors           マージ結果の重み
  config.json                 ベースモデルからコピー（HF のモデル設定）
  tokenizer_config.json など   ベースにあればコピー
  goatmerge.json              GoatMerge のメタデータ
```

`tests/test_metadata.py` と `tests/test_consensus_merge.py` の CLI e2e が
カバーしてる。
