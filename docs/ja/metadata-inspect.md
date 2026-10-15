# メタデータと inspect

抽出した TV とマージ結果は `goatmerge.json` を持つ。
同ファイルはベースモデル、使用した TV、適用した設定、指紋の記録である。

モジュールは `goatmerge/metadata.py` であり、公開要素は `build_merged_metadata`、`write_metadata`、`load_metadata`、`validate_metadata` である。
検査は `goatmerge/inspect.py` の `inspect_dir` と `verify_against_base` が担う。

- `goatmerge.json` はこのディレクトリが何から作られたかを示す由来記録であり、レシピではない。
マージのレシピ（どの TV をどの重みでどの設定で混ぜるか）は YAML で管理する（`goatmerge merge -c recipe.yaml`、[yaml-recipes.md](yaml-recipes.md)）。
- 利用者は `goatmerge inspect --dir MODEL` でメタデータ、manifest、指紋を JSON 表示できる。
- マージ結果のディレクトリは `transformers` で直接読み込める。
GoatMerge が書き換えるのは重みだけであり、ベースモデルの `config.json` とトークナイザ関連ファイルを自動的にコピーする（`io.copy_support_files`）からである。

```
merged_model/
  model.safetensors           マージ結果の重み
  config.json                 ベースモデルからコピー（HF のモデル設定）
  tokenizer_config.json など   ベースにあればコピー
  goatmerge.json              GoatMerge のメタデータ
```

`tests/test_metadata.py` と `tests/test_consensus_merge.py` の CLI e2e がこの層をカバーする。
