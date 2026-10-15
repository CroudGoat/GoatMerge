# クイックスタート

## インストール

```bash
pip install -e .
```

`pip install -e .` で導入する。
要件は Python ≥ 3.10、`torch>=2.1`、`safetensors>=0.4`、`pyyaml>=6.0` である。
導入すると `goatmerge` コマンドが使えるようになる。
利用者は `goatmerge --help` でサブコマンド一覧を確認できる。
モデル指定にはローカルディレクトリを使うか、ローカル HF キャッシュに存在する HF リポジトリ ID を使う（実装は `goatmerge/hf.py`）。

## タスクベクトルの抽出

ベースモデルとファインチューン済みモデルの差分を一度だけ計算し、タスクベクトル（TV）として保存する。
以降のマージではソースモデルを読み込まない。

```bash
goatmerge extract \
  --base   /path/to/base_model \
  --source /path/to/finetuned_model \
  --out    /path/to/tv_jp
```

テンソルが大きい場合は行チャンク抽出を使う。
`--chunk-elements 33554432` を付けると、ピークメモリは 2 S ではなく O(S) + O(chunk) になる。
出力は標準パスの出力とビット単位で一致する。
詳細は [chunk-merge.md](chunk-merge.md) を参照する。

## マージの実行

```bash
goatmerge merge \
  --base /path/to/base_model \
  --tv /path/to/tv_jp:0.7 \
  --tv /path/to/tv_math:0.5 \
  --out /path/to/merged_model \
  --consensus sum
```

- 重みは `--tv ディレクトリ:重み` の形式で指定する（`--model` も同じ形式）。
- ファインチューン済みモデルを直接使う場合は `--model` を指定する。
この場合 GoatMerge は delta を `ソース − ベース` としてその場で計算する。
- レシピは YAML で管理する。
`goatmerge merge -c recipe.yaml` と実行する。
CLI フラグは YAML の値を上書きする（[yaml-recipes.md](yaml-recipes.md)）。

## 結果の確認

```bash
goatmerge inspect --dir /path/to/merged_model
```

マージ結果のディレクトリは `transformers` で直接読み込める。
GoatMerge が書き換えるのは重みだけであり、ベースモデルの `config.json` とトークナイザ関連ファイルを自動的にコピーするからである。

```
merged_model/
  model.safetensors           マージ結果の重み
  config.json                 ベースモデルからコピー（HF のモデル設定）
  tokenizer_config.json など   ベースにあればコピー
  goatmerge.json              GoatMerge のメタデータ
```

マージのレシピは YAML で管理する。
`goatmerge.json` はレシピではなく、このディレクトリが何から作られたかを示す由来記録である（[metadata-inspect.md](metadata-inspect.md)）。

## ベース指紋の照合

抽出した TV は、抽出時のベースモデルの指紋を持つ。
マージ時に GoatMerge は現在のベースの指紋と照合し、不一致ならエラー終了する（終了コード 3）。
照合を外すのは意図的な場合に限り、`--skip-fingerprint-check` を付ける（[fingerprint.md](fingerprint.md)）。
