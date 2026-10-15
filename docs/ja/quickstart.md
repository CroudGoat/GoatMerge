# クイックスタート

## インストール

```bash
pip install -e .
```

- Python ≥ 3.10
- PyTorch、`safetensors`
- `pyyaml`（YAML レシピを使う場合）

インストールすると `goatmerge` コマンドが使える。`goatmerge --help` で
サブコマンド一覧が表示される。モデルの指定にはローカルディレクトリ、または
ローカル HF キャッシュに存在する HF リポジトリ ID が使える（`goatmerge/hf.py`）。

## 1. タスクベクトルを抽出する（推奨）

ベースモデルとファインチューン済みモデルとの差分を 1 度だけ計算して
タスクベクトル（TV）として保存する。以降のマージではソースモデルを
読み込み直さない。

```bash
goatmerge extract \
  --base   /path/to/base_model \
  --source /path/to/finetuned_model \
  --out    /path/to/tv_jp
```

大テンソル向けに行チャンク抽出もある：`--chunk-elements 33554432` を付けると
O(S) + O(chunk) になる（通常は 2 S）。結果は標準パスとビット一致。
詳しくは [chunk-merge.md](chunk-merge.md)。

## 2. マージする

```bash
goatmerge merge \
  --base /path/to/base_model \
  --tv /path/to/tv_jp:0.7 \
  --tv /path/to/tv_math:0.5 \
  --out /path/to/merged_model \
  --consensus sum
```

- 重みは `--tv ディレクトリ:重み` のようにコロン付きで指定（`--model` も同じ書式）。
- ファインチューン済みモデルを直接マージに使う場合は `--model` を使う。この場合
  delta は `ソース − ベース` としてその場で計算される。
- レシピは YAML で管理する：`goatmerge merge -c recipe.yaml`。CLI フラグは YAML を
  上書きする（[yaml-recipes.md](yaml-recipes.md)）。

## 3. 結果を確認する

```bash
goatmerge inspect --dir /path/to/merged_model
```

マージ結果のディレクトリは **そのまま `transformers` で読み込める**。
ベースモデルの `config.json` とトークナイザ関連ファイルは自動的にコピー
されるからだ（GoatMerge が書き換えるのは重みだけ）。

```
merged_model/
  model.safetensors           マージ結果の重み
  config.json                 ベースモデルからコピー（HF のモデル設定）
  tokenizer_config.json など   ベースにあればコピー
  goatmerge.json              GoatMerge のメタデータ
```

> マージの「レシピ」は YAML で管理すること。`goatmerge.json` はレシピの代わりじゃなく
> 「このディレクトリが何から作られたか」の記録だ（[metadata-inspect.md](metadata-inspect.md)）。

## 指紋ゲート

抽出した TV には、抽出時のベースモデルの指紋が付く。マージ時には今指定されたベースの
指紋と自動で照合され、不一致ならエラー終了する（終了コード 3）。
分かってて外すときだけ `--skip-fingerprint-check` を使うこと
（[fingerprint.md](fingerprint.md)）。
