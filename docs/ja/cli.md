# CLI リファレンス

サブコマンドは `extract`、`merge`、`inspect` の 3 個である（実装は `goatmerge/cli.py`）。
終了コードは `0` が成功、`2` が引数不足、`3` がベース指紋ミスマッチを示す。

## `merge`

| フラグ | 既定値 | 説明 |
|---|---|---|
| `--base` | 必須（または YAML の `base`） | ベースモデルのディレクトリ、または HF リポジトリ ID |
| `--tv DIR:WEIGHT` | — | タスクベクトルを指定する（繰り返し指定できる。YAML を上書きする） |
| `--model DIR:WEIGHT` | — | ソースモデルを直接指定する（繰り返し指定できる。YAML を上書きする） |
| `--out` | 必須（または YAML の `out`） | 出力先ディレクトリ |
| `--merge-method` | `gta` | `gta` \| `linear` \| `mixture` \| `slerp` \| `ties` |
| `--consensus` | `none` | `none` \| `sum` \| `count` |
| `--density` | `1.0` | スパース化率。1.0 で無効になる |
| `--method` | なし | `magnitude` \| `random` \| `magnitude_outliers` \| `della_magprune` \| `bs` |
| `--n` | `64` | BS（n:m）のブロックあたり保持数 |
| `--m` | `256` | BS（n:m）のブロックサイズ |
| `--gamma` | `0.0` | `magnitude_outliers` が上位から落とす割合 |
| `--epsilon` | `0.0` | `della_magprune` の確率変動幅 |
| `--no-rescale` | false | スパース化後のノーム再正規化を無効にする |
| `--no-normalize` | false | 除数による正規化を無効にする |
| `--lambda` | `1.0` | 混合済みの delta にかける倍率 |
| `--chunk-elements` | なし | N 要素を超えるテンソルを行チャンクで処理する（Chunk Merge） |
| `--skip-fingerprint-check` | false | ベース指紋の照合をスキップする |
| `--drop-caches` | false | 読み済み範囲をページキャッシュから外す（`POSIX_FADV_DONTNEED`） |
| `-c, --config` | なし | YAML レシピファイル。詳細は [yaml-recipes.md](yaml-recipes.md) を参照する |

エントリ書式は `DIR:WEIGHT` であり、`cli._parse_entry` が最後のコロンで分割する。
`--tv` と `--model`（または YAML の `tv` と `model`）を合わせて最低 1 件指定する。

`--drop-caches`（merge 用）はページキャッシュ対策の opt-in である。
各テンソルを正確に 1 回だけ読む（除数の再読は符号キャッシュで廃止した）ため、再読ペナルティはない。
書き済みシャードも close 時にキャッシュから外す。
`fdatasync` しないため、writeback 中の残余 dirty ページが数 GB 生じる場合がある。
これは速度優先の判断である（`io._close_shard` の注記を参照する）。
対応する YAML キーは `drop_caches` である。

## `extract`

| フラグ | 既定値 | 説明 |
|---|---|---|
| `--base` | 必須 | ベースモデルのディレクトリ、または HF リポジトリ ID |
| `--source` | 必須 | ファインチューン済みモデルのディレクトリ、または HF リポジトリ ID |
| `--out` | 必須 | タスクベクトルの出力先 |
| `--chunk-elements` | なし | この要素数を超えるテンソルを行チャンクで抽出する（O(S) + O(chunk)） |
| `--drop-caches` | false | 読み済み範囲をページキャッシュから外す |

減算は `load_delta(kind="model")` を使い回す in-place 方式であり、ピークは 2 S である。
チャンクパスは出力バッファを 1 回だけ確保し、スライス代入で埋める。
形状不一致のテンソルは非チャンクパスにフォールバックする。
出力は標準パスの出力とビット単位で一致する。

## `inspect`

| フラグ | 既定値 | 説明 |
|---|---|---|
| `--dir` | 必須 | 対象のタスクベクトルまたはマージ結果ディレクトリ |

TV とマージ結果のメタデータ、manifest、指紋を JSON で表示する（[metadata-inspect.md](metadata-inspect.md)）。

## 例

```bash
# YAML を雛形に 1 項目だけ CLI で上書きする
goatmerge merge -c examples/merge_recipe.yaml --consensus sum

# ソースモデルを直接マージする（TV 事前抽出なし）
goatmerge merge --base B --model S1:0.5 --model S2:0.5 --out OUT

# 大テンソル向け
goatmerge merge --base B --tv TV1:0.7 --tv TV2:0.3 --out OUT --chunk-elements 33554432

# ページキャッシュを汚さない一括処理
goatmerge merge --base B --tv TV1:1.0 --out OUT --drop-caches
goatmerge extract --base B --source S --out TV --drop-caches
```
