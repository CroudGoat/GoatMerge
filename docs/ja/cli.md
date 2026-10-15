# CLI リファレンス

サブコマンドは `extract` / `merge` / `inspect`（`goatmerge/cli.py`）。

終了コードは `0` 成功 / `2` 引数不足 / `3` ベース指紋ミスマッチ。

## `merge`

| フラグ | 既定値 | 説明 |
|---|---|---|
| `--base` | 必須（または YAML の `base`） | ベースモデルのディレクトリ、または HF リポジトリ ID |
| `--tv DIR:WEIGHT` | — | タスクベクトルを指定（繰り返し指定可、YAML を上書き） |
| `--model DIR:WEIGHT` | — | ソースモデルを直接指定（繰り返し指定可、YAML を上書き） |
| `--out` | 必須（または YAML の `out`） | 出力先ディレクトリ |
| `--merge-method` | `gta` | `gta` \| `linear` \| `mixture` \| `slerp` \| `ties` |
| `--consensus` | `none` | `none` \| `sum` \| `count` |
| `--density` | `1.0` | スパース化率。1.0 で無効 |
| `--method` | なし | `magnitude` \| `random` \| `magnitude_outliers` \| `della_magprune` \| `bs` |
| `--n` | `64` | BS（n:m）のブロックあたり保持数 |
| `--m` | `256` | BS（n:m）のブロックサイズ |
| `--gamma` | `0.0` | `magnitude_outliers` で捨てる上位側の割合 |
| `--epsilon` | `0.0` | `della_magprune` の確率変動幅 |
| `--no-rescale` | false | スパース化後のノーム再正規化をオフにする |
| `--no-normalize` | false | 除数による正規化をオフにする |
| `--lambda` | `1.0` | ミックスされた delta にかける倍率 |
| `--chunk-elements` | なし | Chunk Merge：この要素数以上のテンソルを行チャンクに分割して処理 |
| `--skip-fingerprint-check` | false | ベース指紋の照合をスキップする |
| `--drop-caches` | false | 読み済み範囲をページキャッシュから外す（`POSIX_FADV_DONTNEED`） |
| `-c, --config` | なし | YAML レシピファイル（[yaml-recipes.md](yaml-recipes.md)） |

エントリ書式は `DIR:WEIGHT` で、**最後の**コロンで分割する（`cli._parse_entry`）。
`--tv` / `--model`（または YAML の `tv` / `model`）が最低 1 件は必要。

`--drop-caches`（merge）は opt-in のページキャッシュ対策。各テンソルは正確に 1 回
しか読まない（除数の再読は符号キャッシュで廃止済み）ので、再読ペナルティはない。
書き済みシャードも close 時にキャッシュから外す。`fdatasync` はしない（速度優先。
数 GB の残余 dirty ページはあり得る。`io._close_shard` のコメント参照）。
YAML キーは `drop_caches`。

## `extract`

| フラグ | 既定値 | 説明 |
|---|---|---|
| `--base` | 必須 | ベースモデルのディレクトリ、または HF リポジトリ ID |
| `--source` | 必須 | ファインチューン済みモデルのディレクトリ、または HF リポジトリ ID |
| `--out` | 必須 | タスクベクトルの出力先 |
| `--chunk-elements` | なし | この要素数以上を行チャンク抽出（O(S) + O(chunk)） |
| `--drop-caches` | false | 読み済み範囲をページキャッシュから外す |

減算は `load_delta(kind="model")` を使い回すインプレース方式（ピーク 2 S）。
チャンクパスは出力バッファを 1 回だけ確保してスライス代入する。
形状不一致は非チャンクパスにフォールバックする。結果はビット一致。

## `inspect`

| フラグ | 既定値 | 説明 |
|---|---|---|
| `--dir` | 必須 | 対象のタスクベクトル / マージ結果ディレクトリ |

TV やマージ結果のメタデータ・manifest・指紋を JSON で表示する
（[metadata-inspect.md](metadata-inspect.md)）。

## 例

```bash
# YAML を雛形に 1 つだけ CLI で上書き
goatmerge merge -c examples/merge_recipe.yaml --consensus sum

# ソースモデルを直接マージ（TV 事前抽出なし）
goatmerge merge --base B --model S1:0.5 --model S2:0.5 --out OUT

# 大テンソル向け
goatmerge merge --base B --tv TV1:0.7 --tv TV2:0.3 --out OUT --chunk-elements 33554432

# ページキャッシュを汚さない一括処理
goatmerge merge --base B --tv TV1:1.0 --out OUT --drop-caches
goatmerge extract --base B --source S --out TV --drop-caches
```
