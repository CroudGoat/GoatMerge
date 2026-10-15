# YAML レシピ

`-c recipe.yaml` を渡すと、複数のパラメータをファイルで管理できる。
CLI フラグは YAML の値を上書きする。
このためレシピを雛形にし、1 項目だけコマンドラインで変える使い方もできる。

解決順序は CLI、YAML、既定値の順である。
`base` と `out` は CLI から YAML の順に解決し、どちらも必須である。
`tv` と `model` は CLI 指定があれば CLI を、なければ YAML を使う。
各エントリは文字列 `DIR` か `{dir, weight}` であり、weight の既定値は 1.0 である。

## 最小構成

```yaml
base: /path/to/base_model
out: /path/to/merged_model
tv:
  - dir: /path/to/tv_jp
    weight: 0.7
```

## 全項目

```yaml
# --- 必須 ---
base: /path/to/base_model          # ベースモデル（HF シャード型 safetensors）
out: /path/to/merged_model         # 出力先

# --- マージ対象（tv か model のどちらか 1 つ以上） ---
tv:
  - dir: /path/to/tv_jp            # タスクベクトル
    weight: 0.7                    # マージ重み
  - dir: /path/to/tv_math
    weight: 0.3
# model:
#   - dir: /path/to/source_model   # ソースモデル（delta をその場で計算）
#     weight: 0.5

# --- マージ方式 ---
merge_method: gta                  # gta | linear | mixture | slerp | ties

# --- コンセンサス ---
consensus: sum                     # none | sum | count

# --- スパース化 ---
density: 1.0                       # 1.0 で無効
method: null                       # null | magnitude | random | magnitude_outliers | della_magprune | bs
n: 64                              # --n
m: 256                             # --m
gamma: 0.0                         # --gamma
epsilon: 0.0                       # --epsilon
no_rescale: false                  # true にするとノーム再正規化を無効にする
no_normalize: false                # true にすると除数正規化を無効にする
lambda: 1.0                        # --lambda

# --- 大テンソル ---
chunk_elements: null               # --chunk-elements

# --- 指紋照合 ---
skip_fingerprint_check: false      # true にすると照合をスキップする

# --- ページキャッシュ対策（merge のみ） ---
drop_caches: false                 # true = 読み書き後に POSIX_FADV_DONTNEED を発行する
```

同梱の例は `examples/merge_recipe.yaml` である。
その他の YAML キーは CLI オプション名と 1 対 1 に対応する（`lambda`、`no_rescale`、`no_normalize`、`chunk_elements`、`skip_fingerprint_check`、`drop_caches`）。
`extract` に YAML モードはない。
上記のキーはすべて `merge` 用である。
