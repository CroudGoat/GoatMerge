# YAML レシピ

`-c recipe.yaml` を渡すと、複数のパラメータをファイルで管理できる。
CLI フラグは YAML の値を上書きするので、レシピを雛形にして 1 つだけ
コマンドラインで変更する、といった使い方もできる。

解決順序は **CLI → YAML → 既定値**。`base` / `out` は CLI → YAML の順に解決（必須）。
`tv` / `model` は CLI 指定があれば CLI、なければ YAML。各エントリは文字列 `DIR`
または `{dir, weight}`（weight 既定 1.0）。

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
no_rescale: false                  # true にするとノーム再正規化をオフ
no_normalize: false                # true にすると除数正規化をオフ
lambda: 1.0                        # --lambda

# --- 大テンソル ---
chunk_elements: null               # --chunk-elements

# --- 指紋照合 ---
skip_fingerprint_check: false      # true にすると照合をスキップ

# --- ページキャッシュ対策（merge のみ） ---
drop_caches: false                 # true = 読み書き後に POSIX_FADV_DONTNEED
```

同梱の例は `examples/merge_recipe.yaml`。その他の YAML キーは CLI オプション名と
1:1 に対応する（`lambda`、`no_rescale`、`no_normalize`、`chunk_elements`、
`skip_fingerprint_check`、`drop_caches`）。

注意：`extract` に YAML モードはない。上記キーはすべて `merge` 用だ。
