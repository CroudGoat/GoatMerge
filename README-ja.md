# GoatMerge

![GoatMerge](GoatMerge.png)

[English](README.md)

**GoatMerge** は、ファインチューン済みモデル（タスクベクトル）をベースモデルに
マージする**ストリーミング型**のタスク算術エンジンです。`torch.stack` で全 delta を
積み上げる代わりに 1 つずつストリーミングし、ピークメモリを大幅に抑えながら
mergekit の GTA（Generalized Task Arithmetic）と数値パリティを保ちます。

- **`torch.stack` を使わない** — delta を 1 個ずつストリーミングして in-place 蓄積
- **ピーク RAM ≈ 5–7 S**（S = 最大テンソルのバイト数）。実測では **1.06 S**
- **mergekit GTA との数値パリティ** — bf16 テンソルで rtol = 2e-2, atol = 1e-2
- **Chunk Merge（チャンク分割マージ）** — 数百億〜数千億パラメータ級の大テンソルを O(S) + O(chunk) で処理。結果は通常経路と完全一致
- **HF シャード型 safetensors** — 標準的な HuggingFace 配置をそのまま読み書き
- **タスクベクトルの事前抽出と再利用** — 抽出後はソースモデルを再読込しない
- **ベース指紋検証** — 別のベースで作られた TV を誤って混ぜるのを防止

---

## なぜ GoatMerge なのか

mergekit の GTA は 1 テンソルを処理するだけで、全 delta を `torch.stack` で
`(k, shape)` に積み上げます。k = 4 の 400B 級モデル（S ≈ 1.7 GB）では
ピークが **34–53 GB** に達し、単一 GPU では実行不可能になります。

GoatMerge は delta を 1 個ずつストリームし、`acc` / `l1` / `c` という 3 つの
蓄積領域だけでコンセンサスを求めます。コンセンサスのマスク和も
厳密な恒等式 `(acc + M·l1)/2` に置き換えるため、全 delta のマスク積を
物資化しません。

| 観点 | GoatMerge | mergekit GTA |
|---|---|---|
| delta の扱い | 1 個ずつストリーミング | `torch.stack` で全件積み上げ |
| ピーク RAM（k 個の TV、1 テンソル） | ≈ 5–7 S | (4k+4) S – (6k+7) S |
| コンセンサス | 恒等式 `(acc + M·l1)/2` | weighted 全件のマスク積 |
| 大テンソル | Chunk Merge で O(S) + O(chunk) | 全件resident |
| I/O | HF シャード型 safetensors | HF シャード型 safetensors |

### ベンチマーク（300 MB bf16 テンソル × 3 TV、consensus=sum）

| エンジン | ピーク RSS | 増分メモリ | 数値パリティ |
|---|---|---|---|
| **GoatMerge** | 3394.6 MB | 3009.1 MB | max\|d\| = 0.0625 |
| **mergekit GTA** | 7289.4 MB | 6903.8 MB | 同上 |

GoatMerge は mergekit の **約 47%** のピーク RAM で、マージ処理そのものに
帰属する増分メモリでは **約 44%** です。計測方法と詳細は
[`Benchmarks/RESULTS.md`](Benchmarks/RESULTS.md) を参照してください。

---

## インストール

```bash
pip install -e .
```

- Python ≥ 3.10
- PyTorch、`safetensors`
- `pyyaml`（YAML レシピを使う場合）

インストールすると `goatmerge` コマンドが使えます。`goatmerge --help` で
サブコマンド一覧が表示されます。

---

## クイックスタート

### 1. タスクベクトルを抽出する（推奨）

ベースモデルと、ファインチューン済みモデルとの差分を 1 度だけ計算して
タスクベクトル（TV）として保存します。以降のマージではソースモデルを
読み込み直しません。

```bash
goatmerge extract \
  --base   /path/to/base_model \
  --source /path/to/finetuned_model \
  --out    /path/to/tv_jp
```

### 2. マージする

```bash
goatmerge merge \
  --base /path/to/base_model \
  --tv /path/to/tv_jp:0.7 \
  --tv /path/to/tv_math:0.5 \
  --out /path/to/merged_model \
  --consensus sum
```

重みは `--tv ディレクトリ:重み` のようにコロン付きで指定します（`--model` も
同じ書式）。ファインチューン済みモデルを直接マージに使う場合は `--model` を
使います。この場合 delta は `ソース − ベース` としてその場で計算されます。

### 3. 結果を確認する

```bash
goatmerge inspect --dir /path/to/merged_model
```

抽出した TV とマージ結果には `goatmerge.json` が付き、ベースモデル・使用した
TV・適用した設定・フィンガープリントが記録されます。マージ時には、TV を
作成したときのベースと、いま指定されたベースのフィンガープリントが自動で
照合されます。不一致の場合はエラー終了します
（`--skip-fingerprint-check` で回避可能）。

マージ結果のディレクトリは **そのまま `transformers` で読み込めます**。
ベースモデルの `config.json` とトークナイザ関連ファイルは自動的にコピー
されるためです（GoatMerge が書き換えるのは重みだけなので、これらの内容は
変わりません）。

```
merged_model/
  model.safetensors           マージ結果の重み
  config.json                 ベースモデルからコピー（HF のモデル設定）
  tokenizer_config.json など   ベースにあればコピー
  goatmerge.json              GoatMerge のメタデータ
```

> マージの「レシピ」（どの TV をどの重みで、どんな設定で混ぜるか）は YAML
> ファイルで管理してください（`goatmerge merge -c recipe.yaml`）。
> `goatmerge.json` はレシピの代わりではなく、「このディレクトリが何から
> 作られたか」の記録です。

モデルの指定にはローカルディレクトリ、またはローカル HF キャッシュに存在する
HF リポジトリ ID を使えます。

---

## CLI リファレンス

### `merge`

| フラグ | 既定値 | 説明 |
|---|---|---|
| `--base` | 必須 | ベースモデルのディレクトリ、または HF リポジトリ ID |
| `--tv DIR:WEIGHT` | — | タスクベクトルを指定（繰り返し指定可） |
| `--model DIR:WEIGHT` | — | ソースモデルを直接指定（繰り返し指定可） |
| `--out` | 必須 | 出力先ディレクトリ |
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
| `--chunk-elements` | なし | Chunk Merge: この要素数以上のテンソルを行チャンクに分割して処理 |
| `--skip-fingerprint-check` | false | ベース指紋の照合をスキップする |
| `-c, --config` | なし | YAML レシピファイル（後述） |

### `extract`

| フラグ | 既定値 | 説明 |
|---|---|---|
| `--base` | 必須 | ベースモデルのディレクトリ、または HF リポジトリ ID |
| `--source` | 必須 | ファインチューン済みモデルのディレクトリ、または HF リポジトリ ID |
| `--out` | 必須 | タスクベクトルの出力先 |

### `inspect`

| フラグ | 既定値 | 説明 |
|---|---|---|
| `--dir` | 必須 | 対象のタスクベクトル / マージ結果ディレクトリ |

---

## YAML レシピ

`-c recipe.yaml` を渡すと、複数のパラメータをファイルで管理できます。
CLI フラグは YAML の値を上書きするため、レシピを雛形にして 1 つだけ
コマンドラインで変更する、といった使い方もできます。

**最小構成**（必須フィールドのみ）:

```yaml
base: /path/to/base_model
out: /path/to/merged_model
tv:
  - dir: /path/to/tv_jp
    weight: 0.7
```

**全項目**:

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
```

---

## マージ方式

| `merge_method` | 動作 | 用途 |
|---|---|---|
| `gta` | mergekit GTA と同じタスク算術（consensus / sparsify 対応） | 標準。mergekit と数値パリティ |
| `linear` | `base + Σ wᵢ·δᵢ` | 単純な重み付き和 |
| `mixture` | `base + Σ wᵢ·δᵢ / Σ wᵢ` | 重み付き平均 |
| `slerp` | `base` と `base + Σ wᵢ·δᵢ` の球面線形補間 | 2 モデル間の補間 |
| `ties` | 符号が多数決と一致するソースで平均化（TIES 風） | 符号の一致で剪定したい場合 |

> `gta` のみ mergekit の同名メソッドと数値パリティを取ります。`linear` /
> `mixture` / `slerp` / `ties` は GoatMerge 独自の実装で、mergekit の同名
> メソッドとは定義が異なります（たとえば GoatMerge の `slerp` は `t = 1` で
> `base + Σ wᵢ·δᵢ` を返します）。

### スパース化の方式

`--method` で指定します。`--density 1.0` では無効になります。

| `method` | 動作 | 関連フラグ |
|---|---|---|
| `magnitude` | 絶対値が大きい上位 density の割合を保持 | — |
| `magnitude_outliers` | 上位 gamma と下位を落とし、中間を保持 | `--gamma` |
| `random` | 要素ごとに確率 density で保持（DARE 風） | — |
| `della_magprune` | 行内の絶対値ランクに応じて保持確率を変える | `--epsilon` |
| `bs` | m 要素のブロックごとに上位 n 個を保持 | `--n`, `--m` |

スパース化は**コンセンサスより前**、各 delta に対して適用します。マスクは
`magnitude` / `magnitude_outliers` ではテンソル全体の順位、それ以外では
要素単位・行単位・ブロック単位で決まります。乱数を使う方式
（`random` / `della_magprune`）は「テンソル名 × ソース × チャンク」から
決定論的にシードされるため、同じ入力からは常に同じ結果が得られます。

---

## 設計

### ストリーミングカーネル

1 テンソルあたり、各 delta を 1 パスでちょうど 1 回ずつストリームして
次のように蓄積します。

```
acc  = Σᵢ αᵢ · δᵢ         （base dtype, in place）
l1   = Σᵢ |αᵢ · δᵢ|      （base dtype, in place, consensus 時のみ）
c    = Σᵢ sign(αᵢ·δᵢ)    （int8, count 方式のみ）
```

重み付き積 `δᵢ · αᵢ` は **bf16・bf16 のテンソル積**として計算します
（mergekit の `stacked · weights` と同じ）。Python の float スカラーを
渡す `add_(alpha=...)` では内部積の精度が異なり、近接した要素で多数決の
符号が反転することがあります。

### コンセンサス恒等式

```
mixed = (acc + M · l1) / 2      M = sign(acc)  （sum 方式）
                               M = sign(c)    （count 方式）
```

多数派の符号と一致しない要素はこの式で自動的に 0 になるため、全 delta の
マスク和を 2 回の in-place 演算で表現できます。飽和までマスクを
物資化しないので、メモリは k に依存しません。

コンセンサス有効時の結果は次のようになります。

```
result = base + mixed / divisor        divisor = 符号一致 TV の重み和
```

`divisor` は 1 要素ずつ求める必要があるため、delta を第 2 パスで
ストリーミングし直して計算します。Chunk Merge では、この第 2 パスも
**チャンクの行範囲だけ**を読み直すため、フルテンソルがresidentになることは
ありません。

### Chunk Merge（チャンク分割マージ）

**Chunk Merge** は、チャンク分割によるマージ経路の名称です。

`--chunk-elements N`（または YAML の `chunk_elements`）を指定すると、
1 テンソルが N 要素以下の行チャンクに分割されて処理されます。
ディスクから読み込み → 蓄積 → 出力バッファへ書き込み、を繰り返すので、
ピーク RAM は **O(S) + O(chunk)** になります（S は出力テンソル自体で回避
不能）。

- 大域的なスパース化（`magnitude` / `magnitude_outliers`）は、テンソル全体を
  1 回スキャンして閾値とタイ数だけを求め（O(1) メモリ）、各チャンクに適用します
- コンセンサスの除数はチャンクの行範囲だけ再読込します
- 乱数ベースのスパース化はチャンクごとのシードで、パス間で常に同じマスクになります
- 結果は通常経路と**完全一致**します（同一カーネル・同一マスク）
- `slerp` は全域ノームに依存するため Chunk Merge に対応しておらず、
  警告とともに通常経路にフォールバックします

### I/O

HF シャード型 safetensors（`model.safetensors.index.json` +
`model-XXXXX-of-NNNNN.safetensors`、単一シャードなら
`model.safetensors`）をそのまま読み書きします。同時にresidentするのは
1 テンソルだけです。シャードのヘッダは読み込みごとにパースせず、
ShardReader 内でキャッシュします。

行範囲の読み出しには mmap を使わず、safetensors ヘッダの
`data_offsets` から直接バイトオフセットを計算して読みます。

### フィンガープリント

TV を抽出したベースモデルの指紋（テンソル名・形状・dtype・アンカーテンソルの
ハッシュ）を `goatmerge.json` に保存します。マージ時に現在のベースの指紋と
照合し、不一致ならエラー終了します。「別のベースで抽出した TV を
うっかり混ぜてしまう」事故を防げます。

---

## ピークメモリ

合成の 100 MB bf16 テンソル（2 TV、consensus=sum）での実測値です。

```
S（最大テンソル）: 100.0 MB
マージ時のピーク RSS: 105.6 MB
Peak / S:           1.06
```

最悪ケースの見積もり 5–7 S より小さくなるのは、パス間でテンソルを解放して
いるためです（`l1` は除数計算前に解放、各 `delta` は蓄積直後に解放）。
さらに Chunk Merge を使えば、1 テンソルあたりのピークは
O(S) + O(chunk) まで下がります。

規模の目安（1 層あたり）:

| モデル規模 | S（bf16, 1 層） | GoatMerge | mergekit GTA |
|---|---|---|---|
| 9B 級 | ≈ 1.8 GB | ≈ 2–4 GB | ≈ 22–34 GB |
| 400B 級 | ≈ 1.7 GB | ≈ 2 GB | ≈ 34–53 GB（単一 GPU では困難） |

---

## テスト

```bash
python -m pytest tests/ -v
```

74 個のテストが以下をカバーします。

- ストリーミングマージ（consensus なし / sum / count、負の重み、TV 欠損）
- mergekit GTA との数値パリティ（rtol=2e-2, atol=1e-2）
- スパース化の各方式、同値が多いテンソルでのチャンク一致
- カーネル単体（linear / mixture / slerp / ties）
- Chunk Merge 経路の一致（方式 × コンセンサス × 1〜3 次元テンソル）
- フィンガープリント照合
- I/O（シャード型 / 単一シャード / サブ行列切り詠め）
- メタデータエンベローブ

---

## ファイル構成

```
goatmerge/
  __init__.py      # パッケージ初期化
  cli.py           # CLI（extract / merge / inspect、YAML レシピ対応）
  consensus.py     # ConsensusAccumulator, GtaKernel（ストリーミングカーネル）
  extract.py       # タスクベクトル抽出（T = W_source − W_base）
  fingerprint.py   # ベースモデルの指紋計算と照合
  hf.py            # HF モデルディレクトリ / ローカルキャッシュの解決
  inspect.py       # TV・マージ結果の検査
  io.py            # ShardReader, TensorWriter, ShardedTensorIndex
  kernels.py       # linear / mixture / slerp / ties の各カーネル
  merge.py         # merge_model, merge_tensor（Chunk Merge 経路を含む）
  merge_method.py  # MergeMethod と build_kernel の分岐
   metadata.py      # goatmerge.json メタデータ
  sparsify.py      # スパース化カーネル、チャンク分割変種、大域マスク
tests/
  test_consensus_merge.py   # mergekit GTA パリティとマージロジック
  test_chunk_merge.py       # Chunk Merge 経路の一致
  test_kernels.py           # 各カーネルの単体テスト
  test_sparsify.py          # スパース化各方式
  test_fingerprint.py       # 指紋照合
  test_io.py                # I/O 層
  test_metadata.py          # メタデータ
  measure_peak_ram.py       # ピーク RAM 計測
bench_real_models.py        # 実モデルでのベンチマーク
Benchmarks/                 # GoatMerge vs mergekit GTA の比較結果
```
