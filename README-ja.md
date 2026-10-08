# GoatMerge

ストリーミング型タスク算術マージエンジン。ファインチューン済みモデル
（タスクベクトル）をベースモデルにマージします。

- **`torch.stack` 不使用** — 各 delta を1つずつストリーミング
- **ピーク RAM ≈ 5–7 S**（S = 最大テンソルバイト数）
- **mergekit GTA（Generalized Task Arithmetic）との数値パリティ**

## 主な特性

| 項目 | GoatMerge | mergekit GTA |
|---|---|---|
| 全 delta をスタック？ | しない — 1つずつストリーミング | する（`torch.stack`） |
| ピーク RAM（k 個 TV、1 テンソル） | ≈ 5–7 S | (4k+4) S – (6k+7) S |
| コンセンサス（マスク和） | 厳密恒等式 `(acc + M·l1)/2` | `stacked · weights` 後にマスク |
| スパルシファイ | コンセンサス前に各 delta へ | コンセンサス前 |
| I/O | HF シャード型 safetensors | HF シャード型 safetensors |

## mergekit との比較

### メモリ（実測、100 MB bf16 テンソル、2 TV、consensus=sum）

| | GoatMerge | mergekit GTA（理論値） |
|---|---|---|
| ピーク RSS | **105.6 MB**（1.06 × S） | **1.2 – 1.9 GB**（12 S – 19 S） |
| スタック済みテンソル | 物化しない | `k × S`（k=2 で 200 MB） |
| 各 delta の一時テンソル | 1 個ずつ（100 MB） | k 個全部 resident（200 MB） |
| 蓄積簿 | acc + l1 + c = 2.5 S | stacked + weighted + mask ≈ 3 S |

9B 級モデル（層あたり S ≈ 1.8 GB）では、GoatMerge は層あたり
≈ 2–4 GB、mergekit は ≈ 22–34 GB。

### 速度

GoatMerge は各 delta をパスにつき1回だけストリームし、in place で
蓄積します。`torch.stack` の割当、`stacked · weights` の全テンソル積、
マスクのための全 delta 第 2 パスがありません。第 2 パスは要素毎
`divisor`（コンセンサスのみ）で、各 delta の `add_` 1 回だけです。

| 操作 | GoatMerge | mergekit GTA |
|---|---|---|
| k 個 delta の読み込み | k × (1 load) | k × (1 load) + 1 stack |
| 重み付き和 | k × (in-place `add_`) | 1 × (full `stacked · weights`) |
| マスク（コンセンサス） | in-place 恒等式 | 1 × (full mask multiply) |
| 除数 | k × (in-place `add_`) | 1 × (full `weights · mask`) |

in-place 蓄積は `torch.stack` + `stacked · weights` に必要な O(k·S)
の一時割当を回避し、コンセンサス恒等式 `(acc + M·l1)/2` は全テンソル
マスク積を 2 回の in-place `add_`/`mul_` に置き換えます。

### 数値パリティ

GoatMerge は bf16 テンソルで mergekit GTA と **rtol = 2e-2、atol = 1e-2**
の範囲で一致します。重み付き積 `δᵢ · αᵢ` は bf16·bf16 テンソル積
（参照の `stacked · weights` と一致）であり、Python フロート標量
`add_` ではありません — 後者の内部積精度は近接要素で乖離します。

## インストール

```bash
pip install -e .
```

Python ≥ 3.10、PyTorch、`safetensors` が必要です。

## CLI

```bash
# 2 つの TV ディレクトリをベースモデルにマージ
goatmerge merge \
  --base /path/to/base_model \
  --tv /path/to/tv1 --weight 0.5 \
  --tv /path/to/tv2 --weight 0.7 \
  --out /path/to/output \
  --consensus sum \
  --normalize

# モデル対からタスクベクトルを抽出
goatmerge extract \
  --base /path/to/base \
  --model /path/to/finetuned \
  --out /path/to/tv_dir
```

### YAML レシピ

コマンドラインにすべてのフラグを打つ代わりに、YAML ファイルを `-c` で
渡します：

```bash
goatmerge merge -c recipe.yaml
```

**最小レシピ**（必須フィールドのみ）：

```yaml
base: /path/to/base_model
out: /path/to/merged_output
tv:
  - dir: /path/to/tv1
    weight: 0.7
```

これだけで動きます。それ以外はすべて任意で、省略したフィールドは既定値に
フォールバックします。

**完全レシピ**（全フィールド、コメント付き）：

```yaml
# --- 必須 ---
base: /path/to/base_model        # ベースモデルディレクトリ（HF シャード型 safetensors）
out: /path/to/merged_output      # マージ結果の出力先

# --- タスクベクトル（1 以上） ---
tv:
  - dir: /path/to/tv1            # タスクベクトルディレクトリ
    weight: 0.7                  # この TV のマージ重み
  - dir: /path/to/tv2
    weight: 0.3

# --- ソースモデル（tv の代替；tv/model のいずれか 1 以上） ---
# model:
#   - dir: /path/to/source_model
#     weight: 0.5

# --- コンセンサス（マスク和） ---
consensus: sum                   # none | sum | count  （既定: none）

# --- スパルシファイ ---
density: 1.0                     # 0 = スキップ、1.0 = 全保持（既定: 1.0）
method: null                     # null = スパルシファイなし; l1 | l2 | gamma | topk
n: 64                            # top-k 件数（既定: 64）
m: 256                           # ブロックサイズ（既定: 256）
gamma: 0.0                       # ガンマ閾値（既定: 0.0）
epsilon: 0.0                     # エプシロン下限（既定: 0.0）
rescale: true                    # スパルシファイ後のノーム再計算（既定: true）

# --- 正規化・倍率 ---
normalize: true                  # 要素ごと除数で割る（既定: true）
lambda: 1.0                      # ミックステンソルの倍率（既定: 1.0）

# --- チャンク分割モード（大テンソル用） ---
chunk_elements: null             # テンソルをこの要素数でチャンク分割

# --- 指紋 ---
skip_fingerprint_check: false    # ベース指紋検証をスキップ
```

CLI フラグは対応する YAML 値を上書きするため、レシピをベースにして
コマンドラインで 1 つだけ変える、という使い方もできます。

### オプション

| フラグ | デフォルト | 説明 |
|---|---|---|
| `--base` | （必須） | ベースモデルディレクトリ（HF シャード型 safetensors） |
| `--tv` / `--model` | （1 以上） | タスクベクトルディレクトリ、またはソースモデルディレクトリ |
| `--weight` | 1.0 | 各エントリのマージ重み |
| `--out` | （必須） | 出力ディレクトリ |
| `--consensus` | `none` | `none` \| `sum` \| `count` |
| `--normalize` | true | 要素ごとの除数で割る |
| `--lambda` | 1.0 | ミックステンソルの倍率 |
| `--density` | 1.0 | スパルシファイ密度（0 = スキップ） |
| `--method` | （なし） | スパルシファイ方式: `l1` \| `l2` \| `gamma` |
| `--n` | 64 | スパルシファイの top-k 件数 |
| `--m` | 256 | スパルシファイのブロックサイズ |
| `--gamma` | 0.0 | ガンマ閾値 |
| `--epsilon` | 0.0 | エプシロン下限 |
| `--rescale` | true | スパルシファイ後の正規化再計算 |
| `--chunk-elements` | null | テンソルをこのサイズのチャンクに分割（チャンク分割モード） |

## 設計

### ストリーミングカーネル

テンソルごとに、各 delta を1パスにつきちょうど1回ストリームします：

```
acc  = Σᵢ αᵢ · δᵢ        （in place, base dtype）
l1   = Σᵢ |αᵢ · δᵢ|     （in place, base dtype）
c    = Σᵢ sign(αᵢ·δᵢ)   （int8, count 方式のみ）
```

重み付き積 `δᵢ · αᵢ` は **bf16·bf16 テンソル積**（参照の `stacked · weights`
と一致）であり、Python のスカラー `add_` ではありません — 後者の内部積
精度は近接要素で乖離し、要素ごとの多数決符号を反転させます。

### コンセンサス恒等式

```
mixed = (acc + M · l1) / 2,   M = (acc|c) ≥ 0 なら +1、さもなくば −1
```

ゼロは両形式で 0 に寄与し、参照のマスク（符号 0 要素を除外）と一致します。
`divisor`（符号一致 TV の要素ごと重み和）のみ第 2 ストリーミングパスを
必要とします。

### スパルシファイ

**コンセンサス前に**、各 delta へ適用。`torch.topk`（`argsort` ではなく）
を使用し、CPU での topk 用に bf16/fp16 を f32 へ幅広げします。

### I/O

HF シャード型 safetensors 配置（`model.safetensors.index.json` +
`model-XXXXX-of-NNNNN.safetensors`）。単一シャード → `model.safetensors`。
同時に1つのテンソルのみ常駐；`get_tensor` はシャードのキャッシュ
コピーと共有ストレージのビューを返します — 変更前に `.clone()` が必要です。

## ピーク RAM

合成 100 MB bf16 テンソル（2 TV、consensus=sum）での実測：

```
S (最大テンソル):  100.0 MB
マージピーク RSS:    105.6 MB
Peak / S:          1.06  （目標: 5–7）
```

ストリーミングカーネルは 5–7 S の最悪ケース予算よりメモリ効率が高く、
パス間でテンソルが解放されるため（`l1` は `divisor` 前に、各 `delta` は
蓄積後）です。

### チャンク分割モード（大テンソル）

`chunk_elements` 以上のテンソルでは、フラットなチャンク（
`chunk_elements` 要素）に分割して処理します。各チャンクをディスクから
読み込み、独立して蓄積し、事前割当の出力バッファにスライス代入で
書き込みます。ピーク RAM は O(6–7 S) ではなく **O(S) + O(chunk)**：

- `base` は遅延スライス（O(S) 割当なし）
- チャンクごと: `base_chunk`、`delta_chunk`、`acc`/`l1`/`c` は O(chunk)
- `out_flat` は O(S)（結果そのもの — 避けれない）

YAML レシピの `chunk_elements`、または CLI の `--chunk-elements` で
設定します。Slerp はチャンク分割非対応（全域ノーム依存）のため、
警告とともに非チャンク経路にフォールバックします。

## テスト

```bash
python -m pytest tests/ -v
```

61 テストがカバー：
- ストリーミングマージ（コンセンサスなし、consensus sum/count）
- スパルシファイ（l1, l2, gamma, top-k）
- 指紋検証
- I/O（シャード型、単一シャード、サブ行列切り詰め）
- メタデータエンベローブ
- mergekit GTA との数値パリティ（rtol=2e-2, atol=1e-2）
- チャンク分割マージ（パリティ + スパルシファイ有効性）

## ファイル構成

```
goatmerge/
  __init__.py      # パッケージ
  cli.py            # CLI エントリ
  consensus.py      # ConsensusAccumulator（ストリーミングカーネル）
  extract.py        # モデル対からの TV 抽出
  fingerprint.py    # 指紋検証
  hf.py             # HF モデルディレクトリヘルパ
  inspect.py        # モデル検査
  io.py              # ShardReader, TensorWriter, ShardedTensorIndex
  kernels.py         # 方式別マージカーネル
  merge.py            # merge_model, merge_tensor（チャンク分割含む）
  merge_method.py    # MergeMethod enum + build_kernel 分岐
  metadata.py         # メタデータエンベローブ
  sparsify.py         # スパルシファイカーネル + チャンク分割変種
tests/
  test_consensus_merge.py   # パリティ + マージテスト
  test_chunked_merge.py     # チャンク分割マージ（パリティ + 有効性）
  test_fingerprint.py
  test_io.py
  test_kernels.py           # カーネル単体テスト
  test_metadata.py
  test_sparsify.py
  measure_peak_ram.py       # ピーク RAM 計測
```
