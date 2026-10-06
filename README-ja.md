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

## テスト

```bash
python -m pytest tests/ -v
```

43 テストがカバー：
- ストリーミングマージ（コンセンサスなし、consensus sum/count）
- スパルシファイ（l1, l2, gamma, top-k）
- 指紋検証
- I/O（シャード型、単一シャード、サブ行列切り詰め）
- メタデータエンベローブ
- mergekit GTA との数値パリティ（rtol=2e-2, atol=1e-2）

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
  merge.py            # merge_model, merge_tensor
  metadata.py         # メタデータエンベローブ
  sparsify.py         # スパルシファイカーネル
tests/
  test_consensus_merge.py   # パリティ + マージテスト
  test_fingerprint.py
  test_io.py
  test_metadata.py
  test_sparsify.py
  measure_peak_ram.py       # ピーク RAM 計測
```
