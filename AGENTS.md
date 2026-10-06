# AGENTS.md

## 会話言語

- 本プロジェクトでは**日本語**で会話する。
- コード内のコメント・docstring は英語でよいが、ユーザーへの説明・チャットは日本語。

## プロジェクト概要

GoatMerge は、ファインチューン済みモデル（タスクベクトル）をベースモデルに
マージする**ストリーミング型タスク算術マージエンジン**である。

### 設計目標

- **`torch.stack` 不使用** — 各 delta を 1 個ずつストリーミングし、全 delta の
  スタックを物化しない。
- **ピーク RAM ≈ 5–7 S**（S = 最大テンソルのバイト数）。実測では 1.06 S。
- **mergekit GTA（Generalized Task Arithmetic）との数値パリティ** —
  bf16 テンソルで rtol=2e-2、atol=1e-2 の範囲で一致。
- **HF シャード型 safetensors I/O** — 標準的な HuggingFace 配置をそのまま使う。

### 主要コンポーネント

| ファイル | 責務 |
|---|---|
| `consensus.py` | ストリーミング蓄積カーネル（`ConsensusAccumulator`）。`acc`/`l1`/`c` を in-place 更新。 |
| `merge.py` | `merge_model` / `merge_tensor`。テンソルごとのマージロジック、YAML/CLI 設定の適用。 |
| `extract.py` | `extract_task_vector`。ベースとソースの差分 TV を抽出。 |
| `sparsify.py` | スパルシファイカーネル（`l1`、`l2`、`gamma`、top-k）。`torch.topk` を使用。 |
| `consensus.py` 内 | コンセンサス恒等式 `(acc + M·l1)/2`。`M = sign(majority)`。 |
| `hf.py` | HF モデルディレクトリの解決（`resolve_model_dir`）。 |
| `io.py` | `ShardReader`、`TensorWriter`、`ShardedTensorIndex`。シャード型 safetensors の読み書き。 |
| `fingerprint.py` | ベースモデルの指紋計算と検証（`compute_base_fingerprint`、`verify_against_base`）。 |
| `metadata.py` | マージ結果のメタデータエンベローブ（`build_merged_metadata`、`write_metadata`）。 |
| `inspect.py` | モデル/TV ディレクトリの検査（`inspect_dir`）。 |
| `cli.py` | CLI エントリポイント。`extract` / `merge` / `inspect` サブコマンド。YAML レシピ（`-c`）対応。 |
| `__init__.py` | パッケージ初期化。 |

### テスト

| ファイル | 内容 |
|---|---|
| `tests/test_consensus_merge.py` | ストリーミングマージ + mergekit GTA との数値パリティ（rtol=2e-2, atol=1e-2）。 |
| `tests/test_sparsify.py` | スパルシファイ各方式（l1, l2, gamma, top-k）。 |
| `tests/test_fingerprint.py` | 指紋検証。 |
| `tests/test_io.py` | シャード型/単一シャード/サブ行列切り詰め。 |
| `tests/test_metadata.py` | メタデータエンベローブ。 |
| `tests/measure_peak_ram.py` | ピーク RAM 計測スクリプト。 |

### 数値パリティの要点

- 重み付き積 `δᵢ · αᵢ` は **bf16·bf16 テンソル積**（参照の `stacked · weights`
  と一致）。Python スカラー `add_(alpha=...)` では内部積精度が近接要素で
  乖離し、多数決符号が反転する。
- スパルシファイは**コンセンサス前**に各 delta へ適用。
- コンセンサス恒等式: `mixed = (acc + M·l1)/2`、`divisor[divisor==0]=1`。

### YAML レシピ

`goatmerge merge -c recipe.yaml` で YAML ファイルからマージパラメータを
読み込む。CLI フラグは YAML 値を上書きする。

```yaml
base: /path/to/base
out: /path/to/out
tv:
  - dir: /path/to/tv1
    weight: 0.7
consensus: sum
density: 1.0
method: null
n: 64
m: 256
```

### 環境

- ワークスペース: `/home/CloudGoat/llms_merge/GoatMerge`
- インタープリタ: `/home/CloudGoat/venvs/mergekit/bin/python`
- 依存: Python ≥ 3.10、PyTorch、`safetensors`、`pyyaml`

### Git

- リポジトリ: `CroudGoat/GoatMerge`（private）、master ブランチ。
- `ambition.md` は gitignore 対象（ambition の記録用）。
