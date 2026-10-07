# GoatMerge — 課題・実装状況・将来展望

## 1. 元の問題（mergekit GTA）

mergekit (v0.1.4) の Task Arithmetic (GTA) マージパスの問題点：

- `torch.stack` で全 delta を (k, shape) に積み上げる → ピーク ≈ (4k+4)S ～ (6k+7)S
- 全 `argsort` の O(N log N) + N·4B インデックスアロケーション
- loaded tensor の滞留（Executor `values` 辞書）
- dtype 変換のコピー
- 9B bf16（S≈110MB, k=4）でピーク ≈ 2.2–4 GB
- 400B 級（S≈1.7GB, k=4）で ≈ 34–53 GB → 単一 GPU で不可行

参照実装:
- カーネル: `mergekit/merge_methods/generalized_task_arithmetic.py`
- スパース化: `mergekit/sparsify.py`
- ストリーミング実証例: `Smart-Task-Arithmetic/taskvector/merge.py`

## 2. GoatMerge の実装（完了）

### 2.1 ストリーミング累加（`torch.stack` 廃止）

```python
acc = base.clone()
for i in range(k):
    delta = load(tv_i).clone()
    delta = sparsify(delta)          # コンセンサス前
    w = torch.tensor(alpha_i, dtype=delta.dtype)
    delta.mul_(w)                    # bf16·bf16 テンソル積（parity 重要）
    acc.add_(delta)                  # in-place
    l1.add_(delta.abs())
    # delta は即解放
```

- ピーク: **1.06 S**（実測、100 MB bf16、2 TV、consensus=sum）
- 9B 級（S≈1.8GB/層）で層あたり ≈ 2–4 GB

### 2.2 コンセンサス恒等式

```
mixed = (acc + M · l1) / 2,   M = sign(majority) = ±1
divisor[divisor == 0] = 1
result = (base + mixed).to(base.dtype)
```

- 全 delta のマスク積を 2 回の in-place `add_`/`mul_` に置き換え
- int8 符号カウンタ（count 方式）

### 2.3 数値パリティ

- 重み付き積 `δᵢ · αᵢ` は **bf16·bf16 テンソル積**（`stacked · weights` と一致）
- `add_(alpha=scalar)` では内部積精度が近接要素で乖離 → 多数決符号反転
- rtol=2e-2, atol=1e-2 で mergekit GTA と一致（43 テスト）

### 2.4 スパルシファイ

- **コンセンサス前に**各 delta へ適用
- `torch.topk`（`argsort` ではなく）— インデックス k 個のみ
- bf16/fp16 は f32 へ幅広げて topk（CPU）
- 方式: `l1`, `l2`, `gamma`, BS（n:m ブロック）

### 2.5 TV 事前抽出・再利用

- `extract_task_vector`: `T = W_source − W_base` を safetensors に保存
- 以降のマージは TV をストリーム（source model 再読不要）
- 指紋検証: ベース fingerprint が一致するかチェック

### 2.6 HF シャード型 safetensors I/O

- `model.safetensors.index.json` + `model-XXXXX-of-NNNNN.safetensors`
- 単一シャード → `model.safetensors`
- 同時に 1 テンソルのみ常駐

### 2.7 CLI + YAML レシピ

- `goatmerge extract` / `goatmerge merge` / `goatmerge inspect`
- `goatmerge merge -c recipe.yaml` で YAML からパラメータ読み込み
- CLI フラグは YAML 値を上書き

### 2.8 メタデータエンベローブ

- マージ結果に `metadata.json`（base、TV 一覧、settings、fingerprint）
- `inspect` で確認可能

### 2.9 比較ベンチマーク（GoatMerge vs mergekit GTA）

- 3×300 MB bf16 タスクベクトルで、ストリーミング（GoatMerge）vs stack 方式（mergekit GTA）を比較
- 各エンジンを独立サブプロセスで実行し、`/proc/self/status` VmRSS を 5 ms 間隔でサンプリング
- **増分メモリ**（ピーク − ベースライン）で、PyTorch ランタイム等の不可避オーバーヘッドを差し引いた純粋なマージ帰属メモリを計測
- 結果: GoatMerge ピーク 3395 MB / 増分 3009 MB、mergekit ピーク 7289 MB / 増分 6904 MB（比 0.436）
- 数値パリティ max|d| = 0.0625（bf16 rtol=2e-2 範囲で一致）
- 詳細: [`Benchmarks/RESULTS.md`](Benchmarks/RESULTS.md)

## 3. 残課題・将来展望

| # | 課題 | 現状 | 目標 |
|---|---|---|---|
| 1 | 大 tensor のチャンク分割 | `--chunk-elements` あり（未実装テスト） | 400B 級を 24–48 GB GPU で実行可 |
| 2 | 多 GPU 分配 | 未 | チャンク単位で GPU 間分配 |
| 3 | GPU カーネル | CPU 専用 | CUDA 対応（`torch` GPU 経路） |
| 4 | 大規模 AWA 探索 | 未 | λ/consensus/sparsify の自動探索 |
| 5 | RoMM ルーティング | 未 | 複数ベース間ルーティング |
| 6 | CABS 統合 | 未 | BS 剪定を GAT に統合（一部実装済み） |
| 7 | 並列 TV 抽出 | 未 | k 個 TV を並列抽出 |

## 4. ファイル構成（現状）

```
goatmerge/
  __init__.py      # パッケージ
  cli.py            # CLI（extract/merge/inspect + YAML -c）
  consensus.py      # ConsensusAccumulator（ストリーミングカーネル）
  extract.py        # TV 抽出
  fingerprint.py    # 指紋検証
  hf.py             # HF モデルディレクトリ解決
  inspect.py        # モデル検査
  io.py              # ShardReader, TensorWriter, ShardedTensorIndex
  merge.py            # merge_model, merge_tensor
  metadata.py         # メタデータエンベローブ
  sparsify.py         # スパルシファイカーネル
tests/
  test_consensus_merge.py   # パリティ + マージ
  test_fingerprint.py
  test_io.py
  test_metadata.py
  test_sparsify.py
  measure_peak_ram.py
examples/
  merge_recipe.yaml   # YAML レシピ例
Benchmarks/
  benchmark_compare.py  # GoatMerge vs mergekit GTA 比較ベンチマーク
  RESULTS.md            # ベンチマーク結果
AGENTS.md             # プロジェクトガイド（gitignore）
ambition.md           # 野心記録（gitignore）
```

## 5. 環境

- ワークスペース: `/home/CloudGoat/llms_merge/GoatMerge`
- インタープリタ: `/home/CloudGoat/venvs/mergekit/bin/python`
- 依存: Python ≥ 3.10, PyTorch, `safetensors`, `pyyaml`
- Git: `CroudGoat/GoatMerge`（private, master）
