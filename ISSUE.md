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

### 2.1 ストリーミング蓄積（`torch.stack` 廃止）

```python
acc = torch.zeros_like(base)
for i, entry in enumerate(entries):
    delta = load_delta(base, entry.dir, entry.kind, key, readers, clone=False)
    sparsify_delta(delta, ...)     # コンセンサス前、in-place
    delta.mul_(torch.tensor(alpha, dtype=delta.dtype))  # bf16·bf16 テンソル積
    acc.add_(delta)                # in-place
    l1.add_(delta.abs())
    # delta は即解放。テンソルあたりディスク読みは正確に 1 回
```

- ピーク: **6.14 S**（実測、`tests/measure_peak_ram.py`、100 MB bf16、
  consensus=sum、2 TV）。設計目標 5–7 S の範囲内
- 読みは 1 回の `readinto` でバッファを 1 つだけ確保（`bytes → bytearray →
  clone` の 3 コピーを廃止、`io.py::_read_rows`）
- merge 系の全 load は `clone=False`（direct リーダーは既所有ストレージを
  返す。mmap 使用箇所はコードベースに存在しない）

### 2.2 コンセンサス恒等式＋符号キャッシュ divisor

```
mixed = (acc + M · l1) / 2,   M = sign(majority) = ±1
divisor = Σ αᵢ·[signᵢ == M]    # 2 ビットパック符号キャッシュから組立
divisor[divisor == 0] = 1
result = (base + mixed).to(base.dtype)
```

- 全 delta のマスク積を 2 回の in-place `add_`/`mul_` に置き換え
- int8 符号カウンタ（count 方式）
- divisor 用のストリーミング**2 周目を廃止**: 蓄積中に 2 ビットパック符号
  （`_pack_signs`、k·S/4）をキャッシュし、再読・再 sparsify なしで組立
  てる。`normalize=False` では符号キャッシュ自体を取らない
- 副次効果: consensus merge のディスク読みが TV あたり 2 回→1 回に
  （実測: 1 テンソル×2 TV で 6 回→4 回の row-read）

### 2.3 数値パリティ

- 重み付き積 `δᵢ · αᵢ` は **bf16·bf16 テンソル積**（`stacked · weights` と一致）
- `add_(alpha=scalar)` では内部積精度が近接要素で乖離 → 多数決符号反転
  （このため §4.3 の「スカラー直接化」は**採用しない**。不変条件 #1）
- rtol=2e-2, atol=1e-2 で mergekit GTA と一致（**97 テスト**）

### 2.4 スパルシファイ

- **コンセンサス前に**各 delta へ適用。各 delta は正確に 1 回だけ
  sparsify される（divisor 用の再適用は §2.2 で消滅）
- `torch.topk`（`argsort` ではなく）— インデックス k 個のみ
- 方式: `magnitude` | `random` | `magnitude_outliers` | `della_magprune` | `bs`
- マスク計算は `chunk_elements` 未指定でも内部バジェット（8M 要素）で
  常時チャンク化: della の int64 double-`argsort`（16 B/要素）が O(chunk)
  に収まる。決定的手法はビット一致、確率的手法は同一 seed で決定的
- `bernoulli` は in-place 描画（`full_like` 確率テンソルを廃止）
- 確率的マスクの generator は `(テンソル名, ソース番号, チャンク番号)` で
  Seeding（実行ごとの再現性を保証）

### 2.5 TV 事前抽出・再利用

- `extract_task_vector`: `T = W_source − W_base` を safetensors に保存
- 減算はインプレースでピーク 2 S、`--chunk-elements` 指定時は行チャンク
  抽出で O(S) + O(chunk)。結果は標準パスとビット一致
- 以降のマージは TV をストリーム（source model 再読不要）
- 指紋検証: ベース fingerprint が一致するかチェック（出口コード 3 で拒否）

### 2.6 HF シャード型 safetensors I/O

- `model.safetensors.index.json` + `model-XXXXX-of-NNNNN.safetensors`
- 単一シャード → `model.safetensors`
- 同時に 1 テンソルのみ常駐
- **ストリーミング書き出し**: manifest からシャード分割・オフセット・
  ヘッダを事前確定し、到着順に 64 MB スライスで追記。ステージングも
  `save_file` バッファも持たない。ヘッダは正準ソート順で決定的
  （`save_file` 自身のキー順序は実装定義のためバイト一致は求めない）
- **ページキャッシュ対策**（opt-in `--drop-caches`）: 読み済み範囲・
  書き済みシャードに `POSIX_FADV_DONTNEED` を発行。再読ペナルティは
  ない（各テンソルは正確に 1 回読み）。実測は §2.9 参照

### 2.7 CLI + YAML レシピ

- `goatmerge extract` / `goatmerge merge` / `goatmerge inspect`
- `goatmerge merge -c recipe.yaml` で YAML からパラメータ読み込み
- CLI フラグは YAML 値を上書き
- `extract --chunk-elements`、`--drop-caches`（両サブコマンド、merge は
  YAML の `drop_caches` にも対応）

### 2.8 メタデータエンベローブ

- マージ結果に `goatmerge.json`（base、TV 一覧、settings、fingerprint）。
  HF の `config.json` はベースからコピーされるため、出力はそのまま
  `transformers` で読み込める
- `inspect` で確認可能

### 2.9 ベンチマーク・実測

- **GoatMerge vs mergekit GTA**（[`Benchmarks/RESULTS.md`](Benchmarks/RESULTS.md)）:
  3×300 MB bf16 タスクベクトルで増分メモリ比 **0.436**、数値パリティ
  max|d| = 0.0625（bf16 rtol=2e-2 範囲で一致）
- **ピーク RAM**（`tests/measure_peak_ram.py`、フィクスチャ構築と各モードを
  子プロセス隔離で計測）: 標準パス **6.14 S**、Chunk Merge **3.21 S**
  （100 MB bf16、consensus=sum）
- **指紋**: アンカーの行チャンク・ストリーミングハッシュで 3.02 S → 0.62 S
- **ページキャッシュ A/B**（Baphomet レシピ実走行、9.3 GB モデル、
  S=1.27 GB の embed を含む）: 通常 96s・キャッシュ +17.2 GB に対し
  `--drop-caches` で **45s・キャッシュ +2.7 GB**。プロセス peak は両方
  2.3 GB。残余は出力シャードの writeback 中ページ（fdatasync でゼロに
  できるが低速ディスクで約 2 倍遅くなるため速度優先で外した。
  `io.py::_close_shard` のコメント参照）

## 3. 省メモリ化・高速化の改善ポイント（処理状況）

§4 と違い、ここは調査で挙がった個別項目の処理ログである。
**全項目が処理済み**（対応 or 却下）。

### 3.1 `TensorWriter` — 全テンソルをRAMに滞留 → 対応済み

- **旧現状:** `save_tensor()` が全テンソルを `current_shard`（dict）に蓄積し、
  `save_file` がシャード全体のシリアライズバッファを確保（書き出し時に 2 S）。
- **対応:** ストリーミング書き出し（§2.6）。12×32 MB 連続書きで 466 MB
  （旧フロー換算 ~1.2 GB）。実モデル merge（738 テンソル、10 シャード）で
  ヘッダ tiling・キー網羅・値を検証。

### 3.2 `SlerpKernel.finish` — float32一時テンソル → 対応済み

- 現在は base dtype のまま計算し、フルサイズ一時テンソルは `v2` 1 個のみ
  （`kernels.py` の docstring 参照）。追加作業なし。

### 3.3 `_weighted` / `accumulate` の1要素テンソル生成 → 却下（設計）

- `torch.tensor(weight)` ラッパーを除去し `delta.mul_(weight)`（Python float
  直接）にする案は、bf16·bf16 テンソル積のパリティを壊すため**採用しない**。
  参照実装（`stacked * weights`）との内部積精度の一致が不変条件 #1。
  微小オーバーヘッドは parity の対価として維持する。

### 3.4 `della_mask` — int64インデックス全サイズ割り当て → 対応済み

- `sparsify_inplace` の既定チャンク予算（8M 要素）で常時分割。
  行境界アライン＋同一 generator のため mask は決定的。
  既定バジェットと明示チャンクの一致はテストで強制。

### 3.5 `fingerprint.tensor_content_hash` — float32コピー → 対応済み

- アンカーを行チャンクでストリーミングハッシュ（O(chunk)）。
  ダイジェストは一括ハッシュと同一（チャンク幅・dtype を変えて検証）。

### 3.6 `magnitude_mask` / `bs_mask` — 全サイズ一時テンソル → 対応済み

- §4.4 と同じ既定チャンク予算で two-pass 化。決定的手法はビット一致
  （`test_chunked_magnitude_matches_global` 等で強制）。

### 3.7 `bernoulli_mask` — `torch.full_like` → 対応済み

- `torch.empty` + in-place `bernoulli_` に置換。同一 seed で決定的
  （torch CPU RNG は形状依存のため旧実装とはビット不一致になり得るが、
  分布は同一。Chunk Merge が既に持つ性質と同類）。

## 4. ファイル構成（現状）

```
goatmerge/
  __init__.py      # パッケージ（公開 API の再エクスポート）
  cli.py            # CLI（extract/merge/inspect + YAML -c）
  consensus.py      # ConsensusAccumulator / GtaKernel（符号キャッシュ divisor）
  extract.py        # TV 抽出（インプレース減算・行チャンク抽出）
  fingerprint.py    # 指紋検証（ストリーミングアンカーハッシュ）
  hf.py             # HF モデルディレクトリ解決
  inspect.py        # モデル検査
  io.py              # ShardReader, TensorWriter（ストリーミング）, ShardedTensorIndex
  kernels.py         # linear / mixture / slerp / ties カーネル
  merge.py           # merge_model, merge_tensor（Chunk Merge 含む）
  merge_method.py    # MergeMethod / MergeKernel / build_kernel
  metadata.py        # メタデータエンベローブ（goatmerge.json）
  sparsify.py        # スパルシファイカーネル（既定チャンク予算付き）
tests/
  test_consensus_merge.py   # パリティ + マージ + extract/指紋 e2e
  test_chunk_merge.py       # Chunk Merge 等価性
  test_kernels.py           # 各マージメソッドのカーネル
  test_fingerprint.py
  test_io.py                # I/O + ライタ（等価性・順序違反・未完除去）
  test_metadata.py
  test_sparsify.py          # 各方式 + 既定チャンク等価性
  measure_peak_ram.py       # ピーク RAM 計測（標準 + Chunk 両報告）
examples/
  merge_recipe.yaml   # YAML レシピ例
Benchmarks/
  benchmark_compare.py          # GoatMerge vs mergekit GTA 比較ベンチマーク
  benchmark_task_arithmetic.py  # 実モデルでの Task Arithmetic ベンチマーク
  RESULTS.md                    # ベンチマーク結果
AGENTS.md             # プロジェクトガイド（gitignore）
ambition.md           # 野心記録（gitignore）
```

## 5. 環境

- ワークスペース: `/home/CloudGoat/llms_merge/GoatMerge`
- インタープリタ: `/home/CloudGoat/venvs/mergekit/bin/python`
- 依存: Python ≥ 3.10, PyTorch, `safetensors`, `pyyaml`
- Git: `CroudGoat/GoatMerge`（private, master）
- テスト: `pytest`（97 passed が基準）
