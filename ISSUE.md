# Mergekit Task Arithmetic マージ — 課題と改善案

対象: mergekit (GitHub main, v0.1.4, `/home/CloudGoat/venvs/mergekit`) の Task Arithmetic (GTA) マージパス。
参照実装:
- カーネル: `mergekit/merge_methods/generalized_task_arithmetic.py`
- スパース化: `mergekit/sparsify.py`
- ストリーミング実証例: `Smart-Task-Arithmetic/taskvector/merge.py`
- 同型問題を持つCABSカーネル: `mergekit-cabs/mergekit_cabs/methods.py`

## 1. 現行の実態

Tensorごとのパイプライン:

1. `GatherTensors` がそのtensorの **全モデル (k+1) のテンソルを同時にロード**
2. `GTATask.execute` が各モデルについて `delta = W_i − W_base` を計算
3. 各deltaを `sparsify`（magnitude / magnitude_outliers / della_magprune）
4. **`torch.stack` で全deltaを (k, shape) に積み上げ** → 重み乗算 → consensus mask（sign/majority）→ sum → normalize → `base + mixed`

## 2. 課題（メモリ・効率の具体点）

### 2.1 `torch.stack` による全delta同時materialization（最大のメモリ消費）

1 tensorあたりのピーク（S = tensorサイズ, k = fine-tuned数）:

| 要素 | 量 |
|---|---|
| loaded tensors（Executorの`values`にマージ完了まで残存） | (k+1)·S |
| deltas | k·S |
| `torch.stack` | k·S |
| `weighted_deltas` | k·S |
| consensus mask（`sign` + `sign_weight` + bool mask） | ≈(2k+3)·S |
| `mixed` / `divisor` / 結果 | 3·S |

**合計 ≈ (4k+4)·S（consensusなし）～ (6k+7)·S（consensusあり）**、さらにsparsify中のtemp（`abs`コピー + 全`argsort`インデックス N·4B + mask + masked）が **deltaあたり ≈ 4S + N·4B** 追加。

- 9B bf16（最大tensor ≈110MB, k=4）: **ピーク ≈ 2.2〜4 GB**
- 400B級（最大tensor ≈1.7GB, k=4）: **≈ 34〜53 GB** → 単一GPUで不可行

### 2.2 全`argsort`のO(N log N)と巨大インデックスアロケーション

`magnitude`/`magnitude_outliers` は `torch.argsort` で **全N要素のインデックス (N·4B)** をmaterialize。N=45Mで180MB、densityが小さいほど無駄が大きい。CPUではbf16→f32のアップキャストでabs tempが2倍に。

### 2.3 loaded tensorの滞留

`get_task_vectors` はローカルで `del tensors[model]` するが、Executorの `values` 辞書は LoadTensor の結果を **マージ完了まで保持**（`last_use_index` によるevictionはGTATask完了後）。計算中に「loaded + delta」が二重で同居。

### 2.4 dtype変換のコピー

`x = tensors[model].to(base.dtype)` — sourceがF32・baseがBF16等の場合、全Sのコピー。

### 2.5 意味論的注意点（軽微だが実害あり）

- `divisor[divisor == 0] = 1` — 重み同定で0の位置を静かに1除算に置換
- size不一致tensorは警告のみで**静かにスキップ**（マージの欠落が検知しにくい）
- embedはサブマトリックス切り取りで警告のみ

## 3. 改善案（効率化・省メモリ化）

### A. ストリーミング累加（`torch.stack` 廃止）— 最大の効果

```python
acc = base.clone()
for i in models:
    delta = load(model).clone().sub_(base)   # in-place
    delta = sparsify_inplace(delta)
    acc.add_(delta, alpha=weight_i)        # in-place
    # delta は即解放
```

- ピーク: **base + acc + 1 delta + sparsify temp ≈ 5S**（k=4, S=110MBで **≈550MB**）→ 現行比で **4〜7倍削減**
- consensus（TIES）もstack不要で実装可能:
  - **Pass 1**: `acc = Σ αᵢδᵢ`（in-place）+ 位置ごとの符号カウント `cnt`（int8, **S bytes**）
  - majority = `sign(acc)`（sum法）または `sign(cnt)`（count法）
  - **Pass 2**: 各TVをディスクから再読し、符号一致位置のみ `acc.add_(αᵢδᵢ·mask)`
- これは既存の **Smart-Task-Arithmetic（`taskvector`）** が実証済みの方式（9B+4TVで ~0.4GB vs load-all ~90GB）。**ギャップ: sparsify/consensus/normalize未統合** → ここに統合する。

### B. タスクベクトル(TV)の事前抽出・再利用

- `T_i = W_i − W_base` を**1回だけ**safetensorsに保存し、以降のマージはTVをストリーム
- λ/consensus/sparsifyの再実験（AWA探索、RoMMルーティング反復）でsource modelを再読不要 → I/O削減
- コスト: ディスク k×model size（sequential readで安価）

### C. `topk` で全`argsort`を置換

- `torch.topk(w, k)` はインデックス **k個のみ**（N·4B → k·4B）、O(N log k)
- N=45M, density=0.1: 180MB → 18MB（10倍）、density=0.5: 180MB → 90MB
- `magnitude_outliers` は top-γ と bottom-(1−d−γ) の2回partial selectionで同型化

### D. BS（n:m ブロック）剪定をGATカーネルに追加

- ブロックごとの `blocks.topk(n, dim=1)` — 全N sort不要、O(N log m)、アロケは N·(n/m)·4B のみ
- CABS系スパース化をGATに統合し、グローバルsortを回避（mergekit-cabsのカーネルも同じstack問題を持つため共通有効）

### E. in-place化と即解放

- `x.sub_(base)`（clone後）、`delta *= mask`、`acc.add_(t, alpha)` — 全tempのコピーを削減
- **LoadTensorの即消費**（per-model delta task化）で `values` 滞留を解消 → loaded tensorは1個ずつのみ

### F. int8 mask / 符号カウンタの圧縮

- consensus maskを既定int8（1B/要素）→ bf16比 ½、f32比 ¼
- 符号カウントはint8（S bytes）で保持

### G. 大tensorのチャンク分割処理

- 1 tensorを64〜256MBチャンクに分割し逐次処理 → **ピークをチャンクサイズで上限設定**（400B級でも24〜48GB GPUで実行可）
- 注意: global magnitude剪定はチャンク局所でない → 2 pass（quantile閾値算出→適用）またはBS剪定（チャンク局所以来）と組み合わせる（RoMMの `--chunk-elements` と同思想）

### H. 多GPUへのチャンク分配

- チャンク単位でGPU間分配 → per-GPUピーク = チャンク+acc（現行はtensor単位islandで1.7GB×(4k+4)がVRAM超過し得る）

### I. CPUアップキャスト回避

- CPUでbf16 topkを許容（`torch.topk`はbf16対応）→ f32 temp（2S）を削減

## 4. 優先順位（効果/工数比）

| # | 改善 | 効果 |
|---|---|---|
| 1 | ストリーミング累加（stack廃止）+ 2-pass consensus | ピーク (4k+4)S → ~5S（k=4で4〜7倍） |
| 2 | TV事前抽出・再利用 | 再実験I/Oほぼゼロ、ストリーム前提 |
| 3 | in-place化 + loaded即解放 | temp 2〜3S削減 |
| 4 | topk置換 | sort時間・インデックスメモリ大幅減 |
| 5 | BS剪定追加 | 全局sort回避、CABS統合 |
| 6 | int8 mask | maskメモリ ½〜¼ |
| 7 | チャンク分割 + 多GPU分配 | 70B〜400Bを実行可能に |
