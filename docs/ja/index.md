# GoatMerge — 概要

**GoatMerge** は、ファインチューン済みモデル（タスクベクトル、TV）をベースモデルに
マージする**ストリーミング型**のタスク算術エンジンだ。`torch.stack` で全 delta を
積み上げる代わりに 1 つずつストリーミングし、ピークメモリを大幅に抑えながら
mergekit の GTA（Generalized Task Arithmetic）と数値パリティを保つ。

- **`torch.stack` を使わない** — delta を 1 個ずつストリーミングして in-place 蓄積。
- **ピーク RAM ≈ 5–7 S**（S = 最大テンソルのバイト数）。100 MB bf16 テンソルで
  実測 **約 1.06 S**、隔離計測ハーネスでは **6.14 S**（[memory.md](memory.md) 参照）。
- **mergekit GTA との数値パリティ** — bf16 テンソルで rtol = 2e-2、atol = 1e-2。
- **Chunk Merge（チャンク分割マージ）** — どんなサイズのテンソルも O(S) + O(chunk)
  で処理。結果は通常経路と完全一致（[chunk-merge.md](chunk-merge.md)）。
- **HF シャード型 safetensors** — 標準的な HuggingFace 配置をそのまま読み書き
  （[io.md](io.md)）。
- **タスクベクトルの事前抽出と再利用** — `extract` で 1 度だけ
  `T = W_source − W_base` を計算し、以降はソースモデルを再読込しない
  （[quickstart.md](quickstart.md)）。
- **ベース指紋検証** — 別のベースで作られた TV の混入を拒否（終了コード 3、
  [fingerprint.md](fingerprint.md)）。

## なぜ GoatMerge なのか

mergekit の GTA は 1 テンソルを処理するだけで、全 delta を `torch.stack` で
`(k, shape)` に積み上げる。k = 4 の 400B 級モデル（S ≈ 1.7 GB）では
ピークが **34–53 GB** に達し、単一 GPU では実行不可能になる。

GoatMerge は delta を 1 個ずつストリームし、`acc` / `l1` / `c` という 3 つの
蓄積領域だけでコンセンサスを求める。コンセンサスのマスク和も
厳密な恒等式 `(acc + M·l1)/2` に置き換えるため、全 delta のマスク積を
物資化しない。

| 観点 | GoatMerge | mergekit GTA |
|---|---|---|
| delta の扱い | 1 個ずつストリーミング | `torch.stack` で全件積み上げ |
| ピーク RAM（k 個の TV、1 テンソル） | ≈ 5–7 S | (4k+4) S – (6k+7) S |
| コンセンサス | 恒等式 `(acc + M·l1)/2` | weighted 全件のマスク積 |
| 大テンソル | Chunk Merge で O(S) + O(chunk) | 全件 resident |
| I/O | HF シャード型 safetensors | HF シャード型 safetensors |

ベンチマーク（300 MB bf16 テンソル × 3 TV、consensus=sum）では、GoatMerge は
mergekit の **約 47%** のピーク RAM で、マージ処理そのものに帰属する増分メモリでは
**約 44%**。数値差は max|d| = 0.0625。計測方法は [testing.md](testing.md) 参照。

## 読み進め方

1. 初めてなら [quickstart.md](quickstart.md)。
2. 普段使いなら [cli.md](cli.md) と [yaml-recipes.md](yaml-recipes.md)。
3. 品質調整なら [merge-methods.md](merge-methods.md) → [sparsification.md](sparsification.md)。
4. 大テンソル・省メモリなら [chunk-merge.md](chunk-merge.md) と [memory.md](memory.md)。
5. 実装を触るなら先に [architecture.md](architecture.md) — 壊しちゃいけない
   不変条件がまとまってる。
