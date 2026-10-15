# GoatMerge の概要

**GoatMerge** は、ベースモデルにタスクベクトルをマージするストリーミング型のタスク算術エンジンである。
**タスクベクトル**：ファインチューン済みモデルの重みからベースモデルの重みを引いた差分。
以後「TV」と表記する。
GoatMerge は `torch.stack` で全 delta を積み上げる代わりに、delta を 1 個ずつストリーミングして in-place に蓄積する。
この方式によりピークメモリを抑えながら、mergekit の GTA（Generalized Task Arithmetic）との数値パリティを保つ。

- `torch.stack` を使わない：delta を 1 個ずつストリーミングして in-place に蓄積する。
- ピーク RAM が小さい：設計予算は 5–7 S である（S は最大テンソル 1 個のバイト数）。
100 MB の bf16 テンソルでは 1.06 S を実測した（[memory.md](memory.md)）。
- mergekit GTA との数値パリティ：bf16 テンソルで rtol = 2e-2、atol = 1e-2 の範囲で一致する。
- **Chunk Merge**：行チャンクに分割するマージ経路。以後「Chunk Merge」と表記する。
どんなサイズのテンソルも O(S) + O(chunk) のメモリで処理し、結果は通常経路の出力と一致する（[chunk-merge.md](chunk-merge.md)）。
- HF シャード型 safetensors を直接読む：標準的な HuggingFace 配置をそのまま読み書きする（[io.md](io.md)）。
- タスクベクトルを抽出して再利用する：`extract` で `T = W_source − W_base` を一度だけ計算し、以降のマージではソースモデルを読み込まない（[quickstart.md](quickstart.md)）。
- ベース指紋で取り違えを防ぐ：別のベースで作られた TV を混ぜると終了コード 3 で停止する（[fingerprint.md](fingerprint.md)）。

## GoatMerge を使う理由

mergekit の GTA は、1 テンソルを処理するために全 delta を `torch.stack` で `(k, shape)` に積み上げる。
k = 4 の 400B 級モデル（S ≈ 1.7 GB）ではピークが 34–53 GB に達し、単一 GPU では実行できない。
そこで GoatMerge は delta を 1 個ずつ流し、`acc`、`l1`、`c` の 3 個の蓄積領域だけでコンセンサスを求める。
コンセンサスのマスク和は厳密な恒等式 `(acc + M·l1)/2` と等価なため、全 delta のマスク積をメモリ上に展開しない。

| 観点 | GoatMerge | mergekit GTA |
|---|---|---|
| delta の扱い | 1 個ずつストリーミング | `torch.stack` で全件積み上げ |
| ピーク RAM（k 個の TV、1 テンソル） | ≈ 5–7 S | (4k+4) S – (6k+7) S |
| コンセンサス | 恒等式 `(acc + M·l1)/2` | weighted 全件のマスク積 |
| 大テンソル | Chunk Merge で O(S) + O(chunk) | 全件をメモリに保持 |
| I/O | HF シャード型 safetensors | HF シャード型 safetensors |

ベンチマーク条件は 300 MB の bf16 テンソル 3 個、consensus=sum である。
この条件で GoatMerge のピーク RSS は mergekit の約 47% であり、マージ処理に帰属する増分メモリでは約 44% である。
数値差の最大は max|d| = 0.0625 である。
計測方法は [testing.md](testing.md) に記す。

## 読み方

初めて利用する開発者は [quickstart.md](quickstart.md) から読む。
日常的にマージを実行する開発者は [cli.md](cli.md) と [yaml-recipes.md](yaml-recipes.md) を手元に置く。
品質を調整する開発者は [merge-methods.md](merge-methods.md) を読んでから [sparsification.md](sparsification.md) を読む。
大テンソルや厳しいメモリ制限に対処する開発者は [chunk-merge.md](chunk-merge.md) と [memory.md](memory.md) を読む。
実装を変更する開発者は、先に [architecture.md](architecture.md) を読む。
同ファイルは破ってはならない不変条件をまとめる。
