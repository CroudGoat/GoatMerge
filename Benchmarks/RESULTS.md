# GoatMerge vs mergekit GTA — 比較ベンチマーク結果

## 設定

- **テンソル**: 300 MB bf16 テンソル × 3（タスクベクトル）＋ ベース 300 MB
- **consensus**: `sum`、**weights**: `[0.5, 0.7, 0.3]`、`lambda_ = 1.0`、`normalize = True`
- **dtype**: bfloat16（2 バイト/要素）、要素数 157,286,400（= 300 MB）

## 結果

| エンジン | 時間 (s) | ピーク RSS (MB) | 増分 (MB) | 備考 |
|---|---|---|---|---|
| **GoatMerge**（ストリーミング） | 1.819 | 3394.6 | 3009.1 | `torch.stack` 不使用 |
| **mergekit GTA**（stack 方式参照） | 2.408 | 7289.4 | 6903.8 | 全 delta をスタック |
| 数値パリティ max\|d\| | 0.0625 | — | — | bf16 rtol=2e-2 範囲で一致 |

- **速度比**（mergekit / GoatMerge）: **1.32x** — mergekit は GoatMerge の約 1.3 倍の時間
- **メモリ比**（GoatMerge / mergekit）: **0.466** — GoatMerge は mergekit の約 **47%** のピーク RAM
- **増分メモリ比**（GoatMerge / mergekit）: **0.436** — マージ操作自体の純粋なメモリ増加分では約 **44%**

## 要点

- **速度**: mergekit が 1.32x 速い。stack 方式は `torch.stack` で一括処理するため、ストリーミングより若干有利。
- **メモリで差が出る**：mergekit GTA は 3 つの delta を `torch.stack` して 900 MB のスタック＋`weighted`/`sign`/`mask` 等を物化しピーク 7.3 GB。GoatMerge は 1 個ずつストリーミングし `acc`/`l1`/`c` だけを持つためピーク 3.4 GB（約 47%）。
- **増分メモリ**（ピーク − ベースライン）: GoatMerge は 3009 MB、mergekit は 6904 MB。マージ操作に帰属する純粋なメモリ増加では GoatMerge が約 **44%** である。
- **数値パリティ**：max|d| = 0.0625 で bf16 の許容範囲（rtol=2e-2, atol=1e-2）内に収まり、GoatMerge のストリーミング実装は mergekit GTA と数値的に一致。

## 計測方法

- 各エンジンを**独立サブプロセス**で実行し、`/proc/self/status` の **VmRSS を 5 ms 間隔でサンプリング**してピークを計測。
- `ru_maxrss`（生涯ハイウォーターマーク）はテンソル生成時の float32 一時領域を含み過大評価するため、真のピークを得るため VmRSS サンプリングを採用。
- **増分メモリ** = ピーク RSS − ベースライン RSS（マージ直前に読み取り）。PyTorch ランタイムやプロセス起動の不可避オーバーヘッドを差し引いた、マージ操作に帰属する純粋なメモリ増加を示す。
- 数値パリティは第 3 のサブプロセスで、GoatMerge の `merge_tensor` と mergekit GTA 参照カーネル（`_ref_gta`）の最大絶対差を報告。

## 再実行

リポジトリルートが `PYTHONPATH` にある状態で:

```bash
python Benchmarks/benchmark_compare.py
```

スクリプト: [`benchmark_compare.py`](./benchmark_compare.py)
