# メモリモデル

`S` = 最大テンソル 1 個のバイト数。

- **設計予算（標準パス）：** ピーク ≈ 5–7 S
  （base + `acc` + `l1` + `c` + 符号 + delta 1 個 + sparsify 一時領域）。
- **Chunk Merge パス：** O(S) + O(chunk) — 出力バッファ 1 個 + チャンク分の作業領域。
  詳しくは [chunk-merge.md](chunk-merge.md)。
- **TV 抽出：** 標準 2 S、`--chunk-elements` 指定時は O(S) + O(chunk)。
- **指紋のアンカーハッシュ：** O(chunk) のストリーミング方式（約 3.02 S → 約 0.62 S に改善）。

## 実測値

- 合成の 100 MB bf16 テンソル（2 TV、consensus=sum）：ピーク **105.6 MB**
  （Peak / S = **1.06**）。パス間でテンソルを解放してるので（`l1` は除数組み立て前に解放、
  各 `delta` は蓄積直後に解放）、最悪見積もりの 5–7 S より小さくなる。
- 隔離ハーネス（`tests/measure_peak_ram.py`、子プロセス隔離）：標準パス **6.14 S**、
  Chunk Merge **3.21 S**（100 MB bf16、consensus=sum）。
- GoatMerge vs mergekit GTA（300 MB bf16 × 3、consensus=sum）：ピーク RSS
  **3394.6 MB** vs **7289.4 MB**（約 47%。マージ帰属分だけだと約 44%）、
  max|d| = 0.0625。

規模の目安（1 層あたり）：

| モデル規模 | S（bf16、1 層） | GoatMerge | mergekit GTA |
|---|---|---|---|
| 9B 級 | ≈ 1.8 GB | ≈ 2–4 GB | ≈ 22–34 GB |
| 400B 級 | ≈ 1.7 GB | ≈ 2 GB | ≈ 34–53 GB（単一 GPU では困難） |

## 自分で測る

```bash
/home/CloudGoat/venvs/mergekit/bin/python -m pytest   # 97 passed が基準
/home/CloudGoat/venvs/mergekit/bin/python tests/measure_peak_ram.py
```

`tests/measure_peak_ram.py` は pytest の `testpaths` 外。フィクスチャ構築と各モードを
子プロセス隔離で実行し、標準パスと Chunk Merge の peak/S を両方報告する。
GoatMerge と mergekit の比較スクリプトと結果は `Benchmarks/` にある。

ページキャッシュの注意：プロセス RSS だけ見ても全体像は分からない。一括処理では
`--drop-caches` で**システム側の**ファイルキャッシュ膨張を抑える（9.3 GB モデルで
+17.2 GB → +2.7 GB、96s → 45s、プロセス peak は同じ 2.3 GB）。[io.md](io.md) 参照。
