# メモリモデル

`S` は最大テンソル 1 個のバイト数である。

- 設計予算（標準パス）：ピークは約 5–7 S である。
内訳は base、`acc`、`l1`、`c`、符号、delta 1 個、sparsify 一時領域である。
- Chunk Merge パス：O(S) + O(chunk) である。
内訳は出力バッファ 1 個とチャンク分の作業領域である。
詳細は [chunk-merge.md](chunk-merge.md) を参照する。
- TV 抽出：標準は 2 S であり、`--chunk-elements` 指定時は O(S) + O(chunk) である。
- 指紋のアンカーハッシュ：O(chunk) のストリーミング方式であり、約 3.02 S から約 0.62 S に改善した。

## 実測値

- 合成の 100 MB bf16 テンソル（2 TV、consensus=sum）ではピーク 105.6 MB であり、Peak / S = 1.06 である。
パス間でテンソルを解放するため（`l1` は除数組み立て前に解放し、各 `delta` は蓄積直後に解放する）、最悪見積もりの 5–7 S を下回る。
- 隔離ハーネス（`tests/measure_peak_ram.py`、子プロセス隔離）では標準パス 6.14 S、Chunk Merge 3.21 S である（100 MB bf16、consensus=sum）。
- GoatMerge と mergekit GTA の比較（300 MB bf16 3 個、consensus=sum）ではピーク RSS が 3394.6 MB 対 7289.4 MB であり、比率は約 47% である。
マージ帰属分だけでは約 44% である。
数値差の最大は max|d| = 0.0625 である。

規模の目安（1 層あたり）は次のとおりである。

| モデル規模 | S（bf16、1 層） | GoatMerge | mergekit GTA |
|---|---|---|---|
| 9B 級 | ≈ 1.8 GB | ≈ 2–4 GB | ≈ 22–34 GB |
| 400B 級 | ≈ 1.7 GB | ≈ 2 GB | ≈ 34–53 GB（単一 GPU では困難） |

## 自分で測る

```bash
/home/CloudGoat/venvs/mergekit/bin/python -m pytest   # 97 passed が基準
/home/CloudGoat/venvs/mergekit/bin/python tests/measure_peak_ram.py
```

`tests/measure_peak_ram.py` は pytest の `testpaths` の外にある。
フィクスチャ構築と各モードを子プロセス隔離で実行し、標準パスと Chunk Merge の peak/S を両方報告する。
GoatMerge と mergekit の比較スクリプトと結果は `Benchmarks/` にある。

ページキャッシュの注意：プロセス RSS だけでは全体像はわからない。
一括処理では `--drop-caches` でシステム側のファイルキャッシュ膨張を抑える。
9.3 GB モデルでは +17.2 GB から +2.7 GB に減り、96s から 45s に短縮した。
プロセス peak はどちらも 2.3 GB である（[io.md](io.md)）。
