# I/O — シャード型 safetensors

HF シャード型 safetensors（`model.safetensors.index.json` +
`model-XXXXX-of-NNNNN.safetensors`、単一シャードなら `model.safetensors`）を
そのまま読み書きする。同時に resident するのは 1 テンソルだけだ。

モジュールは `goatmerge/io.py`：`ShardedTensorIndex` / `ShardReader` /
`TensorWriter` / `load_delta` / `load_delta_chunk` / `copy_support_files` /
`dtype_from_safetensors` / `safetensors_dtype_str` / `shape_fit` /
`advise_drop_caches`。

## 読み

- シャードのヘッダは読み込みごとにパースせず、`ShardReader` 内でキャッシュする。
- 行範囲の読み出しには mmap を使わず、safetensors ヘッダの `data_offsets` から
  直接バイトオフセットを計算して読む。
- `_read_rows` は 1 回の `readinto` でバッファを 1 つだけ確保し、そのバッファを
  所有するテンソルを返す。各 read は独立バッファなのでリード間のエイリアスはない
  （`tests/test_io.py::test_reader_clone_semantics` が強制）。
- mmap の view を in-place 変更しちゃダメ。`load_delta(..., clone=False)` は
  direct リーダーの既所有ストレージが前提。mmap リーダーはコードベースに存在しない。
  merge 系の全 load は `clone=False` だ。
- 形状判定は `shape_fit` に一元化し、`_prepare_delta` と extract の到着プランナーで
  共有する。不一致（submatrix / mismatch）は必ず非チャンクパスにフォールバックし、
  `load_delta` の共有ルールで処理する。

## 書き出し（`TensorWriter`）

到着順のストリーミングライタ。manifest（名前・dtype・形状）からシャード分割・
オフセット・ヘッダを事前確定する：

- テンソル到着順にバイト追記する。ステージング dict も `save_file` バッファも持たない。
- 計画外の到着・サイズ不一致は例外（黙って壊さない）。
- 未完シャードは破棄する（壊れたファイルは残さない）。
- ヘッダは正準ソート順で決定的（`save_file` 自身のキー順序は実装定義なので、
  バイト一致は求めない）。

## `--drop-caches`（opt-in）

読み済み範囲・書き済みシャードに `POSIX_FADV_DONTNEED` を発行する。
各テンソルは正確に 1 回しか読まない（除数再読は符号キャッシュで廃止済み）ので、
再読ペナルティはない。

書き出し側は dirty ページを待たない（fdatasync しない）ので、writeback 中の
残余が数 GB 出ることはある。速度優先の判断だ（`io._close_shard` のコメント参照）。
実測（9.3 GB モデル）：通常 96s・キャッシュ +17.2 GB に対し、`--drop-caches` で
**45s・キャッシュ +2.7 GB**。プロセス peak は両方 2.3 GB。

`tests/test_io.py`（シャード型 / 単一シャード / 行スライス、ライタの命名と分割、
drop-caches の advise 発行）がカバーしてる。
