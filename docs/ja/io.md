# I/O：シャード型 safetensors

HF シャード型 safetensors（`model.safetensors.index.json` と `model-XXXXX-of-NNNNN.safetensors` の組、単一シャードなら `model.safetensors`）をそのまま読み書きする。
同時にメモリに置くテンソルは 1 個だけである。

モジュールは `goatmerge/io.py` であり、公開要素は `ShardedTensorIndex`、`ShardReader`、`TensorWriter`、`load_delta`、`load_delta_chunk`、`copy_support_files`、`dtype_from_safetensors`、`safetensors_dtype_str`、`shape_fit`、`advise_drop_caches` である。

## 読み

- シャードのヘッダは読み込みごとにパースせず、`ShardReader` 内でキャッシュする。
- 行範囲の読み出しには mmap を使わず、safetensors ヘッダの `data_offsets` から直接バイトオフセットを計算して読む。
- `_read_rows` は 1 回の `readinto` でバッファを 1 個だけ確保し、そのバッファを所有するテンソルを返す。
各 read は独立バッファを使うため、リード間のエイリアスはない（`tests/test_io.py::test_reader_clone_semantics` が強制する）。
- mmap の view を in-place 変更してはならない。
`load_delta(..., clone=False)` は direct リーダーの既所有ストレージを前提とする。
mmap リーダーはコードベースに存在しない。
merge 系の全 load は `clone=False` で呼ぶ。
- 形状判定は `shape_fit` に一元化し、`_prepare_delta` と extract の到着プランナーで共有する。
不一致（submatrix と mismatch）は必ず非チャンクパスにフォールバックし、`load_delta` の共有ルールで処理する。

## 書き出し（`TensorWriter`）

`TensorWriter` は到着順のストリーミングライタである。
manifest（名前、dtype、形状）からシャード分割、オフセット、ヘッダを事前確定する。

- テンソル到着順にバイト追記する。
ステージング dict も `save_file` バッファも持たない。
- 計画外の到着とサイズ不一致は例外にする。
黙って破損させない。
- 未完シャードは破棄する。
壊れたファイルを残さない。
- ヘッダは正準ソート順で決定的である。
`save_file` 自身のキー順序は実装定義のため、バイト一致までは求めない。

## `--drop-caches`（opt-in）

読み済み範囲と書き済みシャードに `POSIX_FADV_DONTNEED` を発行する。
各テンソルを正確に 1 回だけ読む（除数再読は符号キャッシュで廃止した）ため、再読ペナルティはない。

書き出し側は dirty ページを待たない（fdatasync しない）ため、writeback 中の残余が数 GB 生じる場合がある。
これは速度優先の判断である（`io._close_shard` の注記を参照する）。
実測（9.3 GB モデル）では、通常 96s かつキャッシュ +17.2 GB に対し、`--drop-caches` では 45s かつキャッシュ +2.7 GB である。
プロセス peak はどちらも 2.3 GB である。

`tests/test_io.py`（シャード型、単一シャード、行スライス、ライタの命名と分割、drop-caches の advise 発行）がこの層をカバーする。
