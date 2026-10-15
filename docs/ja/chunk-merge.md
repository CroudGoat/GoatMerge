# Chunk Merge（チャンク分割マージ）

**Chunk Merge** は、チャンク分割によるマージ経路の名前だ。

`--chunk-elements N`（または YAML の `chunk_elements`）を指定すると、
1 テンソルが N 要素を超えるとき行チャンク（dim 0 の行 = trailing dims の積）に
分割されて処理される。ディスクから読み込み → 蓄積 → 出力バッファへ書き込み、
を繰り返す。

- 出力は `full = torch.empty(base_shape)` に **1 回だけ**確保し、各チャンクの
  結果を `full[start:end]` に埋める。
- ピーク RAM は **O(S) + O(chunk)** になる（S は出力テンソル自体で回避不能）。
- 大域的なスパース化（`magnitude` / `magnitude_outliers`）は、テンソル全体を
  1 回スキャンして閾値とタイ数だけを求め（O(1) メモリ）、各チャンクに適用する
  （`GlobalMasker(mask_params)`）。両パスで同じマスクだ。
- コンセンサスの除数に再読は不要（符号キャッシュ）。各テンソルの読みは正確に 1 回。
  乱数ベースのマスクはチャンクごとのシード（`merge_generator(key, source, chunk)`）で、
  パス間で常に同じになる。
- 結果は通常経路と**完全一致**する（同一カーネル・同一マスク）。
- `slerp` は全域ノームに依存するので Chunk Merge に対応してなくて、
  警告とともに通常経路にフォールバックする。
- `extract --chunk-elements` も同じ行チャンク方式（出力 `full` に 1 回確保して
  スライス代入）。base と source の形状が一致するテンソルのみが対象で、
  不一致（submatrix / mismatch）は `load_delta` の共有ルールを持つ非チャンクパスに
  落ちる。形状判定は `shape_fit` に一元化してる。

行読み（`io._read_rows`）は 1 回の `readinto` でバッファを 1 つだけ確保し、
そのバッファを所有するテンソルを返す。`bytes → bytearray → clone` の 3 コピーは
廃止済み。リード間のエイリアスもない。

等価性は `tests/test_chunk_merge.py`（方式 × コンセンサス × 1〜3 次元テンソル、
スパース化の正当性）と `tests/test_consensus_merge.py`（チャンク抽出の等価性・
フォールバック）で強制してる。
