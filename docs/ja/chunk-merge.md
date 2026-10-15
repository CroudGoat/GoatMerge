# Chunk Merge

**Chunk Merge**：チャンク分割によるマージ経路の名前。
以後「Chunk Merge」と表記する。

`--chunk-elements N`（または YAML の `chunk_elements`）を指定すると、N 要素を超えるテンソルは行チャンクに分割して処理する。
行チャンクは dim 0 の行であり、trailing dims の積が行あたり要素数になる。
処理はディスクからの読み込み、蓄積、出力バッファへの書き込みの繰り返しである。

- 出力は `full = torch.empty(base_shape)` に 1 回だけ確保し、各チャンクの結果を `full[start:end]` に埋める。
- ピーク RAM は O(S) + O(chunk) になる（S は出力テンソル自体であり、回避できない）。
- 大域的なスパース化（`magnitude` と `magnitude_outliers`）はテンソル全体を 1 回スキャンして閾値とタイ数だけを求める（O(1) メモリ）。
その後各チャンクに適用する（`GlobalMasker(mask_params)`）。
両パスは同じマスクを使う。
- コンセンサスの除数に再読は不要であり（符号キャッシュ）、各テンソルの読みは正確に 1 回である。
乱数ベースのマスクはチャンクごとのシード（`merge_generator(key, source, chunk)`）で決まるため、パス間で常に一致する。
- 結果は通常経路の出力と一致する（同一カーネル、同一マスク）。
- `slerp` は全域ノームに依存するため Chunk Merge に対応せず、警告とともに通常経路にフォールバックする。
- `extract --chunk-elements` も同じ行チャンク方式を使う（出力 `full` に 1 回確保してスライス代入する）。
対象は base と source の形状が一致するテンソルだけであり、不一致（submatrix と mismatch）は `load_delta` の共有ルールを持つ非チャンクパスに落ちる。
形状判定は `shape_fit` に一元化する。

行読み（`io._read_rows`）は 1 回の `readinto` でバッファを 1 個だけ確保し、そのバッファを所有するテンソルを返す。
`bytes` から `bytearray` を経て `clone` する 3 コピーは廃止した。
リード間のエイリアスはない。

等価性は `tests/test_chunk_merge.py`（方式、コンセンサス、1〜3 次元テンソルの組み合わせとスパース化の正当性）と `tests/test_consensus_merge.py`（チャンク抽出の等価性とフォールバック）が強制する。
