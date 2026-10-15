# スパース化

利用者は `--method` で方式を指定する。
`--density 1.0` では無効になる。
スパース化はコンセンサスより前に各 delta へ適用する。
各 delta への適用回数は正確に 1 回である（[architecture.md](architecture.md) の不変条件 #3）。

| `method` | 動作 | 関連フラグ |
|---|---|---|
| `magnitude` | 絶対値が大きい上位 density の割合を保持する | — |
| `magnitude_outliers` | 上位 gamma と下位を落とし、中間を保持する | `--gamma` |
| `random` | 要素ごとに確率 density で保持する（DARE 流） | — |
| `della_magprune` | 行内の絶対値ランクに応じて保持確率を変える | `--epsilon` |
| `bs` | m 要素のブロックごとに上位 n 個を保持する | `--n`、`--m` |

実装は `goatmerge/sparsify.py` にある。

- `magnitude` と `magnitude_outliers` は全件 `argsort` ではなく `torch.topk`（インデックス k 個だけ）を使う。
大域しきい値はストリーミング exact 選択（`_select_threshold`、O(1) RAM、ソートとビット一致）で求める。
- マスクはすべて `uint8`（1 要素 1 バイト）であり、適用と rescale は in-place である。
- `bernoulli` は `bernoulli_` で in-place 描画し、`full_like` 確率テンソルを作らない。
- rescale の和は float32 に広げてから足す（`_norm`）。
この拡張によりチャンクパスは非チャンクパスの出力をビット単位で再現する。
- マスク計算は `chunk_elements` 未指定でも内部バジェット `_DEFAULT_SPARSE_CHUNK_ELEMENTS`（8M 要素）で分割する。
決定的手法の出力はビット単位で一致し、確率的手法の出力は同一 seed で決定的になる。
ただし torch の CPU RNG は形状依存のため、全体一括の出力とはビット不一致になり得る。
パス間の一致は保たれる。

## 決定論的マスク

乱数を使う方式（`random` と `della_magprune`）はテンソル名、ソース番号、チャンク番号の組から決定論的にシードする（`sparsify.merge_generator(key, source_index, chunk)`）。
このため同じ入力からは常に同じ結果が出て、同じソースへの繰り返しパスでもマスクが一致する。

## 大域マスクとチャンクマスク

- 大域的手法（`magnitude` と `magnitude_outliers`）はテンソル全体を 1 回スキャンしてしきい値とタイ数を決める（O(1) メモリ）。
その後各チャンクに同じマスクを適用する（`GlobalMasker(mask_params)`）。
両パスは同じマスクを使う。
- チャンク等価性は `tests/test_chunk_merge.py`（方式、コンセンサス、1〜3 次元の組み合わせ）が強制する。
方式ごとの正しさは `tests/test_sparsify.py` と `tests/test_select_threshold.py` が強制する。
