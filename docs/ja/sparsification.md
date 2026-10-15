# スパース化

`--method` で指定する。`--density 1.0` では無効になる。スパース化は
**コンセンサスより前**、各 delta に対して適用する。各 delta は正確に 1 回だけ
sparsify される（[architecture.md](architecture.md) の不変条件 #3）。

| `method` | 動作 | 関連フラグ |
|---|---|---|
| `magnitude` | 絶対値が大きい上位 density の割合を保持 | — |
| `magnitude_outliers` | 上位 gamma と下位を落とし、中間を保持 | `--gamma` |
| `random` | 要素ごとに確率 density で保持（DARE 風） | — |
| `della_magprune` | 行内の絶対値ランクに応じて保持確率を変える | `--epsilon` |
| `bs` | m 要素のブロックごとに上位 n 個を保持 | `--n`、`--m` |

実装（`goatmerge/sparsify.py`）：

- `magnitude` / `magnitude_outliers` は全件 `argsort` じゃなく `torch.topk`
 （インデックス k 個だけ）を使う。大域しきい値はストリーミング exact 選択
  （`_select_threshold`、O(1) RAM、ソートとビット一致）で求める。
- マスクは全部 `uint8`（1 要素 1 バイト）。適用と rescale は in-place。
- `bernoulli` は in-place 描画（`bernoulli_`。`full_like` 確率テンソルは作らない）。
- rescale の和は float32 に広げてから足す（`_norm`）。チャンクパスが非チャンクパスを
  ビット一致で再現できるのはこれのおかげ。
- マスク計算は `chunk_elements` 未指定でも内部バジェット
  `_DEFAULT_SPARSE_CHUNK_ELEMENTS`（8M 要素）で**常時チャンク化**される。
  決定的手法はビット一致、確率的手法は同一 seed で決定的（torch の CPU RNG は
  形状依存なので、全体一括とはビット不一致になり得るけど、パス間の一致は保たれる）。

## 決定論的マスク

乱数を使う方式（`random` / `della_magprune`）は「テンソル名 × ソース × チャンク」から
決定論的にシードされる（`sparsify.merge_generator(key, source_index, chunk)`）。
同じ入力からは常に同じ結果になるし、同じソースへの繰り返しパスでもマスクが一致する。

## 大域マスクとチャンクマスク

- 大域的手法（`magnitude` / `magnitude_outliers`）はテンソル全体を 1 回スキャンして
  しきい値とタイ数を決め（O(1) メモリ）、各チャンクに同じマスクを適用する
  （`GlobalMasker(mask_params)`）。両パスで同じマスクを使う。
- チャンク等価性は `tests/test_chunk_merge.py`（方式 × コンセンサス × 1〜3 次元）、
  方式ごとの正しさは `tests/test_sparsify.py` と `tests/test_select_threshold.py` で
  強制してる。
