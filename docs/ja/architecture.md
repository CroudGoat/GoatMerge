# アーキテクチャと不変条件

## モジュール

| ファイル | 責務 |
|---|---|
| `merge.py` | `merge_model`、`merge_tensor`、`MergeSettings`、`ModelEntry`。テンソルごとのマージロジックと Chunk Merge 分岐 |
| `merge_method.py` | `MergeMethod` enum と `MergeKernel` 基底クラス、`build_kernel()` ディスパッチ（カーネルごとの遅延 import で循環を回避する） |
| `consensus.py` | `ConsensusAccumulator`、`ConsensusMethod`、`GtaKernel`。ストリーミング蓄積カーネルであり、`acc`、`l1`、`c`、`nz` を in-place 更新する。符号は 2 ビットパック（`_pack_signs` と `_unpack_match`）でキャッシュし、divisor を再読なしで組み立てる |
| `kernels.py` | 非 GTA の各カーネル：`LinearKernel`、`MixtureKernel`、`SlerpKernel`、`TiesKernel` |
| `sparsify.py` | スパース化カーネル（`magnitude`、`random`、`magnitude_outliers`、`della_magprune`、`bs`）。大域しきい値はストリーミング exact 選択（`_select_threshold`、O(1) RAM、ソートとビット一致）、`uint8` マスク、chunked マスクと global マスク。`bernoulli` は in-place 描画。未指定時は `_DEFAULT_SPARSE_CHUNK_ELEMENTS`（8M）で常時チャンク化する |
| `io.py` | `ShardedTensorIndex`、`ShardReader`、`TensorWriter`、`load_delta`、`load_delta_chunk`、`copy_support_files`、`dtype_from_safetensors`、`safetensors_dtype_str`、`shape_fit`、`advise_drop_caches`。シャード型 safetensors の読み書き。行読みは 1 回の `readinto` でバッファを 1 個だけ確保し、そのバッファを所有するテンソルを返す。`drop_caches=True` で読み済み範囲と書き済みシャードに `POSIX_FADV_DONTNEED` を発行する。`TensorWriter` は manifest から事前確定し到着順に追記する（ステージングなし） |
| `extract.py` | `extract_task_vector`。`T = W_source - W_base` で TV を書き出す。減算はインプレース（`load_delta(kind="model")` を再利用し、`clone=False`）で 2 S であり、`chunk_elements` 指定時は行チャンク抽出で O(S) + O(chunk) になる。結果は標準パスの出力とビット単位で一致する |
| `fingerprint.py` | ベースモデルの指紋（`compute_base_fingerprint`、`model_manifest`、`tensor_content_hash`）と検証。アンカーテンソルは行チャンクでストリーミングハッシュする（O(chunk)） |
| `metadata.py` | `goatmerge.json` メタデータエンベローブ（`build_merged_metadata`、`write_metadata`、`load_metadata`、`validate_metadata`） |
| `inspect.py` | `inspect_dir`（ディレクトリ検査）と `verify_against_base` |
| `hf.py` | `resolve_model_dir` と `is_repo_id`。ローカルパスと HF キャッシュを解決する |
| `cli.py` | CLI エントリポイント。`extract`、`merge`、`inspect` サブコマンドと YAML レシピ（`-c`）を持つ。`extract` は `--chunk-elements` を受け付け、両サブコマンドとも `--drop-caches` を受け付ける（YAML の `drop_caches` は merge のみ） |
| `__init__.py` | パブリック API の再エクスポート |

### enum 値（誤りやすい箇所）

- `MergeMethod`：`gta`、`linear`、`mixture`、`slerp`、`ties`
- `ConsensusMethod`：`none`、`sum`、`count`
- `SparsificationMethod`：`magnitude`、`random`、`magnitude_outliers`、`della_magprune`、`bs`
- `RescaleNorm`：`l1`、`l2`、`linf`
- `MergeSettings` フィールド：`merge_method`、`density`、`method`、`n`、`m`、`gamma`、`epsilon`、`rescale`、`normalize`、`lambda_`、`consensus`、`chunk_elements`、`drop_caches`
- `ModelEntry`：`(dir: Path, kind: "tv" | "model", weight: float)`

## データフロー（テンソルごと）

1. 各 delta を 1 回ずつストリームする：`load_delta(..., clone=False)` から `sparsify_delta`（コンセンサス前、in-place）を通し、bf16·bf16 重み付けを経てカーネルの `accumulate` に入れる。
2. `finish(base)` はコンセンサス恒等式 `(acc + M·l1)/2` を計算し、符号キャッシュから divisor を組み立て、`base + mixed / divisor` を求め、`lambda_` をかけ、dtype に戻す。
3. Chunk Merge は行チャンクごとに同じカーネルを回し、事前確保した `full` バッファに書き込む。
4. `TensorWriter` が完成テンソルを到着順にバイト追記する。

## 不変条件（破ってはならない）

1. **重み付き積は bf16·bf16 テンソル積**：`δ * α` は `delta.mul_(torch.tensor(alpha, dtype=delta.dtype))` と書く。
Python スカラーの `add_(alpha=...)` は参照実装（`stacked * weights`）と内部積精度が乖離し、近接要素で多数決符号が反転する。
2. **コンセンサス恒等式**：`mixed = (acc + M·l1)/2`（`M = sign(majority)`）を使う。
マスク済み和を 2 周目で取り直さず、1 周目の蓄積で済ます。
`divisor` は蓄積中にキャッシュした 2 ビット符号（`_pack_signs`、k·S/4）から組み立て、delta の再読はしない。
`normalize=False` の consensus では符号キャッシュ自体を取らない。
3. **スパース化はコンセンサス前**：各 delta へ適用し、各 delta への適用回数は正確に 1 回である（divisor 用の再適用は存在しない）。
4. **決定論的なマスク**：確率的マスク（`random` と `della_magprune`）のジェネレータは `merge_generator(key, source_index, chunk)` でシードする。
引数はテンソル名、ソース番号、チャンク番号の組であり、実行ごとの再現性を保証する。
5. **rescale の和は float32 拡張**：チャンクパスが非チャンクパスの出力をビット単位で再現できるのはこの拡張に依存する（`sparsify.py::_norm`）。
6. **Chunk Merge の結果は標準パスの出力と一致する**。
7. **行読みの返値は所有ストレージ**：`_read_rows` は `readinto` で確保したバッファを所有するテンソルを返す。
各 read は独立バッファを使うため、リード間のエイリアスはない。
mmap パスの view を in-place 変更してはならない。
`load_delta(..., clone=False)` は direct リーダーの既所有ストレージを前提とする。
merge 系の全 load 呼び出しは `clone=False` で呼ぶ。
8. **抽出の等価性**：行チャンク抽出とインプレース減算の結果は標準パス（`s.to(dtype) - b`）の出力とビット単位で一致する。
形状不一致（submatrix と mismatch）は必ず非チャンクパスにフォールバックし、`load_delta` の共有ルールで処理する。
形状判定は `shape_fit` に一元化する。
9. **`--drop-caches` は opt-in**：`POSIX_FADV_DONTNEED` で読み済み範囲と書き済みシャードをページキャッシュから外す。
各テンソルを正確に 1 回だけ読むため、再読ペナルティはない。
書き出し側は dirty ページを待たない（fdatasync しない）ため、数 GB の残余は生じ得る。
速度優先の判断である。
10. **Writer は到着順書き込み**：`TensorWriter` は manifest からシャード分割、オフセット、ヘッダを事前確定し、到着順にバイト追記する。
ステージングも `save_file` バッファも持たない。
計画外の到着とサイズ不一致は例外にする（黙って破損させない）。
未完シャードは破棄し、壊れたファイルを残さない。
ヘッダは正準ソート順で決定的である。
11. **sparsify の既定チャンク**：`chunk_elements` 未指定でも `_DEFAULT_SPARSE_CHUNK_ELEMENTS`（8M）でマスク計算を分割する。
決定的手法の出力はビット単位で一致し、確率的手法の出力は同一 seed で決定的になる。
ただし torch の CPU RNG は形状依存のため、全体一括の出力とはビット不一致になり得る。
パス間の一致は保たれる。

これらの不変条件を守るテストは `tests/test_consensus_merge.py`（パリティ）、`tests/test_chunk_merge.py`（チャンク等価性）、`tests/test_io.py::test_reader_clone_semantics`（読みの所有権）である。
