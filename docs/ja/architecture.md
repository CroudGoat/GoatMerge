# アーキテクチャと不変条件

## モジュール

| ファイル | 責務 |
|---|---|
| `merge.py` | `merge_model` / `merge_tensor` / `MergeSettings` / `ModelEntry`。テンソルごとのマージロジック、Chunk Merge 分岐。 |
| `merge_method.py` | `MergeMethod` enum と `MergeKernel` 基底クラス、`build_kernel()` ディスパッチ（カーネルごとの遅延 import で循環を回避）。 |
| `consensus.py` | `ConsensusAccumulator` / `ConsensusMethod` / `GtaKernel`。ストリーミング蓄積カーネル、`acc`/`l1`/`c`/`nz` を in-place 更新。符号は 2 ビットパック（`_pack_signs` / `_unpack_match`）でキャッシュし divisor を再読なしで組み立てる。 |
| `kernels.py` | 非 GTA の各カーネル：`LinearKernel` / `MixtureKernel` / `SlerpKernel` / `TiesKernel`。 |
| `sparsify.py` | スパース化カーネル（`magnitude`、`random`、`magnitude_outliers`、`della_magprune`、`bs`）。大域しきい値はストリーミング exact 選択（`_select_threshold`、O(1) RAM・ソートとビット一致）、`uint8` マスク、chunked / global マスク。`bernoulli` は in-place 描画。未指定時は `_DEFAULT_SPARSE_CHUNK_ELEMENTS`（8M）で常時チャンク化。 |
| `io.py` | `ShardedTensorIndex` / `ShardReader` / `TensorWriter` / `load_delta` / `load_delta_chunk` / `copy_support_files` / `dtype_from_safetensors` / `safetensors_dtype_str` / `shape_fit` / `advise_drop_caches`。シャード型 safetensors の読み書き。行読みは 1 回の `readinto` でバッファを 1 つだけ確保。そのバッファを所有するテンソルを返す。`drop_caches=True` で読み済み範囲・書き済みシャードに `POSIX_FADV_DONTNEED`。`TensorWriter` は manifest から事前確定し到着順に追記（ステージングなし）。 |
| `extract.py` | `extract_task_vector` — `T = W_source - W_base` で TV を書き出す。減算はインプレース（`load_delta(kind="model")` を再利用、`clone=False`）で 2 S、`chunk_elements` 指定時は行チャンク抽出で O(S) + O(chunk)。結果は標準パスとビット一致。 |
| `fingerprint.py` | ベースモデルの指紋（`compute_base_fingerprint` / `model_manifest` / `tensor_content_hash`）と検証。アンカーテンソルは行チャンクでストリーミングハッシュ（O(chunk)）。 |
| `metadata.py` | `goatmerge.json` メタデータエンベローブ（`build_merged_metadata` / `write_metadata` / `load_metadata` / `validate_metadata`）。 |
| `inspect.py` | `inspect_dir`（ディレクトリ検査）と `verify_against_base`。 |
| `hf.py` | `resolve_model_dir` / `is_repo_id` — ローカルパスと HF キャッシュの解決。 |
| `cli.py` | CLI エントリポイント。`extract` / `merge` / `inspect` サブコマンド、YAML レシピ（`-c`）。`extract` は `--chunk-elements` を受け付ける。両サブコマンドとも `--drop-caches`（YAML：`drop_caches`、merge のみ）を受け付ける。 |
| `__init__.py` | パブリック API の再エクスポート。 |

### enum 値（シグネチャのブレやすい箇所）

- `MergeMethod`：`gta` | `linear` | `mixture` | `slerp` | `ties`
- `ConsensusMethod`：`none` | `sum` | `count`
- `SparsificationMethod`：`magnitude` | `random` | `magnitude_outliers` | `della_magprune` | `bs`
- `RescaleNorm`：`l1` | `l2` | `linf`
- `MergeSettings` フィールド：`merge_method, density, method, n, m, gamma, epsilon, rescale, normalize, lambda_, consensus, chunk_elements, drop_caches`
- `ModelEntry`：`(dir: Path, kind: "tv" | "model", weight: float)`

## データフロー（テンソルごと）

1. 各 delta を 1 回ずつストリーム：`load_delta(..., clone=False)` → `sparsify_delta`
  （コンセンサス前、in-place）→ bf16·bf16 重み付け → カーネルの `accumulate`。
2. `finish(base)`：コンセンサス恒等式 `(acc + M·l1)/2`、符号キャッシュから divisor、
   `base + mixed / divisor`、`lambda_` スケール、dtype 戻し。
3. Chunk Merge：行チャンクごとに同じカーネルを回し、事前確保した `full` バッファへ。
4. `TensorWriter` が完成テンソルを到着順にバイト追記する。

## 不変条件（絶対に壊しちゃダメ）

1. **重み付き積は bf16·bf16 テンソル積。** `δ * α` は
   `delta.mul_(torch.tensor(alpha, dtype=delta.dtype))`。Python スカラーの
   `add_(alpha=...)` は参照実装（`stacked * weights`）と内部積精度が乖離し、
   近接要素で多数決符号が反転する。
2. **コンセンサス恒等式** `mixed = (acc + M·l1)/2`（`M = sign(majority)`）。
   マスク済み和を 2 周目で取り直さず、1 周目の蓄積で済ます。`divisor` は
   蓄積中にキャッシュした 2 ビット符号（`_pack_signs`、k·S/4）から組み立て、
   delta の再読はしない。`normalize=False` の consensus では符号キャッシュ
   自体を取らない。
3. **スパース化はコンセンサス前**に各 delta へ適用。各 delta は正確に
   1 回だけ sparsify される（divisor 用の再適用は存在しない）。
4. **決定論的なマスク** — 確率的マスク（`random` / `della_magprune`）の
   ジェネレータは `merge_generator(key, source_index, chunk)`、すなわち
   `（テンソル名、ソース番号、チャンク番号）` で Seeding。実行ごとの再現性を保証する。
5. **rescale の和は float32 拡張** — チャンクパスが非チャンクパスを
   ビット一致で再現できるのはこれに依存（`sparsify.py::_norm`）。
6. **Chunk Merge の結果は標準パスと一致する。**
7. **行読みの返値は所有ストレージ** — `_read_rows` は `readinto` で確保した
   バッファを所有するテンソルを返す。各 read は独立バッファなので
   リード間のエイリアスはない。mmap パスの view を in-place 変更しちゃダメ。
   `load_delta(..., clone=False)` は direct リーダーの既所有ストレージが前提。
   merge 系の全 load 呼び出しは `clone=False`。
8. **抽出の等価性** — 行チャンク抽出・インプレース減算の結果は
   標準パス（`s.to(dtype) - b`）と**ビット一致**する。
   形状不一致（submatrix / mismatch）は必ず非チャンクパスにフォールバックし、
   `load_delta` の共有ルールで処理する。形状判定は `shape_fit` に一元化する。
9. **`--drop-caches` は opt-in** — `POSIX_FADV_DONTNEED` で読み済み範囲・
   書き済みシャードをページキャッシュから外す。各テンソルは正確に
   1 回しか読まないので再読ペナルティはない。書き出し側は dirty ページを待たない
   （fdatasync しない）ので数 GB の残余はあり得る。速度優先の判断だ。
10. **Writer は到着順書き込み** — `TensorWriter` は manifest からシャード分割・
    オフセット・ヘッダを事前確定し、到着順にバイト追記する。ステージングも
    `save_file` バッファも持たない。計画外の到着・サイズ不一致は例外
    （黙って壊さない）。未完シャードは破棄する。ヘッダは正準ソート順で決定的。
11. **sparsify の既定チャンク** — `chunk_elements` 未指定でも
    `_DEFAULT_SPARSE_CHUNK_ELEMENTS`（8M）でマスク計算は分割される。
    決定的手法はビット一致、確率的手法は同一 seed で決定的（torch の CPU RNG は
    形状依存のため全体一括とはビット不一致になり得るけど、パス間の一致は保たれる）。

これらを守ってるテスト：`tests/test_consensus_merge.py`（パリティ）、
`tests/test_chunk_merge.py`（チャンク等価性）、
`tests/test_io.py::test_reader_clone_semantics`（読みの所有権）。
