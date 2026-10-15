# 指紋（フィンガープリント）

TV を抽出したベースモデルの指紋（テンソル名・形状・dtype・アンカーテンソルの
ハッシュ）を `goatmerge.json` に保存する。マージ時に現在のベースの指紋と
照合し、不一致ならエラー終了する。「別のベースで抽出した TV を
うっかり混ぜてしまう」事故を防ぐためだ。

モジュールは `goatmerge/fingerprint.py`：`compute_base_fingerprint` /
`model_manifest` / `tensor_content_hash`。

- アンカーは行チャンクでストリーミングハッシュする（O(chunk)）。
- `merge` は各 TV の保存済み指紋と今のベースを比べ、不一致なら拒否する
  （終了コード 3）。分かってて外すときだけ `--skip-fingerprint-check`
  （CLI）/ `skip_fingerprint_check`（YAML）を使うこと。
- `verify_against_base` が `cli.cmd_merge` のゲート本体だ。

`tests/test_fingerprint.py`（決定性・感度）と `tests/test_consensus_merge.py` の
指紋ゲート / extract ラウンドトリップテストがカバーしてる。
