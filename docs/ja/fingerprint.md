# 指紋

**ベース指紋**：TV を抽出したベースモデルを識別する記録。
テンソル名、形状、dtype、アンカーテンソルのハッシュから成る。
以後「指紋」と表記する。
GoatMerge は指紋を `goatmerge.json` に保存する。
マージ時に現在のベースの指紋と照合し、不一致ならエラー終了する。
この照合により、別のベースで抽出した TV の混入を防ぐ。

モジュールは `goatmerge/fingerprint.py` であり、公開要素は `compute_base_fingerprint`、`model_manifest`、`tensor_content_hash` である。

- アンカーは行チャンクでストリーミングハッシュする（O(chunk)）。
- `merge` は各 TV の保存済み指紋と現在のベースを比べ、不一致なら拒否する（終了コード 3）。
照合を外すのは意図的な場合に限り、`--skip-fingerprint-check`（CLI）か `skip_fingerprint_check`（YAML）を付ける。
- `verify_against_base` が `cli.cmd_merge` のゲート本体である。

`tests/test_fingerprint.py`（決定性と感度）と `tests/test_consensus_merge.py` の指紋ゲートおよび extract ラウンドトリップテストがこの層をカバーする。
