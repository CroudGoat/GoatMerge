# トラブルシューティングと FAQ

## エラー

### `error: --base (or 'base' in config) is required`（終了コード 2）

`base` は CLI → YAML の順に解決する。`--base` を渡すか、レシピに `base:` を書くこと。
`--out` / `out:` も同じ。

### `error: no task vectors or models given`（終了コード 2）

`--tv` / `--model` を最低 1 件渡すか、YAML に `tv:` / `model:` を書くこと。
注意：CLI で `--tv` を渡すと YAML 側リストより優先される（マージはされない）。

### `error: base fingerprint mismatch for: ...`（終了コード 3）

別のベースで抽出した TV を混ぜようとしてる。正しい `--base` か確認するか、
このベースで TV を抽出し直すこと。分かってて外すときだけ
`--skip-fingerprint-check`（`skip_fingerprint_check: true`）を使う。
[fingerprint.md](fingerprint.md) 参照。

### `bad entry '...' (expected DIR:WEIGHT)`

エントリは**最後の**コロンで分割する。`--tv /path/to/tv` みたいな重みなし指定は
ダメで、`--tv /path/to/tv:0.7` と書くこと。パス自体にコロンが入ってても、
最後のコロンだけが重みの区切りになるので大丈夫。

### ライタのエラー（計画外の到着 / サイズ不一致）

`TensorWriter` は manifest から分割を事前確定してて、計画外は黙って壊す代わりに
例外にする。だいたい manifest（名前・dtype・形状）と流し込まれるテンソルが
合ってないのが原因。握りつぶさずに入力セットを直すこと。

## 挙動の疑問

### mergekit とわずかに結果が違うけど大丈夫？

bf16 で rtol=2e-2、atol=1e-2 の範囲（観測 max|d| = 0.0625）ならパリティ予算内だ。
それより大きいズレや符号反転は異常。bf16·bf16 の重み付き積を Python スカラーの
`add_(alpha=...)` に「単純化」してないか疑うこと（不変条件 #1）。

### Chunk Merge と通常パスで結果が違う気がするけどバグ？

うん、バグだ。両者は一致しなきゃいけない（`slerp` だけは意図的にフォールバックして
警告を出す）。テンソル形状・方式・consensus・スパース化設定・`chunk_elements` を
添えて報告して。[chunk-merge.md](chunk-merge.md) 参照。

### 確率的スパース化が毎回違う結果になるんだけど？

同じ入力なら同じになるはず。マスクは（テンソル名、ソース番号、チャンク番号）から
シードされてる。テンソル名を変えたりソースの順番を変えたりしたら、seed —
ひいてはマスク — が正当に変わる。

### デカいテンソルで OOM するんだけど？

`--chunk-elements`（例：`33554432`）を付けて、N 要素超えを O(S) + O(chunk) パスに
回すこと。`slerp` だけはフルパスに落ちる。[memory.md](memory.md) 参照。

### プロセス RSS は小さいのにシステムメモリが膨らむんだけど？

それはリークじゃなくページキャッシュだ。一括処理なら `--drop-caches`
（merge は `drop_caches: true`）を使うこと。[io.md](io.md) 参照。

## 寄与するときのチェックリスト

1. 既存の enum・命名・`from __future__ import annotations` スタイルに合わせる。
2. 公開 API は `__init__.py` の `__all__` に追加する。
3. docstring は英語で（理由・注意点を書き、自明な説明は書かない）。
4. フルテストを実行する：97 passed が基準。
5. `test_consensus_merge.py` のパターンで mergekit パリティを確認する。
6. [architecture.md](architecture.md) の不変条件を読み返す。マージ系バグの大半は
   不変条件 #1 / #2 / #3 の違反だ。
