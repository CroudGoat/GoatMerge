# トラブルシューティングと FAQ

## エラー

### `--base` 不足（終了コード 2）

`error: --base (or 'base' in config) is required` が出る場合、`base` の解決に失敗している。
`base` は CLI から YAML の順に解決する。
利用者は `--base` を渡すか、レシピに `base:` を書く。
`--out` と `out:` も同じ規則である。

### マージ対象の不足（終了コード 2）

`error: no task vectors or models given` が出る場合、マージ対象が空である。
利用者は `--tv` と `--model` を合わせて最低 1 件渡すか、YAML に `tv:` か `model:` を書く。
CLI で `--tv` を渡すと YAML 側リストより優先され、両者はマージされない。

### ベース指紋のミスマッチ（終了コード 3）

`error: base fingerprint mismatch for: ...` が出る場合、別のベースで抽出した TV を混ぜようとしている。
利用者は正しい `--base` か確認するか、このベースで TV を抽出し直す。
照合を外すのは意図的な場合に限り、`--skip-fingerprint-check`（`skip_fingerprint_check: true`）を付ける。
[fingerprint.md](fingerprint.md) を参照する。

### エントリ書式の誤り

`bad entry '...' (expected DIR:WEIGHT)` が出る場合、エントリ書式が誤っている。
エントリは最後のコロンで分割する。
`--tv /path/to/tv` のような重みなし指定は誤りであり、`--tv /path/to/tv:0.7` と書く。
パス自体にコロンが入っていても、最後のコロンだけが重みの区切りになる。

### ライタのエラー（計画外の到着とサイズ不一致）

`TensorWriter` は manifest から分割を事前確定し、計画外の到着とサイズ不一致は黙って破損させる代わりに例外にする。
原因の大半は manifest（名前、dtype、形状）と流し込まれるテンソルの不一致である。
例外を握りつぶさず、入力セットを直す。

## 挙動の疑問

### mergekit との数値差

bf16 で rtol=2e-2、atol=1e-2 の範囲（観測 max|d| = 0.0625）ならパリティ予算内である。
範囲を超えるズレや符号反転は異常であり、bf16·bf16 の重み付き積を Python スカラーの `add_(alpha=...)` に書き換えていないか疑う（不変条件 #1）。

### Chunk Merge と通常パスの不一致

両者の出力は一致しなくてはならない。
不一致はバグであり、`slerp` だけが意図的にフォールバックして警告を出す。
報告にはテンソル形状、方式、consensus、スパース化設定、`chunk_elements` を添える（[chunk-merge.md](chunk-merge.md)）。

### 確率的スパース化の再現性

同じ入力からは同じ結果が出る。
マスクはテンソル名、ソース番号、チャンク番号の組からシードするため、テンソル名の変更やソース順序の変更はマスクを正当に変える。

### 大テンソルの OOM

`--chunk-elements`（例：`33554432`）を付けて、N 要素超えを O(S) + O(chunk) パスに回す。
`slerp` だけはフルパスに落ちる（[memory.md](memory.md)）。

### プロセス RSS が小さいのにシステムメモリが膨らむ

原因はリークではなくページキャッシュである。
一括処理では `--drop-caches` を使う（merge は `drop_caches: true` も使える）。
[io.md](io.md) を参照する。

## 寄与するときのチェックリスト

1. 既存の enum、命名、`from __future__ import annotations` スタイルに合わせる。
2. 公開 API は `__init__.py` の `__all__` に追加する。
3. docstring は英語で書く（理由と注意点を書き、自明な説明は書かない）。
4. フルテストを実行する。97 passed が基準である。
5. `test_consensus_merge.py` のパターンで mergekit パリティを確認する。
6. [architecture.md](architecture.md) の不変条件を読み返す。
マージ系バグの大半は不変条件 #1、#2、#3 の違反である。
