# マージ方式

`--merge-method` で選ぶ（`MergeMethod`: `gta` | `linear` | `mixture` |
`slerp` | `ties`。`merge_method.build_kernel()` が遅延 import で分岐し、
循環 import を避けてる）。

| `merge_method` | 動作 | 用途 |
|---|---|---|
| `gta` | mergekit GTA と同じタスク算術（consensus / sparsify 対応） | 標準。mergekit と数値パリティ |
| `linear` | `base + Σ wᵢ·δᵢ` | 単純な重み付き和 |
| `mixture` | `base + Σ wᵢ·δᵢ / Σ wᵢ` | 重み付き平均 |
| `slerp` | `base` と `base + Σ wᵢ·δᵢ` の球面線形補間 | 2 モデル間の補間 |
| `ties` | 符号が多数決と一致するソースで平均化（TIES 風） | 符号の一致で剪定したい場合 |

> `gta` のみ mergekit の同名メソッドと数値パリティを取る。`linear` /
> `mixture` / `slerp` / `ties` は GoatMerge 独自の実装で、mergekit の同名
> メソッドとは定義が異なる（たとえば GoatMerge の `slerp` は `t = 1` で
> `base + Σ wᵢ·δᵢ` を返す）。

## コンセンサス（`--consensus`）

`ConsensusMethod`: `none` | `sum` | `count`。カーネル（だいたい `gta`）の上に載る。

- `none` — 符号フィルタなし。
- `sum` — 多数派 = 重み付き和の符号（`acc >= 0`）。
- `count` — 多数派 = 要素ごとの投票数の符号（`c >= 0`）。

仕組みとしては `mixed = (acc + M·l1)/2`（`M = ±1` は多数派の符号）を計算し、
`result = base + mixed / divisor` になる。`divisor` は符号一致 TV の重み和で
要素ごとに求める（`divisor[divisor == 0] = 1`）。詳しくは
[streaming-kernel.md](streaming-kernel.md)。

## スケールと正規化

- `--lambda`（`MergeSettings.lambda_`）：ミックス済み delta にかける倍率。
- `--no-normalize`：除数による正規化をオフ（`normalize=False`。このとき符号
  キャッシュ自体を取らない）。
- `--no-rescale`：スパース化後のノーム再正規化をオフ
  （[sparsification.md](sparsification.md) 参照）。

## 方式ごとの注意

- `slerp` は全域ノームに依存するので、`--chunk-elements` 指定時は警告とともに
  非チャンクパスにフォールバックする。
- `ties` と `gta` + consensus はどっちも符号一致で剪定するけど、平均化の分母が
  違う。`goatmerge/kernels.py` の `TiesKernel` と `goatmerge/consensus.py` の
  `GtaKernel` を見比べるといい。
- `MergeKernel` を足すときは `kernels.py` か `consensus.py` にクラスを置いて、
  `MergeMethod` と `build_kernel()` を拡張し、`goatmerge/__init__.py` の
  `__all__` に追加すること。
