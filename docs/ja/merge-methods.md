# マージ方式

利用者は `--merge-method` で方式を選ぶ（`MergeMethod`：`gta`、`linear`、`mixture`、`slerp`、`ties`）。
`merge_method.build_kernel()` がカーネルごとの遅延 import で分岐するため、循環 import は生じない。

| `merge_method` | 動作 | 用途 |
|---|---|---|
| `gta` | mergekit GTA と同じタスク算術（consensus と sparsify に対応する） | 標準。mergekit と数値パリティを取る |
| `linear` | `base + Σ wᵢ·δᵢ` | 単純な重み付き和 |
| `mixture` | `base + Σ wᵢ·δᵢ / Σ wᵢ` | 重み付き平均 |
| `slerp` | `base` と `base + Σ wᵢ·δᵢ` の球面線形補間 | 2 モデル間の補間 |
| `ties` | 符号が多数決と一致するソースだけで平均化する（TIES 流） | 符号の一致で剪定する場合 |

数値パリティを取るのは `gta` だけであり、対象は mergekit の同名メソッドである。
`linear`、`mixture`、`slerp`、`ties` は GoatMerge 独自の実装であり、mergekit の同名メソッドとは定義が異なる。
たとえば GoatMerge の `slerp` は `t = 1` で `base + Σ wᵢ·δᵢ` を返す。

## コンセンサス（`--consensus`）

`ConsensusMethod` は `none`、`sum`、`count` の 3 値であり、カーネル（通常は `gta`）の上に載る。

- `none`：符号フィルタを使わない。
- `sum`：多数派は重み付き和の符号である（`acc >= 0`）。
- `count`：多数派は要素ごとの投票数の符号である（`c >= 0`）。

コンセンサスは `mixed = (acc + M·l1)/2`（`M = ±1` は多数派の符号）を計算し、`result = base + mixed / divisor` を返す。
`divisor` は符号一致 TV の重み和であり、要素ごとに求める（`divisor[divisor == 0] = 1`）。
機構の詳細は [streaming-kernel.md](streaming-kernel.md) を参照する。

## スケールと正規化

- `--lambda`（`MergeSettings.lambda_`）：混合済みの delta にかける倍率である。
- `--no-normalize`：除数による正規化を無効にする（`normalize=False`）。
この場合 GoatMerge は符号キャッシュ自体を取らない。
- `--no-rescale`：スパース化後のノーム再正規化を無効にする（[sparsification.md](sparsification.md)）。

## 方式ごとの注意

- `slerp` は全域ノームに依存するため、`--chunk-elements` 指定時は警告とともに非チャンクパスにフォールバックする。
- `ties` と `gta` + consensus はどちらも符号一致で剪定するが、平均化の分母が異なる。
定義の差分は `goatmerge/kernels.py` の `TiesKernel` と `goatmerge/consensus.py` の `GtaKernel` を見比べるとわかる。
- `MergeKernel` を追加する開発者は、`kernels.py` か `consensus.py` にクラスを置き、`MergeMethod` と `build_kernel()` を拡張し、`goatmerge/__init__.py` の `__all__` に追加する。
