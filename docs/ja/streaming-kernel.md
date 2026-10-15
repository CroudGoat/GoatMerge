# ストリーミングカーネルとコンセンサス

GoatMerge は 1 テンソルあたり各 delta を 1 パスで正確に 1 回ずつストリームし、次のように蓄積する（実装は `goatmerge/merge.py` と `goatmerge/consensus.py`）。

```
acc  = Σᵢ αᵢ · δᵢ         （base dtype、in place）
l1   = Σᵢ |αᵢ · δᵢ|      （base dtype、in place、consensus 時のみ）
c    = Σᵢ sign(αᵢ·δᵢ)    （int8、count 方式のみ）
```

参照動作（mergekit GTA、テンソルごと）は次のとおりである。

```
weighted[i] = alpha_i * delta_i
majority    = (weighted.sum(0) >= 0)   [sum] / (sign.sum(0) >= 0) [count]
mask[i]     = sign[i] == majority
mixed       = (weighted * mask).sum(0)
divisor     = (weights * mask).sum(0);  divisor[divisor == 0] = 1
result      = (base + mixed).to(base.dtype)
```

## コンセンサス恒等式

```
mixed = (acc + M · l1) / 2      M = sign(acc)  （sum 方式）
                               M = sign(c)    （count 方式）
```

多数派の符号と一致しない要素はこの式で自動的に 0 になる（0 は両辺とも 0 に寄与する）。
このため全 delta のマスク和を 2 回の in-place 演算で表現でき、k サイズのマスクをメモリ上に展開しない。
最終結果は次のとおりである。

```
result = base + mixed / divisor        divisor = 符号一致 TV の重み和
```

`divisor` は要素ごとに求める必要があるが、蓄積中にキャッシュした 2 ビットパック符号（`_pack_signs`、k·S/4）から組み立てる。
delta の再読も sparsify の再適用もない。
`normalize=False` の場合 GoatMerge は符号キャッシュ自体を取らない。

## bf16·bf16 ルール（不変条件 #1）

重み付き積 `δᵢ · αᵢ` は bf16 同士のテンソル積として計算する。

```python
delta.mul_(torch.tensor(alpha, dtype=delta.dtype))
```

Python の float スカラーを渡す `add_(alpha=...)` では内部積の精度が異なり、近接した要素で多数決の符号が反転する場合がある。
この書き換えは禁止する。
参照実装（`stacked · weights`）との内部積精度の一致が不変条件 #1 であり、微小なオーバーヘッドはパリティの対価である。
詳細は `ConsensusAccumulator` の docstring と `tests/test_consensus_merge.py` を参照する。

## 蓄積領域の内訳

`ConsensusAccumulator` は base dtype で `acc` を持つ。
consensus 時は `l1` も持ち、`count` では `c` と `nz` も持つ。
必要に応じて符号パケットと重みも持つ。
`GtaKernel` はソースごとに `accumulate(delta, alpha)` を流し、最後に `finish(base)` を 1 回呼ぶ。
非 GTA カーネルは `goatmerge/kernels.py` にある（`LinearKernel`、`MixtureKernel`、`SlerpKernel`、`TiesKernel`）。
