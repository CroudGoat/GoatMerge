# ストリーミングカーネルとコンセンサス

1 テンソルあたり、各 delta を 1 パスでちょうど 1 回ずつストリームして
次のように蓄積する（`goatmerge/merge.py`、`goatmerge/consensus.py`）。

```
acc  = Σᵢ αᵢ · δᵢ         （base dtype、in place）
l1   = Σᵢ |αᵢ · δᵢ|      （base dtype、in place、consensus 時のみ）
c    = Σᵢ sign(αᵢ·δᵢ)    （int8、count 方式のみ）
```

参照動作（mergekit GTA、テンソルごと）：

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

多数派の符号と一致しない要素はこの式で自動的に 0 になる（0 は両辺とも 0 に寄与）。
だから全 delta のマスク和を 2 回の in-place 演算で表現できて、
k サイズのマスクを物資化しない。最終結果はこうなる：

```
result = base + mixed / divisor        divisor = 符号一致 TV の重み和
```

`divisor` は 1 要素ずつ求める必要があるけど、蓄積中にキャッシュした 2 ビットパック符号
（`_pack_signs`、k·S/4）から組み立てる。delta の再読も、sparsify の再適用もない。
`normalize=False` なら符号キャッシュ自体を取らない。

## bf16·bf16 ルール（不変条件 #1）

重み付き積 `δᵢ · αᵢ` は **bf16・bf16 のテンソル積**として計算すること：

```python
delta.mul_(torch.tensor(alpha, dtype=delta.dtype))
```

Python の float スカラーを渡す `add_(alpha=...)` では内部積の精度が異なり、
近接した要素で多数決の符号が反転することがある。この「単純化」はやっちゃダメ。
参照実装（`stacked · weights`）との一致がかかってる。
`ConsensusAccumulator` の docstring と `tests/test_consensus_merge.py` 参照。

## 蓄積領域の内訳

`ConsensusAccumulator` は base dtype で `acc`（consensus 時は `l1` も、
`count` なら `c` / `nz` も）を持ち、必要なら符号パケットと重みも持つ。
`GtaKernel` はソースごとに `accumulate(delta, alpha)` を流し、最後に
`finish(base)` を 1 回呼ぶ。非 GTA カーネルは `goatmerge/kernels.py`
（`LinearKernel` / `MixtureKernel` / `SlerpKernel` / `TiesKernel`）にある。
