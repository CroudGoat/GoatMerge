# Streaming kernel & consensus

Per tensor, every delta is streamed exactly once per pass and accumulated
(`goatmerge/merge.py`, `goatmerge/consensus.py`):

```
acc  = Σᵢ αᵢ · δᵢ         (base dtype, in place)
l1   = Σᵢ |αᵢ · δᵢ|      (base dtype, in place; consensus only)
c    = Σᵢ sign(αᵢ·δᵢ)    (int8; count consensus only)
```

Reference semantics (mergekit GTA, per tensor):

```
weighted[i] = alpha_i * delta_i
majority    = (weighted.sum(0) >= 0)   [sum]  /  (sign.sum(0) >= 0)  [count]
mask[i]     = sign[i] == majority
mixed       = (weighted * mask).sum(0)
divisor     = (weights * mask).sum(0);  divisor[divisor == 0] = 1
result      = (base + mixed).to(base.dtype)
```

## Consensus identity

```
mixed = (acc + M · l1) / 2      M = +1 if acc ≥ 0 else −1   (sum)
                               M = +1 if c   ≥ 0 else −1   (count)
```

Elements whose sign disagrees with the majority cancel out of this identity
(zeros contribute 0 on both sides), so the masked sum never materializes a
k-sized mask. The final result is:

```
result = base + mixed / divisor        divisor = weight sum of sign-matching TVs
```

`divisor` is assembled from 2-bit-packed signs cached during accumulation
(`_pack_signs`, k·S/4 transient bytes) — no delta re-read, no second sparsify.
With `normalize=False` no sign cache is taken at all.

## The bf16·bf16 rule (invariant #1)

The weighted product `δᵢ · αᵢ` **must** be a bf16·bf16 tensor multiply:

```python
delta.mul_(torch.tensor(alpha, dtype=delta.dtype))
```

A Python-float scalar `add_(alpha=...)` has different internal product
precision and can flip the per-element majority sign at near-tie elements.
Do not "simplify" this — parity with the reference (`stacked * weights`)
depends on it. See `ConsensusAccumulator` docstring and
`tests/test_consensus_merge.py`.

## Accumulator layout

`ConsensusAccumulator` holds `acc` (+ `l1` when consensus is on, `c`/`nz` for
`count`) in the base dtype, plus the optional sign packets and their weights.
`GtaKernel` streams `accumulate(delta, alpha)` per source and `finish(base)`
once. Non-GTA kernels live in `goatmerge/kernels.py`
(`LinearKernel` / `MixtureKernel` / `SlerpKernel` / `TiesKernel`).
