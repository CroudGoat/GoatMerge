# テストとベンチマーク

インタープリタは `/home/CloudGoat/venvs/mergekit/bin/python`（mergekit も同一 venv）。
依存は Python ≥ 3.10、`torch>=2.1`、`safetensors>=0.4`、`pyyaml>=6.0`。

```bash
/home/CloudGoat/venvs/mergekit/bin/python -m pytest            # 97 passed が基準
/home/CloudGoat/venvs/mergekit/bin/python -m goatmerge.cli merge -c examples/merge_recipe.yaml
/home/CloudGoat/venvs/mergekit/bin/python -m goatmerge.cli extract --base B --source S --out TV
/home/CloudGoat/venvs/mergekit/bin/python -m goatmerge.cli extract --base B --source S --out TV --chunk-elements 33554432
/home/CloudGoat/venvs/mergekit/bin/python -m goatmerge.cli inspect --dir MODEL
```

## テスト

| ファイル | 内容 |
|---|---|
| `tests/test_consensus_merge.py` | mergekit GTA との数値パリティ、CLI e2e、指紋ゲート、extract ラウンドトリップ、チャンク抽出の等価性・フォールバック |
| `tests/test_chunk_merge.py` | Chunk Merge と標準パスの等価性、スパース化の正当性、consensus 併用 |
| `tests/test_kernels.py` | linear / mixture / slerp / ties の各カーネル |
| `tests/test_sparsify.py` | スパース化各方式、既定チャンク予算の等価性、bernoulli の決定性 |
| `tests/test_select_threshold.py` | `_select_threshold` がソートと完全一致すること（f32/bf16/ties/全同値、factory 一致） |
| `tests/test_fingerprint.py` | 指紋の決定性・感度 |
| `tests/test_io.py` | シャード型 / 単一シャード / 行スライス読み、ライタの命名と分割、drop-caches の advise 発行 |
| `tests/test_metadata.py` | `goatmerge.json` エンベローブ |
| `tests/measure_peak_ram.py` | ピーク RAM 計測（pytest の `testpaths` 外。子プロセス隔離で両パスの peak/S を報告） |

## パリティの確認方法

マージロジックを変えたら、必ず mergekit 参照実装とのパリティを確認すること。
`tests/test_consensus_merge.py` のパターンを使い回すといい：
bf16 テンソルで rtol=2e-2、atol=1e-2。肝は bf16·bf16 の重み付き積と
`(acc + M·l1)/2` 恒等式だ。[streaming-kernel.md](streaming-kernel.md) と
[architecture.md](architecture.md) の不変条件 #1 / #2 参照。

## ベンチマーク

- `benchmark.py` / `bench_real_models.py` / `tests/measure_peak_ram.py`
- `Benchmarks/benchmark_compare.py`、`Benchmarks/benchmark_task_arithmetic.py`、
  `Benchmarks/RESULTS.md` — GoatMerge vs mergekit GTA の方法と結果。
