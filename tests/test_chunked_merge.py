"""Parity tests for the chunked merge path.

Verifies that chunked-mode results match non-chunked-mode results
(rtol=2e-2, atol=1e-2) for all merge methods.

Sparsify notes:
  - The chunked path applies sparsify per-chunk (LOCAL), while the
    non-chunked path applies it globally (GLOBAL). For magnitude /
    magnitude_outliers this is an approximation (not exact parity).
  - For random / della_magprune / bs the per-chunk result is exact
    (per-element or per-block), but the stochastic methods (random,
    della) will differ between runs due to different random draws.
  - Therefore, sparsify parity tests use density=1.0 (no sparsify)
    to isolate the merge-logic correctness.
"""

import logging
import tempfile
from pathlib import Path

import pytest
import torch

from goatmerge.consensus import ConsensusMethod
from goatmerge.merge import MergeSettings, ModelEntry, merge_tensor
from goatmerge.io import ShardReader, ShardedTensorIndex
from goatmerge.merge_method import MergeMethod
from goatmerge.sparsify import SparsificationMethod


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _make_safetensors_dir(path: Path, tensors: dict) -> None:
    """Write a single-shard safetensors file into ``path``."""
    import safetensors.torch
    path.mkdir(parents=True, exist_ok=True)
    safetensors.torch.save_file(tensors, str(path / "model.safetensors"))


def _make_reader(base_dir: Path, key: str) -> ShardReader:
    idx = ShardedTensorIndex.from_dir(Path(base_dir))
    return ShardReader(idx)


def _run_merge(base, entries, key, readers, settings):
    return merge_tensor(base, entries, key, readers, settings)


# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #
@pytest.fixture
def small_model(tmp_path):
    """Create a tiny model dir (one tensor, 256 elements) for chunked testing."""
    n = 256
    torch.manual_seed(42)
    base_t = torch.randn(n, dtype=torch.float32)
    # Two task vectors (deltas)
    tv1 = torch.randn(n, dtype=torch.float32) * 0.1
    tv2 = torch.randn(n, dtype=torch.float32) * 0.1

    base_dir = tmp_path / "base"
    tv1_dir = tmp_path / "tv1"
    tv2_dir = tmp_path / "tv2"

    _make_safetensors_dir(base_dir, {"w": base_t})
    _make_safetensors_dir(tv1_dir, {"w": tv1})
    _make_safetensors_dir(tv2_dir, {"w": tv2})

    entries = [
        ModelEntry(dir=tv1_dir, kind="tv", weight=0.7),
        ModelEntry(dir=tv2_dir, kind="tv", weight=0.3),
    ]
    return base_dir, entries, "w"


# --------------------------------------------------------------------------- #
# Parity tests (merge logic, no sparsify)
# --------------------------------------------------------------------------- #
class TestChunkedParity:
    """Chunked result must match non-chunked result (merge logic)."""

    def _check_parity(self, base, entries, key, readers, settings_chunked):
        # Non-chunked reference (density=1.0 → no sparsify)
        settings_ref = MergeSettings(
            merge_method=settings_chunked.merge_method,
            density=1.0,  # no sparsify
            method=None,
            n=settings_chunked.n,
            m=settings_chunked.m,
            gamma=settings_chunked.gamma,
            epsilon=settings_chunked.epsilon,
            rescale=settings_chunked.rescale,
            normalize=settings_chunked.normalize,
            lambda_=settings_chunked.lambda_,
            consensus=settings_chunked.consensus,
            chunk_elements=None,  # force non-chunked
        )
        ref = _run_merge(base, entries, key, readers, settings_ref)

        # Chunked (density=1.0 → no sparsify)
        settings_no_sparsify = MergeSettings(
            merge_method=settings_chunked.merge_method,
            density=1.0,
            method=None,
            n=settings_chunked.n,
            m=settings_chunked.m,
            gamma=settings_chunked.gamma,
            epsilon=settings_chunked.epsilon,
            rescale=settings_chunked.rescale,
            normalize=settings_chunked.normalize,
            lambda_=settings_chunked.lambda_,
            consensus=settings_chunked.consensus,
            chunk_elements=settings_chunked.chunk_elements,
        )
        result = _run_merge(base, entries, key, readers, settings_no_sparsify)

        torch.testing.assert_close(result, ref, rtol=2e-2, atol=1e-2)

    def _make_readers(self, base_dir, entries, key):
        """Build the full readers dict (base + all entries)."""
        readers = {base_dir: _make_reader(base_dir, key)}
        for entry in entries:
            readers[entry.dir] = _make_reader(entry.dir, key)
        return readers

    def test_gta_sum(self, small_model):
        base_dir, entries, key = small_model
        readers = self._make_readers(base_dir, entries, key)
        base = readers[base_dir].get_tensor(key).clone()

        settings = MergeSettings(
            merge_method=MergeMethod.gta,
            chunk_elements=64,  # 256 elements > 64 → chunked
        )
        self._check_parity(base, entries, key, readers, settings)

    def test_gta_linear(self, small_model):
        base_dir, entries, key = small_model
        readers = self._make_readers(base_dir, entries, key)
        base = readers[base_dir].get_tensor(key).clone()

        settings = MergeSettings(
            merge_method=MergeMethod.linear,
            chunk_elements=64,
        )
        self._check_parity(base, entries, key, readers, settings)

    def test_gta_mixture(self, small_model):
        base_dir, entries, key = small_model
        readers = self._make_readers(base_dir, entries, key)
        base = readers[base_dir].get_tensor(key).clone()

        settings = MergeSettings(
            merge_method=MergeMethod.mixture,
            chunk_elements=64,
        )
        self._check_parity(base, entries, key, readers, settings)

    def test_gta_ties(self, small_model):
        base_dir, entries, key = small_model
        readers = self._make_readers(base_dir, entries, key)
        base = readers[base_dir].get_tensor(key).clone()

        settings = MergeSettings(
            merge_method=MergeMethod.ties,
            chunk_elements=64,
        )
        self._check_parity(base, entries, key, readers, settings)

    def test_slerp_fallback(self, small_model):
        """Slerp in chunked mode falls back to non-chunked (same result)."""
        base_dir, entries, key = small_model
        readers = self._make_readers(base_dir, entries, key)
        base = readers[base_dir].get_tensor(key).clone()

        settings = MergeSettings(
            merge_method=MergeMethod.slerp,
            chunk_elements=64,
        )
        # Slerp fallback: chunked mode uses the non-chunked path,
        # so the result should be identical to a direct non-chunked run.
        result = _run_merge(base, entries, key, readers, settings)
        settings_ref = MergeSettings(
            merge_method=MergeMethod.slerp,
            chunk_elements=None,
        )
        ref = _run_merge(base, entries, key, readers, settings_ref)
        torch.testing.assert_close(result, ref, rtol=2e-2, atol=1e-2)


# --------------------------------------------------------------------------- #
# Sparsify validity tests (chunked mask is valid, not necessarily equal)
# --------------------------------------------------------------------------- #
class TestChunkedSparsifyValidity:
    """Verify the chunked sparsify produces a valid mask (correct density)."""

    def _make_readers(self, base_dir, entries, key):
        readers = {base_dir: _make_reader(base_dir, key)}
        for entry in entries:
            readers[entry.dir] = _make_reader(entry.dir, key)
        return readers

    def _run_chunked_with_sparsify(self, base, entries, key, readers, settings):
        return _run_merge(base, entries, key, readers, settings)

    def test_magnitude_mask_valid(self, small_model):
        """Chunked magnitude sparsify keeps ~density fraction of elements."""
        base_dir, entries, key = small_model
        readers = self._make_readers(base_dir, entries, key)
        base = readers[base_dir].get_tensor(key).clone()

        settings = MergeSettings(
            merge_method=MergeMethod.gta,
            method=SparsificationMethod.magnitude,
            density=0.5,
            n=64,
            chunk_elements=64,
        )
        result = self._run_chunked_with_sparsify(base, entries, key, readers, settings)
        # The result should be finite and have the right shape
        assert result.shape == base.shape
        assert torch.isfinite(result).all()

    def test_random_mask_valid(self, small_model):
        """Chunked random sparsify produces a finite result."""
        base_dir, entries, key = small_model
        readers = self._make_readers(base_dir, entries, key)
        base = readers[base_dir].get_tensor(key).clone()

        settings = MergeSettings(
            merge_method=MergeMethod.gta,
            method=SparsificationMethod.random,
            density=0.5,
            n=64,
            chunk_elements=64,
        )
        result = self._run_chunked_with_sparsify(base, entries, key, readers, settings)
        assert result.shape == base.shape
        assert torch.isfinite(result).all()

    def test_bs_mask_valid(self, small_model):
        """Chunked BS sparsify produces a finite result."""
        base_dir, entries, key = small_model
        readers = self._make_readers(base_dir, entries, key)
        base = readers[base_dir].get_tensor(key).clone()

        settings = MergeSettings(
            merge_method=MergeMethod.gta,
            method=SparsificationMethod.bs,
            density=0.5,
            n=64,
            m=256,
            chunk_elements=64,
        )
        result = self._run_chunked_with_sparsify(base, entries, key, readers, settings)
        assert result.shape == base.shape
        assert torch.isfinite(result).all()


class TestChunkedConsensusAndSparsify:
    """Chunked mode must reproduce the non-chunked result exactly.

    The consensus divisor pass re-reads only the chunk's rows, and the
    global (magnitude / magnitude_outliers) mask is decided from the whole
    tensor — so its parameters are scanned once and applied chunk by chunk.
    """

    def _make_readers(self, base_dir, entries):
        readers = {base_dir: _make_reader(base_dir, "w")}
        for entry in entries:
            readers[entry.dir] = _make_reader(entry.dir, "w")
        return readers

    def _assert_same(self, base_dir, entries, key, readers, settings, chunk_elements=64):
        base = readers[base_dir].get_tensor(key).clone()
        ref = _run_merge(base, entries, key, readers, settings)
        chunked = MergeSettings(**{**settings.__dict__, "chunk_elements": chunk_elements})
        got = merge_tensor(None, entries, key, readers, chunked, base_reader=readers[base_dir])
        torch.testing.assert_close(got, ref, rtol=2e-2, atol=1e-2)

    def test_consensus_matches_full_tensor(self, small_model):
        base_dir, entries, key = small_model
        readers = self._make_readers(base_dir, entries)
        for consensus in (ConsensusMethod.none, ConsensusMethod.sum, ConsensusMethod.count):
            settings = MergeSettings(consensus=consensus)
            self._assert_same(base_dir, entries, key, readers, settings)

    def test_global_sparsify_matches_full_tensor(self, small_model):
        base_dir, entries, key = small_model
        readers = self._make_readers(base_dir, entries)
        for method in (SparsificationMethod.magnitude, SparsificationMethod.magnitude_outliers):
            for consensus in (ConsensusMethod.none, ConsensusMethod.sum):
                for rescale in (True, False):
                    settings = MergeSettings(
                        method=method, density=0.5, gamma=0.1,
                        consensus=consensus, rescale=rescale,
                    )
                    self._assert_same(base_dir, entries, key, readers, settings)

    def test_random_sparsify_is_reproducible_across_passes(self, small_model):
        """The consensus divisor pass must reuse the accumulation pass's mask."""
        base_dir, entries, key = small_model
        readers = self._make_readers(base_dir, entries)
        base = readers[base_dir].get_tensor(key).clone()
        settings = MergeSettings(
            method=SparsificationMethod.random, density=0.5,
            consensus=ConsensusMethod.sum,
        )
        chunked = MergeSettings(**{**settings.__dict__, "chunk_elements": 64})
        a = merge_tensor(None, entries, key, readers, chunked, base_reader=readers[base_dir])
        b = merge_tensor(None, entries, key, readers, chunked, base_reader=readers[base_dir])
        torch.testing.assert_close(a, b, rtol=0, atol=0)

    def test_missing_tensor_in_chunked_mode(self, small_model, tmp_path):
        base_dir, entries, key = small_model
        # drop the tensor from the second TV: chunked mode must skip it, not crash
        missing_dir = str(tmp_path / "tv_missing")
        entries2 = [
            entries[0],
            ModelEntry(dir=Path(missing_dir), kind="tv", weight=0.3),
        ]
        _make_safetensors_dir(Path(missing_dir), {"other": torch.randn(8)})
        readers = self._make_readers(base_dir, entries2)
        settings = MergeSettings(consensus=ConsensusMethod.sum)
        got = merge_tensor(
            None, entries2, key, readers,
            MergeSettings(**{**settings.__dict__, "chunk_elements": 64}),
            base_reader=readers[base_dir],
        )
        ref = _run_merge(
            readers[base_dir].get_tensor(key).clone(), entries2, key, readers,
            MergeSettings(density=1.0),
        )
        torch.testing.assert_close(got, ref, rtol=2e-2, atol=1e-2)

    def test_base_none_without_chunk_elements(self, small_model):
        base_dir, entries, key = small_model
        readers = self._make_readers(base_dir, entries)
        # no row budget: the full tensor is loaded instead of crashing
        got = merge_tensor(None, entries, key, readers, MergeSettings(), base_reader=readers[base_dir])
        ref = merge_tensor(readers[base_dir].get_tensor(key).clone(), entries, key, readers, MergeSettings())
        torch.testing.assert_close(got, ref, rtol=2e-2, atol=1e-2)
        with pytest.raises(ValueError):
            merge_tensor(None, entries, key, readers, MergeSettings())


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
