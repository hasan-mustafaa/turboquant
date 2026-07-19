"""Outlier-split validation (paper Section 4.3 mixed precision), model-free.

The synthetic source plants a known set of high-variance channels; the tests
check that (a) selection recovers exactly those channels, (b) the split
allocates error where it should -- large improvement over uniform low-bit on
this source, bounded below by uniform high-bit, (c) channel reassembly is
positionally exact, and (d) storage accounting matches the closed form.
Also covers the TurboQuantCache integration path synthetically.
"""

import math

import pytest
import torch

from turboquant.outlier import (
    OutlierSplitQuantizer,
    SplitQuantizedBatch,
    select_outlier_channels,
)

D = 64
OUT_IDX = (3, 17, 18, 31, 40, 41, 55, 63)  # planted outlier channels


def make_source(n: int, seed: int, scale: float = 30.0) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(n, D, generator=g)
    x[:, list(OUT_IDX)] *= scale
    return x


class TestSelection:
    def test_recovers_planted_channels(self):
        idx = select_outlier_channels(make_source(2_000, seed=0), len(OUT_IDX))
        assert idx.tolist() == sorted(OUT_IDX)

    def test_multidim_sample(self):
        x = make_source(1_024, seed=1).reshape(4, 16, 16, D)
        idx = select_outlier_channels(x, len(OUT_IDX))
        assert idx.tolist() == sorted(OUT_IDX)

    def test_rejects_bad_counts(self):
        x = make_source(10, seed=2)
        with pytest.raises(ValueError):
            select_outlier_channels(x, 0)
        with pytest.raises(ValueError):
            select_outlier_channels(x, D)


class TestOutlierSplitQuantizer:
    def _rel_err(self, q, x):
        return ((x - q.roundtrip(x)).pow(2).sum(-1) / x.pow(2).sum(-1)).mean().item()

    def test_beats_uniform_low_bits_on_outlier_source(self):
        from turboquant.quantizer import TurboQuantMSE

        x = make_source(20_000, seed=3)
        idx = select_outlier_channels(x[:2_000], len(OUT_IDX))
        split = OutlierSplitQuantizer(D, idx, bits_outlier=8, bits_regular=2,
                                      seed=4)
        uni_lo = TurboQuantMSE(D, 2, seed=4)
        uni_hi = TurboQuantMSE(D, 8, seed=4)
        e_split, e_lo, e_hi = (self._rel_err(q, x)
                               for q in (split, uni_lo, uni_hi))
        # On this source ~99% of the energy sits in the 8 planted channels;
        # giving them 8 bits must recover most of the uniform-2-bit loss.
        assert e_split < e_lo / 5, (e_split, e_lo)
        assert e_hi < e_split  # more bits everywhere is still strictly better

    def test_channel_reassembly_positionally_exact(self):
        """Quantize at b=8 on both groups: per-channel error must be tiny in
        *every* channel -- a scatter/gather bug would misplace channels and
        show up as O(1) error in the swapped positions."""
        x = make_source(2_000, seed=5)
        idx = select_outlier_channels(x, len(OUT_IDX))
        split = OutlierSplitQuantizer(D, idx, bits_outlier=8, bits_regular=8,
                                      seed=6)
        err = (x - split.roundtrip(x)).pow(2).mean(0)  # per-channel
        scale = x.pow(2).mean(0)
        assert (err / scale).max().item() < 1e-3

    def test_effective_bits_and_storage(self):
        n = 100
        idx = torch.arange(16)
        split = OutlierSplitQuantizer(D, idx, bits_outlier=3, bits_regular=2,
                                      seed=7)
        # paper's 2.5-bit pattern scaled to d=64: (16*3 + 48*2)/64 = 2.25
        assert split.effective_bits == pytest.approx((16 * 3 + 48 * 2) / 64)
        q = split.quantize(make_source(n, seed=8))
        expected = n * (math.ceil(16 * 3 / 8) + 2      # outlier codes + norm
                        + math.ceil(48 * 2 / 8) + 2)   # regular codes + norm
        assert q.num_bytes == expected

    def test_rejects_duplicate_and_full_indices(self):
        with pytest.raises(ValueError):
            OutlierSplitQuantizer(D, torch.tensor([1, 1, 2]), 4, 2)
        with pytest.raises(ValueError):
            OutlierSplitQuantizer(D, torch.arange(D), 4, 2)

    def test_cat(self):
        split = OutlierSplitQuantizer(D, torch.arange(8), 4, 2, seed=9)
        a = split.quantize(make_source(5, seed=10))
        b = split.quantize(make_source(3, seed=11))
        ab = a.cat(b)
        assert isinstance(ab, SplitQuantizedBatch)
        assert ab.out.codes.shape[-2] == 8
        # codes must concatenate exactly ...
        assert torch.equal(ab.out.codes[:5], a.out.codes)
        assert torch.equal(ab.reg.codes[5:], b.reg.codes)
        assert torch.equal(ab.out.norms[:5], a.out.norms)
        # ... while decoded values are allclose, not bit-equal: batched matmul
        # blocking differs between a size-8 and a size-5 decode of identical
        # codes, so exact float equality is not a valid invariant here.
        got = split.dequantize(ab)
        want = torch.cat(
            [split.roundtrip(make_source(5, seed=10)),
             split.roundtrip(make_source(3, seed=11))], dim=-2)
        # fp32 few-ulp tolerance at the data's scale (outlier channels ~ 1e2)
        assert torch.allclose(got, want, rtol=1e-5, atol=1e-3)


class TestKVCacheIntegration:
    def test_synthetic_update_with_outlier_split(self):
        from turboquant.kv_cache import TurboQuantCache

        torch.manual_seed(0)
        B, H, T, hd = 1, 2, 48, 64
        k = torch.randn(B, H, T, hd, dtype=torch.float16)
        k[..., :4] *= 25.0  # planted outlier key channels
        v = torch.randn(B, H, T, hd, dtype=torch.float16)
        for packed in (False, True):
            cache = TurboQuantCache(bits_k=2, bits_v=4, outlier_channels=4,
                                    bits_k_outlier=8, packed=packed)
            ko, vo = cache.update(k, v, 0)
            assert ko.shape == k.shape and vo.shape == v.shape
            assert torch.isfinite(ko).all() and torch.isfinite(vo).all()
            layer = cache.layers[0]
            assert layer._split_k is not None
            assert set(layer._split_k.idx_out.tolist()) == {0, 1, 2, 3}
            # warmup tokens exact, quantized region not
            assert torch.equal(ko[..., :32, :], k[..., :32, :])
            assert not torch.equal(ko[..., 32:, :], k[..., 32:, :])
            # split (8 bits on the hot channels) must beat uniform b=2 keys
            uni = TurboQuantCache(bits_k=2, bits_v=4, packed=packed)
            ku, _ = uni.update(k, v, 0)
            e_split = (ko - k)[..., 32:, :].float().pow(2).mean()
            e_uni = (ku - k)[..., 32:, :].float().pow(2).mean()
            assert e_split < e_uni / 2

    def test_incremental_matches_oneshot(self):
        from turboquant.kv_cache import TurboQuantCache

        torch.manual_seed(1)
        k = torch.randn(1, 2, 50, 64, dtype=torch.float16)
        v = torch.randn(1, 2, 50, 64, dtype=torch.float16)
        one = TurboQuantCache(bits_k=2, outlier_channels=8, seed=5)
        ko_one, _ = one.update(k, v, 0)
        two = TurboQuantCache(bits_k=2, outlier_channels=8, seed=5)
        two.update(k[..., :40, :], v[..., :40, :], 0)
        ko_two, _ = two.update(k[..., 40:, :], v[..., 40:, :], 0)
        assert torch.equal(ko_one, ko_two)
