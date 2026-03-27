"""Tests for low-level kernel implementations."""

import numpy as np
import pytest

from encoder.kernels import (
    tiled_matmul,
    online_softmax,
    fused_layernorm,
    fused_bias_gelu,
    fused_residual_layernorm,
    causal_mask,
    padding_mask,
)


class TestTiledMatmul:
    def test_matches_numpy_matmul(self):
        A = np.random.randn(4, 32, 64).astype(np.float32)
        B = np.random.randn(4, 64, 48).astype(np.float32)
        ours = tiled_matmul(A, B, tile_size=16)
        expected = A @ B
        np.testing.assert_allclose(ours, expected, atol=1e-4)

    def test_various_tile_sizes(self):
        A = np.random.randn(2, 17, 33).astype(np.float32)
        B = np.random.randn(2, 33, 21).astype(np.float32)
        expected = A @ B
        for tile in [4, 8, 16, 32, 64]:
            result = tiled_matmul(A, B, tile_size=tile)
            np.testing.assert_allclose(result, expected, atol=1e-4,
                                       err_msg=f"Failed with tile_size={tile}")

    def test_non_divisible_dimensions(self):
        A = np.random.randn(3, 7, 13).astype(np.float32)
        B = np.random.randn(3, 13, 5).astype(np.float32)
        ours = tiled_matmul(A, B, tile_size=4)
        np.testing.assert_allclose(ours, A @ B, atol=1e-4)


class TestOnlineSoftmax:
    def test_sums_to_one(self):
        x = np.random.randn(8, 16).astype(np.float32)
        out = online_softmax(x, axis=-1)
        np.testing.assert_allclose(out.sum(axis=-1), 1.0, atol=1e-6)

    def test_stable_with_large_values(self):
        x = np.array([[1e4, 1e4 + 1, 1e4 + 2]], dtype=np.float32)
        out = online_softmax(x, axis=-1)
        assert not np.any(np.isnan(out))
        np.testing.assert_allclose(out.sum(), 1.0, atol=1e-6)

    def test_matches_naive_softmax(self):
        from encoder.attention import _softmax
        x = np.random.randn(4, 8, 16).astype(np.float32)
        ours = online_softmax(x, axis=-1)
        theirs = _softmax(x, axis=-1)
        np.testing.assert_allclose(ours, theirs, atol=1e-6)


class TestFusedLayerNorm:
    def test_output_normalized(self):
        x = np.random.randn(4, 10, 64).astype(np.float32) * 5 + 3
        gamma = np.ones(64, dtype=np.float32)
        beta = np.zeros(64, dtype=np.float32)
        out = fused_layernorm(x, gamma, beta)
        np.testing.assert_allclose(out.mean(axis=-1), 0.0, atol=1e-5)
        np.testing.assert_allclose(out.std(axis=-1), 1.0, atol=0.05)

    def test_matches_standard_layernorm(self):
        from encoder.layernorm import LayerNorm
        d = 128
        x = np.random.randn(2, 8, d).astype(np.float32)
        ln = LayerNorm(d)
        standard = ln(x)
        fused = fused_layernorm(x, ln.gamma, ln.beta)
        np.testing.assert_allclose(fused, standard, atol=1e-5)


class TestFusedBiasGelu:
    def test_matches_separate_ops(self):
        from encoder.feedforward import _gelu
        x = np.random.randn(4, 10, 256).astype(np.float32)
        bias = np.random.randn(256).astype(np.float32)
        fused = fused_bias_gelu(x, bias)
        separate = _gelu(x + bias)
        np.testing.assert_allclose(fused, separate, atol=1e-6)


class TestFusedResidualLayerNorm:
    def test_matches_separate_ops(self):
        from encoder.layernorm import LayerNorm
        d = 64
        residual = np.random.randn(2, 10, d).astype(np.float32)
        x = np.random.randn(2, 10, d).astype(np.float32)
        ln = LayerNorm(d)
        separate = ln(residual + x)
        fused = fused_residual_layernorm(residual, x, ln.gamma, ln.beta)
        np.testing.assert_allclose(fused, separate, atol=1e-5)


class TestCausalMask:
    def test_shape(self):
        mask = causal_mask(8)
        assert mask.shape == (1, 1, 8, 8)

    def test_lower_triangular(self):
        mask = causal_mask(5)
        m = mask[0, 0]
        # Upper triangle should be 0
        for i in range(5):
            for j in range(i + 1, 5):
                assert m[i, j] == 0.0
        # Lower triangle + diagonal should be 1
        for i in range(5):
            for j in range(i + 1):
                assert m[i, j] == 1.0


class TestPaddingMask:
    def test_shape(self):
        lengths = np.array([3, 5, 2])
        mask = padding_mask(lengths, max_len=6)
        assert mask.shape == (3, 1, 1, 6)

    def test_correct_masking(self):
        lengths = np.array([3, 5])
        mask = padding_mask(lengths, max_len=6)
        # First sequence: length 3, positions 0,1,2 valid
        np.testing.assert_array_equal(mask[0, 0, 0], [1, 1, 1, 0, 0, 0])
        # Second sequence: length 5, positions 0-4 valid
        np.testing.assert_array_equal(mask[1, 0, 0], [1, 1, 1, 1, 1, 0])
