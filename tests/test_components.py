"""Unit tests for every component of the encoder block."""

import numpy as np
import pytest

from encoder.attention import ScaledDotProductAttention, MultiHeadSelfAttention, _softmax
from encoder.layernorm import LayerNorm
from encoder.feedforward import FeedForward
from encoder.positional import PositionalEncoding
from encoder.embedding import TokenEmbedding
from encoder.encoder import EncoderBlock, TransformerEncoder


# ── Softmax ──────────────────────────────────────────────────────────────────

class TestSoftmax:
    def test_output_sums_to_one(self):
        x = np.random.randn(4, 10).astype(np.float32)
        out = _softmax(x, axis=-1)
        np.testing.assert_allclose(out.sum(axis=-1), 1.0, atol=1e-6)

    def test_numerical_stability_with_large_values(self):
        x = np.array([[1000.0, 1001.0, 1002.0]], dtype=np.float32)
        out = _softmax(x, axis=-1)
        assert not np.any(np.isnan(out)), "softmax produced NaN on large inputs"
        np.testing.assert_allclose(out.sum(axis=-1), 1.0, atol=1e-6)

    def test_uniform_input_gives_uniform_output(self):
        x = np.ones((2, 5), dtype=np.float32) * 3.0
        out = _softmax(x, axis=-1)
        np.testing.assert_allclose(out, 0.2, atol=1e-6)


# ── Scaled Dot-Product Attention ─────────────────────────────────────────────

class TestScaledDotProductAttention:
    def setup_method(self):
        self.attn = ScaledDotProductAttention()
        self.batch, self.heads, self.seq, self.dk = 2, 4, 8, 16

    def test_output_shape(self):
        Q = np.random.randn(self.batch, self.heads, self.seq, self.dk).astype(np.float32)
        K = np.random.randn(self.batch, self.heads, self.seq, self.dk).astype(np.float32)
        V = np.random.randn(self.batch, self.heads, self.seq, self.dk).astype(np.float32)
        out, weights = self.attn(Q, K, V)
        assert out.shape == (self.batch, self.heads, self.seq, self.dk)
        assert weights.shape == (self.batch, self.heads, self.seq, self.seq)

    def test_attention_weights_sum_to_one(self):
        Q = np.random.randn(self.batch, self.heads, self.seq, self.dk).astype(np.float32)
        K = np.random.randn(self.batch, self.heads, self.seq, self.dk).astype(np.float32)
        V = np.random.randn(self.batch, self.heads, self.seq, self.dk).astype(np.float32)
        _, weights = self.attn(Q, K, V)
        row_sums = weights.sum(axis=-1)
        np.testing.assert_allclose(row_sums, 1.0, atol=1e-5)

    def test_mask_zeros_out_positions(self):
        Q = np.random.randn(1, 1, 4, 8).astype(np.float32)
        K = np.random.randn(1, 1, 4, 8).astype(np.float32)
        V = np.random.randn(1, 1, 4, 8).astype(np.float32)
        # Mask out positions 2 and 3
        mask = np.array([[[[1, 1, 0, 0]]]]).astype(np.float32)
        _, weights = self.attn(Q, K, V, mask)
        # Masked positions should have near-zero attention weight
        assert np.all(weights[0, 0, :, 2] < 1e-4)
        assert np.all(weights[0, 0, :, 3] < 1e-4)


# ── Multi-Head Self-Attention ────────────────────────────────────────────────

class TestMultiHeadSelfAttention:
    def test_output_shape(self):
        mha = MultiHeadSelfAttention(d_model=64, num_heads=8)
        x = np.random.randn(2, 10, 64).astype(np.float32)
        out, weights = mha(x)
        assert out.shape == (2, 10, 64)
        assert weights.shape == (2, 8, 10, 10)

    def test_different_seeds_give_different_outputs(self):
        mha1 = MultiHeadSelfAttention(d_model=32, num_heads=4, seed=1)
        mha2 = MultiHeadSelfAttention(d_model=32, num_heads=4, seed=2)
        x = np.random.randn(1, 5, 32).astype(np.float32)
        out1, _ = mha1(x)
        out2, _ = mha2(x)
        assert not np.allclose(out1, out2)

    def test_deterministic_with_same_seed(self):
        mha1 = MultiHeadSelfAttention(d_model=32, num_heads=4, seed=42)
        mha2 = MultiHeadSelfAttention(d_model=32, num_heads=4, seed=42)
        x = np.random.randn(1, 5, 32).astype(np.float32)
        out1, _ = mha1(x)
        out2, _ = mha2(x)
        np.testing.assert_array_equal(out1, out2)


# ── Layer Normalization ──────────────────────────────────────────────────────

class TestLayerNorm:
    def test_output_shape_preserved(self):
        ln = LayerNorm(64)
        x = np.random.randn(2, 10, 64).astype(np.float32)
        out = ln(x)
        assert out.shape == x.shape

    def test_normalized_mean_near_zero(self):
        ln = LayerNorm(128)
        x = np.random.randn(4, 20, 128).astype(np.float32) * 10 + 5
        out = ln(x)
        means = out.mean(axis=-1)
        np.testing.assert_allclose(means, 0.0, atol=1e-5)

    def test_normalized_std_near_one(self):
        ln = LayerNorm(128)
        x = np.random.randn(4, 20, 128).astype(np.float32) * 10 + 5
        out = ln(x)
        stds = out.std(axis=-1)
        np.testing.assert_allclose(stds, 1.0, atol=0.05)

    def test_identity_for_already_normalized(self):
        ln = LayerNorm(64)
        x = np.random.randn(1, 1, 64).astype(np.float32)
        x = (x - x.mean()) / (x.std() + 1e-6)
        out = ln(x)
        np.testing.assert_allclose(out, x, atol=1e-4)


# ── Feed-Forward Network ────────────────────────────────────────────────────

class TestFeedForward:
    def test_output_shape(self):
        ffn = FeedForward(d_model=64, d_ff=256)
        x = np.random.randn(2, 10, 64).astype(np.float32)
        out = ffn(x)
        assert out.shape == (2, 10, 64)

    def test_default_d_ff_is_4x(self):
        ffn = FeedForward(d_model=64)
        assert ffn.d_ff == 256
        assert ffn.W1.shape == (64, 256)
        assert ffn.W2.shape == (256, 64)


# ── Positional Encoding ─────────────────────────────────────────────────────

class TestPositionalEncoding:
    def test_output_shape(self):
        pe = PositionalEncoding(d_model=64, max_seq_len=100)
        x = np.zeros((2, 20, 64), dtype=np.float32)
        out = pe(x)
        assert out.shape == (2, 20, 64)

    def test_different_positions_get_different_encodings(self):
        pe = PositionalEncoding(d_model=64)
        x = np.zeros((1, 10, 64), dtype=np.float32)
        out = pe(x)
        # Each position should have a unique encoding
        for i in range(9):
            assert not np.allclose(out[0, i], out[0, i + 1])

    def test_encoding_values_bounded(self):
        pe = PositionalEncoding(d_model=64)
        # sin/cos values should be in [-1, 1]
        assert np.all(pe.pe >= -1.0)
        assert np.all(pe.pe <= 1.0)


# ── Token Embedding ──────────────────────────────────────────────────────────

class TestTokenEmbedding:
    def test_output_shape(self):
        emb = TokenEmbedding(vocab_size=1000, d_model=64)
        ids = np.array([[1, 5, 10, 20], [3, 7, 15, 25]])
        out = emb(ids)
        assert out.shape == (2, 4, 64)

    def test_same_token_same_embedding(self):
        emb = TokenEmbedding(vocab_size=100, d_model=32)
        ids = np.array([[5, 5, 5]])
        out = emb(ids)
        np.testing.assert_array_equal(out[0, 0], out[0, 1])
        np.testing.assert_array_equal(out[0, 1], out[0, 2])

    def test_scaling_by_sqrt_d_model(self):
        d_model = 64
        emb = TokenEmbedding(vocab_size=100, d_model=d_model)
        ids = np.array([[0]])
        out = emb(ids)
        raw = emb.weight[0]
        np.testing.assert_allclose(out[0, 0], raw * np.sqrt(d_model))


# ── Encoder Block ────────────────────────────────────────────────────────────

class TestEncoderBlock:
    def test_output_shape(self):
        block = EncoderBlock(d_model=64, num_heads=8)
        x = np.random.randn(2, 10, 64).astype(np.float32)
        out, weights = block(x)
        assert out.shape == (2, 10, 64)
        assert weights.shape == (2, 8, 10, 10)

    def test_residual_connection_effect(self):
        """Output should not be identical to input (attention modifies it)
        but also shouldn't be wildly different (residual keeps it grounded)."""
        block = EncoderBlock(d_model=64, num_heads=8)
        x = np.random.randn(1, 5, 64).astype(np.float32)
        out, _ = block(x)
        # Should be different
        assert not np.allclose(out, x)
        # But correlated (residual connection)
        correlation = np.corrcoef(x.flatten(), out.flatten())[0, 1]
        assert correlation > 0.0, "Residual connection should maintain some correlation"


# ── Full Transformer Encoder ─────────────────────────────────────────────────

class TestTransformerEncoder:
    def test_output_shape(self):
        enc = TransformerEncoder(
            vocab_size=1000, d_model=64, num_heads=8, num_layers=2
        )
        ids = np.array([[1, 5, 10, 20, 50], [3, 7, 15, 25, 99]])
        out, attn_weights = enc(ids)
        assert out.shape == (2, 5, 64)
        assert len(attn_weights) == 2
        assert attn_weights[0].shape == (2, 8, 5, 5)

    def test_parameter_count(self):
        enc = TransformerEncoder(
            vocab_size=100, d_model=32, num_heads=4, num_layers=2
        )
        params = enc.count_parameters()
        # Should be a reasonable number
        assert params > 0
        # Embedding: 100*32 = 3200
        # Per layer: 4*(32*32 + 32) + 2*(32*128 + 128 + 128*32 + 32) + 2*(32+32) = ...
        assert params > 3200  # at least the embedding

    def test_deterministic(self):
        enc = TransformerEncoder(
            vocab_size=100, d_model=32, num_heads=4, num_layers=2, seed=42
        )
        ids = np.array([[1, 2, 3]])
        out1, _ = enc(ids)
        out2, _ = enc(ids)
        np.testing.assert_array_equal(out1, out2)

    def test_config_stored(self):
        enc = TransformerEncoder(
            vocab_size=500, d_model=128, num_heads=8, num_layers=4
        )
        assert enc.config["vocab_size"] == 500
        assert enc.config["d_model"] == 128
        assert enc.config["num_heads"] == 8
        assert enc.config["num_layers"] == 4
        assert enc.config["d_ff"] == 512


# ── Integration: end-to-end forward pass ─────────────────────────────────────

class TestEndToEnd:
    def test_full_forward_pass_no_crash(self):
        """Smoke test: build a real-sized encoder and push data through."""
        enc = TransformerEncoder(
            vocab_size=10000, d_model=256, num_heads=8, num_layers=4
        )
        batch = np.random.randint(0, 10000, size=(4, 32))
        out, attn = enc(batch)
        assert out.shape == (4, 32, 256)
        assert len(attn) == 4
        assert not np.any(np.isnan(out)), "Forward pass produced NaN"

    def test_with_padding_mask(self):
        """Test that padding mask works end-to-end."""
        enc = TransformerEncoder(
            vocab_size=100, d_model=64, num_heads=4, num_layers=2
        )
        ids = np.array([[1, 2, 3, 0, 0], [5, 6, 0, 0, 0]])
        # Padding mask: 1 for real tokens, 0 for padding
        mask = (ids != 0).astype(np.float32)
        mask = mask[:, np.newaxis, np.newaxis, :]  # (batch, 1, 1, seq)
        out, _ = enc(ids, mask=mask)
        assert out.shape == (2, 5, 64)
        assert not np.any(np.isnan(out))
