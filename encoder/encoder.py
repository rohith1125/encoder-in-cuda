"""
Transformer Encoder Block and stackable Encoder.

One Encoder Block:
    x -> LayerNorm -> MultiHeadAttention -> + residual
      -> LayerNorm -> FeedForward        -> + residual

Uses Pre-LN architecture (GPT-2 / modern transformers) instead of Post-LN
(original Vaswani). Pre-LN is more stable for training and what most
production models use.

In CUDA, the full encoder block is typically:
  - 2 LayerNorm kernels (fused mean+var+normalize+affine)
  - 1 fused QKV GEMM kernel
  - 1 attention kernel (scores + softmax + value aggregation)
  - 1 output projection GEMM
  - 2 FFN GEMM kernels with fused GELU
  - 2 residual-add kernels (often fused into the preceding operation)
  Total: ~8 kernel launches per block (can be reduced with kernel fusion)
"""

import numpy as np

from .attention import MultiHeadSelfAttention
from .layernorm import LayerNorm
from .feedforward import FeedForward
from .positional import PositionalEncoding
from .embedding import TokenEmbedding


class EncoderBlock:
    """Single Transformer Encoder Block (Pre-LN variant).

    Architecture:
        x' = x + MultiHeadAttention(LayerNorm(x))
        out = x' + FeedForward(LayerNorm(x'))
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int | None = None,
        seed: int = 42,
    ):
        self.norm1 = LayerNorm(d_model)
        self.attention = MultiHeadSelfAttention(d_model, num_heads, seed=seed)
        self.norm2 = LayerNorm(d_model)
        self.ffn = FeedForward(d_model, d_ff, seed=seed + 1)

    def __call__(
        self, x: np.ndarray, mask: np.ndarray | None = None
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Args:
            x: (batch, seq_len, d_model)
            mask: optional attention mask

        Returns:
            output: (batch, seq_len, d_model)
            attention_weights: (batch, heads, seq_len, seq_len)
        """
        # Sub-layer 1: Multi-Head Self-Attention with residual
        normed = self.norm1(x)
        attn_out, attn_weights = self.attention(normed, mask)
        x = x + attn_out  # Residual connection

        # Sub-layer 2: Feed-Forward with residual
        normed = self.norm2(x)
        ff_out = self.ffn(normed)
        x = x + ff_out  # Residual connection

        return x, attn_weights


class TransformerEncoder:
    """Full Transformer Encoder: Embedding + Positional Encoding + N x EncoderBlock.

    This is the complete encoder stack as used in BERT, Vision Transformers,
    and the encoder half of the original Transformer.
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 512,
        num_heads: int = 8,
        num_layers: int = 6,
        d_ff: int | None = None,
        max_seq_len: int = 5000,
        seed: int = 42,
    ):
        self.embedding = TokenEmbedding(vocab_size, d_model, seed=seed)
        self.pos_encoding = PositionalEncoding(d_model, max_seq_len)

        self.layers = [
            EncoderBlock(d_model, num_heads, d_ff, seed=seed + i)
            for i in range(num_layers)
        ]

        self.final_norm = LayerNorm(d_model)

        # Config for inspection
        self.config = {
            "vocab_size": vocab_size,
            "d_model": d_model,
            "num_heads": num_heads,
            "num_layers": num_layers,
            "d_ff": d_ff or 4 * d_model,
            "max_seq_len": max_seq_len,
        }

    def __call__(
        self, token_ids: np.ndarray, mask: np.ndarray | None = None
    ) -> tuple[np.ndarray, list[np.ndarray]]:
        """
        Args:
            token_ids: (batch, seq_len) integer token indices
            mask: optional attention mask

        Returns:
            output: (batch, seq_len, d_model)
            all_attention_weights: list of (batch, heads, seq_len, seq_len) per layer
        """
        # Embedding + positional encoding
        x = self.embedding(token_ids)
        x = self.pos_encoding(x)

        # Pass through encoder blocks
        all_attn_weights = []
        for layer in self.layers:
            x, attn_weights = layer(x, mask)
            all_attn_weights.append(attn_weights)

        # Final layer norm
        x = self.final_norm(x)

        return x, all_attn_weights

    def count_parameters(self) -> int:
        """Count total number of parameters in the encoder."""
        total = 0

        # Embedding
        total += self.embedding.weight.size

        # Per encoder block
        for layer in self.layers:
            # Attention: 4 weight matrices + 4 bias vectors
            attn = layer.attention
            total += attn.W_q.size + attn.W_k.size + attn.W_v.size + attn.W_o.size
            total += attn.b_q.size + attn.b_k.size + attn.b_v.size + attn.b_o.size

            # FFN: 2 weight matrices + 2 bias vectors
            ffn = layer.ffn
            total += ffn.W1.size + ffn.W2.size + ffn.b1.size + ffn.b2.size

            # LayerNorm: 2x (gamma + beta)
            total += layer.norm1.gamma.size + layer.norm1.beta.size
            total += layer.norm2.gamma.size + layer.norm2.beta.size

        # Final LayerNorm
        total += self.final_norm.gamma.size + self.final_norm.beta.size

        return total
