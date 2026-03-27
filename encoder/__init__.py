"""
Transformer Encoder Block — from-scratch implementation in pure Python + NumPy.
No PyTorch, no TensorFlow. Every operation hand-written.

Implements:
  - Scaled Dot-Product Attention
  - Multi-Head Self-Attention
  - Layer Normalization
  - Position-wise Feed-Forward Network (with GELU)
  - Residual Connections
  - Positional Encoding
  - Full Encoder Block + Stackable Encoder
  - Minimal Autograd Engine (backward pass)
  - Low-level Kernel Simulations (tiled GEMM, online softmax, fused ops)
"""

from .attention import ScaledDotProductAttention, MultiHeadSelfAttention
from .layernorm import LayerNorm
from .feedforward import FeedForward
from .encoder import EncoderBlock, TransformerEncoder
from .positional import PositionalEncoding
from .embedding import TokenEmbedding
from .autograd import Tensor

__all__ = [
    "ScaledDotProductAttention",
    "MultiHeadSelfAttention",
    "LayerNorm",
    "FeedForward",
    "EncoderBlock",
    "TransformerEncoder",
    "PositionalEncoding",
    "TokenEmbedding",
    "Tensor",
]
