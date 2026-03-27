"""
Multi-Head Self-Attention — every matrix op written by hand in NumPy.

In a CUDA kernel this would be:
  - One kernel for QKV projection (batched GEMM)
  - One kernel for scaled dot-product + causal mask
  - One kernel for row-wise softmax (online/stable)
  - One kernel for attention-weighted value aggregation
  - One kernel for output projection

Here we mirror that structure function-by-function.
"""

import numpy as np


def _softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """Numerically stable softmax — same trick used in CUDA online softmax kernels.

    Subtract max per row to prevent exp() overflow, then normalize.
    GPU kernels do this in two passes: one for max, one for sum+normalize.
    """
    x_max = np.max(x, axis=axis, keepdims=True)
    exp_x = np.exp(x - x_max)
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)


def _gelu(x: np.ndarray) -> np.ndarray:
    """GELU activation — approximation used in GPT-2 / BERT.

    In CUDA this maps to a single fused kernel: x * 0.5 * (1 + tanh(...))
    """
    return 0.5 * x * (1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (x + 0.044715 * x ** 3)))


class ScaledDotProductAttention:
    """Computes Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) V

    Mirrors what a single CUDA thread block would compute for one attention head.
    """

    def __call__(
        self,
        Q: np.ndarray,
        K: np.ndarray,
        V: np.ndarray,
        mask: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Args:
            Q: (batch, heads, seq_len, d_k)
            K: (batch, heads, seq_len, d_k)
            V: (batch, heads, seq_len, d_k)
            mask: (batch, 1, 1, seq_len) or (batch, 1, seq_len, seq_len)

        Returns:
            output: (batch, heads, seq_len, d_k)
            attention_weights: (batch, heads, seq_len, seq_len)
        """
        d_k = Q.shape[-1]

        # QK^T / sqrt(d_k) — in CUDA this is a batched GEMM + element-wise scale
        scores = np.matmul(Q, K.transpose(0, 1, 3, 2)) / np.sqrt(d_k)

        # Apply mask (padding or causal) — sets masked positions to -inf before softmax
        if mask is not None:
            scores = np.where(mask == 0, -1e9, scores)

        # Row-wise softmax — in CUDA: online softmax kernel (two-pass for numerical stability)
        attention_weights = _softmax(scores, axis=-1)

        # Weighted sum of values — another batched GEMM
        output = np.matmul(attention_weights, V)

        return output, attention_weights


class MultiHeadSelfAttention:
    """Multi-Head Attention: splits d_model into h heads, runs attention in parallel.

    In CUDA, each head maps to an independent thread block (or warp group).
    QKV projections are a single fused GEMM: [x] @ [W_q | W_k | W_v].
    """

    def __init__(self, d_model: int, num_heads: int, seed: int = 42):
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        rng = np.random.default_rng(seed)
        scale = np.sqrt(2.0 / d_model)  # Xavier/Glorot init

        # In CUDA: these would be a single contiguous weight buffer for fused QKV GEMM
        self.W_q = rng.normal(0, scale, (d_model, d_model)).astype(np.float32)
        self.W_k = rng.normal(0, scale, (d_model, d_model)).astype(np.float32)
        self.W_v = rng.normal(0, scale, (d_model, d_model)).astype(np.float32)
        self.W_o = rng.normal(0, scale, (d_model, d_model)).astype(np.float32)

        self.b_q = np.zeros(d_model, dtype=np.float32)
        self.b_k = np.zeros(d_model, dtype=np.float32)
        self.b_v = np.zeros(d_model, dtype=np.float32)
        self.b_o = np.zeros(d_model, dtype=np.float32)

        self.attention = ScaledDotProductAttention()

    def _split_heads(self, x: np.ndarray) -> np.ndarray:
        """Reshape (batch, seq, d_model) -> (batch, heads, seq, d_k).

        In CUDA: this is a memory layout transformation (no data copy needed
        if the kernel reads with appropriate strides).
        """
        batch, seq_len, _ = x.shape
        x = x.reshape(batch, seq_len, self.num_heads, self.d_k)
        return x.transpose(0, 2, 1, 3)  # (batch, heads, seq, d_k)

    def _merge_heads(self, x: np.ndarray) -> np.ndarray:
        """Reshape (batch, heads, seq, d_k) -> (batch, seq, d_model).

        Inverse of _split_heads. In CUDA: contiguous write after attention.
        """
        batch, _, seq_len, _ = x.shape
        x = x.transpose(0, 2, 1, 3)  # (batch, seq, heads, d_k)
        return x.reshape(batch, seq_len, self.d_model)

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
        # Linear projections — in CUDA: single fused GEMM for QKV
        Q = x @ self.W_q + self.b_q
        K = x @ self.W_k + self.b_k
        V = x @ self.W_v + self.b_v

        # Split into heads — in CUDA: just a stride change
        Q = self._split_heads(Q)
        K = self._split_heads(K)
        V = self._split_heads(V)

        # Scaled dot-product attention per head
        attn_output, attn_weights = self.attention(Q, K, V, mask)

        # Merge heads back
        output = self._merge_heads(attn_output)

        # Output projection
        output = output @ self.W_o + self.b_o

        return output, attn_weights
