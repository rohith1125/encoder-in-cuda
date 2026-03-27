"""
Position-wise Feed-Forward Network — two linear layers with GELU activation.

    FFN(x) = GELU(x @ W1 + b1) @ W2 + b2

In CUDA:
  - First GEMM: x @ W1 (cublas or custom tiled GEMM kernel)
  - Fused bias + GELU kernel (one element-wise kernel, no extra memory round-trip)
  - Second GEMM: activated @ W2
  - Bias add (fused into GEMM epilogue on modern GPUs)

The inner dimension (d_ff) is typically 4x d_model.
"""

import numpy as np


def _gelu(x: np.ndarray) -> np.ndarray:
    """GELU activation — tanh approximation.

    In CUDA: single element-wise kernel, often fused with preceding bias-add.
    """
    return 0.5 * x * (1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (x + 0.044715 * x ** 3)))


class FeedForward:
    """Position-wise FFN: Linear(d_model, d_ff) -> GELU -> Linear(d_ff, d_model)."""

    def __init__(self, d_model: int, d_ff: int | None = None, seed: int = 42):
        if d_ff is None:
            d_ff = 4 * d_model  # standard transformer scaling

        self.d_model = d_model
        self.d_ff = d_ff

        rng = np.random.default_rng(seed)
        scale1 = np.sqrt(2.0 / d_model)
        scale2 = np.sqrt(2.0 / d_ff)

        # First linear: d_model -> d_ff
        self.W1 = rng.normal(0, scale1, (d_model, d_ff)).astype(np.float32)
        self.b1 = np.zeros(d_ff, dtype=np.float32)

        # Second linear: d_ff -> d_model
        self.W2 = rng.normal(0, scale2, (d_ff, d_model)).astype(np.float32)
        self.b2 = np.zeros(d_model, dtype=np.float32)

    def __call__(self, x: np.ndarray) -> np.ndarray:
        """
        Args:
            x: (batch, seq_len, d_model)

        Returns:
            output: (batch, seq_len, d_model)

        In CUDA kernel sequence:
          1. tiled GEMM: x @ W1           — shared memory tiling for cache efficiency
          2. fused kernel: + b1, GELU()   — single kernel, no extra global mem write
          3. tiled GEMM: hidden @ W2      — second matmul
          4. + b2                          — fused into GEMM epilogue
        """
        # First linear + GELU
        hidden = x @ self.W1 + self.b1
        hidden = _gelu(hidden)

        # Second linear
        output = hidden @ self.W2 + self.b2
        return output
