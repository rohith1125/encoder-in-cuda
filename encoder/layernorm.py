"""
Layer Normalization — from scratch.

In CUDA this is typically one fused kernel:
  1. Compute mean per token (parallel reduction)
  2. Compute variance per token (parallel reduction)
  3. Normalize + affine transform (element-wise)

All three steps fused into a single kernel launch to avoid
redundant global memory reads/writes.
"""

import numpy as np


class LayerNorm:
    """Layer Normalization over the last dimension (d_model).

    For input shape (batch, seq_len, d_model), normalizes over d_model.
    """

    def __init__(self, d_model: int, eps: float = 1e-6):
        self.eps = eps
        self.gamma = np.ones(d_model, dtype=np.float32)   # learnable scale
        self.beta = np.zeros(d_model, dtype=np.float32)    # learnable shift

    def __call__(self, x: np.ndarray) -> np.ndarray:
        """
        Args:
            x: (batch, seq_len, d_model)

        Returns:
            normalized: (batch, seq_len, d_model)

        In CUDA:
          - Each thread block handles one token (one row of length d_model)
          - Warp-level reduction for mean and variance
          - Single pass: Welford's online algorithm for numerical stability
        """
        # Mean over last axis — in CUDA: parallel reduction across d_model
        mean = np.mean(x, axis=-1, keepdims=True)

        # Variance — in CUDA: fused with mean computation via Welford's algorithm
        variance = np.var(x, axis=-1, keepdims=True)

        # Normalize — element-wise, fully parallel in CUDA
        x_norm = (x - mean) / np.sqrt(variance + self.eps)

        # Affine transform — element-wise multiply + add
        return self.gamma * x_norm + self.beta
