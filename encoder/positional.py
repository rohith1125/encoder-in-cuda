"""
Positional Encoding — sinusoidal (Vaswani et al., 2017).

    PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
    PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))

In CUDA: precomputed once and stored in constant memory or texture memory
for fast broadcast-add during the forward pass.
"""

import numpy as np


class PositionalEncoding:
    """Sinusoidal positional encoding — precomputed lookup table."""

    def __init__(self, d_model: int, max_seq_len: int = 5000):
        self.d_model = d_model

        # Precompute the entire PE table — in CUDA: done once, stored in constant memory
        pe = np.zeros((max_seq_len, d_model), dtype=np.float32)
        position = np.arange(max_seq_len, dtype=np.float32)[:, np.newaxis]
        div_term = np.exp(
            np.arange(0, d_model, 2, dtype=np.float32) * -(np.log(10000.0) / d_model)
        )

        pe[:, 0::2] = np.sin(position * div_term)
        pe[:, 1::2] = np.cos(position * div_term)

        # Shape: (1, max_seq_len, d_model) for broadcasting across batch
        self.pe = pe[np.newaxis, :, :]

    def __call__(self, x: np.ndarray) -> np.ndarray:
        """
        Args:
            x: (batch, seq_len, d_model) — typically token embeddings

        Returns:
            x + positional_encoding: (batch, seq_len, d_model)
        """
        seq_len = x.shape[1]
        return x + self.pe[:, :seq_len, :]
