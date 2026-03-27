"""
Token Embedding — simple lookup table.

In CUDA: this is a gather operation from a (vocab_size, d_model) weight matrix.
Each thread fetches one row. On modern GPUs, embedding lookup is memory-bound
(limited by HBM bandwidth, not compute).
"""

import numpy as np


class TokenEmbedding:
    """Lookup-table embedding with sqrt(d_model) scaling (Vaswani et al.)."""

    def __init__(self, vocab_size: int, d_model: int, seed: int = 42):
        self.d_model = d_model
        rng = np.random.default_rng(seed)

        # Weight matrix — in CUDA: stored in global memory, accessed via gather
        self.weight = rng.normal(0, 1.0, (vocab_size, d_model)).astype(np.float32)

    def __call__(self, token_ids: np.ndarray) -> np.ndarray:
        """
        Args:
            token_ids: (batch, seq_len) integer token indices

        Returns:
            embeddings: (batch, seq_len, d_model) scaled by sqrt(d_model)
        """
        # Gather rows — in CUDA: each thread reads one row from global memory
        return self.weight[token_ids] * np.sqrt(self.d_model)
