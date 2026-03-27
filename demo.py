#!/usr/bin/env python3
"""
Demo: Transformer Encoder Block — from-scratch NumPy implementation.

Shows the full pipeline: token IDs -> embeddings -> positional encoding
-> N encoder blocks -> final output representations.
"""

import time
import numpy as np
from encoder import TransformerEncoder


def main():
    print("=" * 70)
    print("  Transformer Encoder Block — Pure NumPy, No Framework")
    print("  Every matrix op written from scratch")
    print("=" * 70)

    # ── Configuration (BERT-base scale) ──────────────────────────────────
    config = {
        "vocab_size": 30522,   # BERT vocabulary size
        "d_model": 256,        # embedding dimension
        "num_heads": 8,        # attention heads
        "num_layers": 6,       # encoder blocks
        "d_ff": 1024,          # feed-forward inner dimension
        "max_seq_len": 512,
    }

    print(f"\n  Config:")
    for k, v in config.items():
        print(f"    {k}: {v}")

    # ── Build encoder ────────────────────────────────────────────────────
    print("\n  Building encoder...", end=" ", flush=True)
    t0 = time.time()
    encoder = TransformerEncoder(**config, seed=42)
    build_time = time.time() - t0
    print(f"done ({build_time:.2f}s)")

    params = encoder.count_parameters()
    print(f"  Parameters: {params:,} ({params * 4 / 1e6:.1f} MB at fp32)")

    # ── Fake input (simulating tokenized text) ───────────────────────────
    batch_size = 2
    seq_len = 32
    rng = np.random.default_rng(0)
    token_ids = rng.integers(0, config["vocab_size"], size=(batch_size, seq_len))

    # Padding mask — pretend last 4 tokens are padding
    pad_mask = np.ones((batch_size, seq_len), dtype=np.float32)
    pad_mask[:, -4:] = 0.0
    pad_mask = pad_mask[:, np.newaxis, np.newaxis, :]  # (B, 1, 1, S)

    print(f"\n  Input: batch={batch_size}, seq_len={seq_len}")
    print(f"  Token IDs sample: {token_ids[0, :8]}...")

    # ── Forward pass ─────────────────────────────────────────────────────
    print("\n  Running forward pass...", end=" ", flush=True)
    t0 = time.time()
    output, attention_weights = encoder(token_ids, mask=pad_mask)
    forward_time = time.time() - t0
    print(f"done ({forward_time:.3f}s)")

    # ── Results ──────────────────────────────────────────────────────────
    print(f"\n  Output shape: {output.shape}")
    print(f"  Output dtype: {output.dtype}")
    print(f"  Output range: [{output.min():.4f}, {output.max():.4f}]")
    print(f"  Output mean:  {output.mean():.6f}")
    print(f"  Output std:   {output.std():.4f}")

    print(f"\n  Attention weights per layer:")
    for i, aw in enumerate(attention_weights):
        print(f"    Layer {i}: shape={aw.shape}, "
              f"entropy={_attention_entropy(aw):.3f} bits")

    # ── Verify no NaNs ───────────────────────────────────────────────────
    has_nan = np.any(np.isnan(output))
    print(f"\n  NaN check: {'FAIL' if has_nan else 'PASS'}")

    # ── Verify padding mask effect ───────────────────────────────────────
    print("\n  Padding mask verification:")
    last_layer_attn = attention_weights[-1]  # (B, H, S, S)
    padded_attn = last_layer_attn[:, :, :, -4:]  # attention to padded positions
    max_padded_attn = padded_attn.max()
    print(f"    Max attention to padded positions: {max_padded_attn:.6f}")
    print(f"    Padding properly masked: {'YES' if max_padded_attn < 0.01 else 'NO'}")

    print("\n" + "=" * 70)
    print("  All operations executed successfully.")
    print(f"  Total time: {build_time + forward_time:.3f}s")
    print("=" * 70)


def _attention_entropy(weights: np.ndarray) -> float:
    """Average entropy of attention distributions (bits). Higher = more uniform."""
    # weights: (batch, heads, seq, seq)
    w = weights.clip(1e-10, 1.0)
    entropy = -np.sum(w * np.log2(w), axis=-1)
    return float(entropy.mean())


if __name__ == "__main__":
    main()
