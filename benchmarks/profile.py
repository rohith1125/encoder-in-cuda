#!/usr/bin/env python3
"""
Performance profiling — measures time spent in each component of the encoder.

Shows where time is spent in the forward pass, mirroring how you'd
use NVIDIA Nsight Systems or nvprof to profile CUDA kernel execution.

Usage:
    python benchmarks/profile.py
"""

import time
import numpy as np
from encoder import TransformerEncoder
from encoder.attention import MultiHeadSelfAttention, _softmax
from encoder.layernorm import LayerNorm
from encoder.feedforward import FeedForward


def profile_component(name: str, fn, iterations: int = 500) -> float:
    """Time a function over multiple iterations."""
    # Warmup
    for _ in range(10):
        fn()

    t0 = time.perf_counter()
    for _ in range(iterations):
        fn()
    elapsed = time.perf_counter() - t0
    avg_ms = (elapsed / iterations) * 1000

    print(f"    {name:<35s}  {avg_ms:>8.3f} ms/call  ({iterations} iters)")
    return avg_ms


def main():
    print("=" * 70)
    print("  Encoder Block — Component Profiling")
    print("  (analogous to CUDA kernel profiling with nsight-systems)")
    print("=" * 70)

    d_model = 256
    num_heads = 8
    batch_size = 4
    seq_len = 64
    d_ff = 1024

    x = np.random.randn(batch_size, seq_len, d_model).astype(np.float32)

    # ── Individual kernels ───────────────────────────────────────────────
    print("\n  Individual kernel equivalents:")

    # Softmax
    scores = np.random.randn(batch_size, num_heads, seq_len, seq_len).astype(np.float32)
    profile_component("softmax (row-wise)", lambda: _softmax(scores, axis=-1))

    # LayerNorm
    ln = LayerNorm(d_model)
    profile_component("layernorm", lambda: ln(x))

    # Linear projection (single GEMM)
    W = np.random.randn(d_model, d_model).astype(np.float32)
    profile_component("linear (GEMM d×d)", lambda: x @ W)

    # QKV fused projection
    W_qkv = np.random.randn(d_model, 3 * d_model).astype(np.float32)
    profile_component("QKV fused GEMM", lambda: x @ W_qkv)

    # FFN first layer
    W_ff1 = np.random.randn(d_model, d_ff).astype(np.float32)
    profile_component("FFN linear (d→4d)", lambda: x @ W_ff1)

    # ── Composite components ─────────────────────────────────────────────
    print("\n  Composite components:")

    mha = MultiHeadSelfAttention(d_model, num_heads)
    profile_component("multi-head attention", lambda: mha(x))

    ffn = FeedForward(d_model, d_ff)
    profile_component("feed-forward network", lambda: ffn(x))

    # ── Full encoder block ───────────────────────────────────────────────
    print("\n  Full encoder block:")

    from encoder.encoder import EncoderBlock
    block = EncoderBlock(d_model, num_heads, d_ff)
    total = profile_component("encoder block (1 layer)", lambda: block(x))

    # ── Full encoder (6 layers) ──────────────────────────────────────────
    print("\n  Full encoder stack:")

    enc = TransformerEncoder(
        vocab_size=10000, d_model=d_model, num_heads=num_heads,
        num_layers=6, d_ff=d_ff,
    )
    ids = np.random.randint(0, 10000, (batch_size, seq_len))
    full_time = profile_component("full encoder (6 layers)", lambda: enc(ids), iterations=100)

    # ── Summary ──────────────────────────────────────────────────────────
    print("\n  Breakdown (estimated % of encoder block time):")
    attn_time = profile_component("  → attention", lambda: mha(x), iterations=200)
    ffn_time = profile_component("  → FFN", lambda: ffn(x), iterations=200)
    ln_time = profile_component("  → 2× layernorm", lambda: (ln(x), ln(x)), iterations=200)

    total_est = attn_time + ffn_time + ln_time
    print(f"\n    Attention:  {attn_time/total_est*100:.1f}%")
    print(f"    FFN:        {ffn_time/total_est*100:.1f}%")
    print(f"    LayerNorm:  {ln_time/total_est*100:.1f}%")
    print(f"\n    (On GPU, attention dominates due to O(n²) score computation)")

    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()
