#!/usr/bin/env python3
"""
Attention Pattern Visualization — renders attention weight heatmaps.

Shows what each head attends to at each layer, revealing patterns like:
  - Positional attention (diagonal patterns)
  - Global attention (uniform rows — "summary" tokens)
  - Local attention (band patterns)

Usage:
    pip install matplotlib
    python benchmarks/visualize_attention.py
"""

import sys
import numpy as np

try:
    import matplotlib
    matplotlib.use("Agg")  # non-interactive backend
    import matplotlib.pyplot as plt
except ImportError:
    print("matplotlib not installed. Run: pip install matplotlib")
    sys.exit(0)

from encoder import TransformerEncoder


def plot_attention_heads(
    attention_weights: list[np.ndarray],
    layer_idx: int = 0,
    batch_idx: int = 0,
    save_path: str = "attention_heads.png",
):
    """Plot all attention heads for a given layer."""
    weights = attention_weights[layer_idx][batch_idx]  # (heads, seq, seq)
    num_heads = weights.shape[0]

    cols = 4
    rows = (num_heads + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows))
    fig.suptitle(f"Layer {layer_idx} — Attention Heads", fontsize=16, fontweight="bold")

    for i in range(num_heads):
        ax = axes[i // cols][i % cols] if rows > 1 else axes[i % cols]
        im = ax.imshow(weights[i], cmap="viridis", aspect="auto", vmin=0, vmax=0.3)
        ax.set_title(f"Head {i}", fontsize=11)
        ax.set_xlabel("Key position")
        ax.set_ylabel("Query position")

    # Remove empty subplots
    for i in range(num_heads, rows * cols):
        ax = axes[i // cols][i % cols] if rows > 1 else axes[i % cols]
        ax.axis("off")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"    Saved: {save_path}")
    plt.close()


def plot_attention_across_layers(
    attention_weights: list[np.ndarray],
    head_idx: int = 0,
    batch_idx: int = 0,
    save_path: str = "attention_layers.png",
):
    """Plot attention for one head across all layers."""
    num_layers = len(attention_weights)
    cols = min(num_layers, 6)
    rows = (num_layers + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows))
    fig.suptitle(f"Head {head_idx} — Across Layers", fontsize=16, fontweight="bold")

    if rows == 1:
        axes = [axes]

    for i in range(num_layers):
        ax = axes[i // cols][i % cols] if rows > 1 else axes[0][i % cols]
        w = attention_weights[i][batch_idx, head_idx]
        ax.imshow(w, cmap="magma", aspect="auto", vmin=0, vmax=0.3)
        ax.set_title(f"Layer {i}", fontsize=11)

        # Compute entropy for this head/layer
        entropy = -np.sum(w * np.log2(w.clip(1e-10)), axis=-1).mean()
        ax.set_xlabel(f"entropy={entropy:.2f} bits")

    for i in range(num_layers, rows * cols):
        ax = axes[i // cols][i % cols] if rows > 1 else axes[0][i % cols]
        ax.axis("off")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"    Saved: {save_path}")
    plt.close()


def plot_attention_entropy(
    attention_weights: list[np.ndarray],
    batch_idx: int = 0,
    save_path: str = "attention_entropy.png",
):
    """Plot entropy of attention distributions across layers and heads."""
    num_layers = len(attention_weights)
    num_heads = attention_weights[0].shape[1]

    entropies = np.zeros((num_layers, num_heads))
    for l in range(num_layers):
        for h in range(num_heads):
            w = attention_weights[l][batch_idx, h]
            entropies[l, h] = -np.sum(w * np.log2(w.clip(1e-10)), axis=-1).mean()

    fig, ax = plt.subplots(figsize=(10, 5))
    im = ax.imshow(entropies.T, cmap="RdYlBu_r", aspect="auto")
    ax.set_xlabel("Layer", fontsize=12)
    ax.set_ylabel("Head", fontsize=12)
    ax.set_title("Attention Entropy (bits) — Higher = More Uniform", fontsize=14)
    ax.set_xticks(range(num_layers))
    ax.set_yticks(range(num_heads))
    plt.colorbar(im, ax=ax, label="Entropy (bits)")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"    Saved: {save_path}")
    plt.close()


def main():
    print("=" * 70)
    print("  Attention Visualization")
    print("=" * 70)

    # Build encoder
    enc = TransformerEncoder(
        vocab_size=10000, d_model=256, num_heads=8, num_layers=6, seed=42,
    )

    # Generate random input
    rng = np.random.default_rng(0)
    ids = rng.integers(0, 10000, size=(2, 32))

    # Forward pass
    output, attn_weights = enc(ids)

    print("\n  Generating visualizations...")

    # Plot attention heads for first layer
    plot_attention_heads(attn_weights, layer_idx=0, save_path="attention_heads_layer0.png")

    # Plot attention heads for last layer
    plot_attention_heads(attn_weights, layer_idx=5, save_path="attention_heads_layer5.png")

    # Plot one head across all layers
    plot_attention_across_layers(attn_weights, head_idx=0, save_path="attention_across_layers.png")

    # Plot entropy heatmap
    plot_attention_entropy(attn_weights, save_path="attention_entropy.png")

    print("\n" + "=" * 70)
    print("  All visualizations saved.")
    print("=" * 70)


if __name__ == "__main__":
    main()
