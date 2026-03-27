# Transformer Encoder Block — From Scratch

> Every matrix operation hand-written in pure Python + NumPy. No PyTorch. No TensorFlow. No frameworks.

## What This Is

A complete Transformer encoder implementation built from first principles. Each component mirrors what a CUDA kernel would do on a GPU — documented inline with the GPU-parallel equivalent of every operation.

## Architecture

```
Token IDs ──► Embedding (gather) ──► + Positional Encoding (sinusoidal)
                                          │
                                          ▼
                              ┌─── Encoder Block ×N ───┐
                              │                        │
                              │  LayerNorm             │
                              │      ▼                 │
                              │  Multi-Head Attention   │
                              │  (QKV GEMM → scores    │
                              │   → softmax → output)  │
                              │      ▼                 │
                              │  + Residual            │
                              │      ▼                 │
                              │  LayerNorm             │
                              │      ▼                 │
                              │  Feed-Forward          │
                              │  (Linear → GELU        │
                              │   → Linear)            │
                              │      ▼                 │
                              │  + Residual            │
                              └────────────────────────┘
                                          │
                                          ▼
                                    Final LayerNorm
                                          │
                                          ▼
                                  Output Representations
```

## Components

| Module | File | CUDA Kernel Equivalent |
|--------|------|----------------------|
| Scaled Dot-Product Attention | `encoder/attention.py` | Batched GEMM + online softmax + GEMM |
| Multi-Head Self-Attention | `encoder/attention.py` | Fused QKV GEMM + head-parallel attention |
| Layer Normalization | `encoder/layernorm.py` | Fused mean/var reduction + normalize + affine |
| Feed-Forward Network | `encoder/feedforward.py` | 2× tiled GEMM with fused GELU |
| Positional Encoding | `encoder/positional.py` | Precomputed constant memory lookup |
| Token Embedding | `encoder/embedding.py` | Gather from global memory |
| Encoder Block | `encoder/encoder.py` | ~8 kernel launches per block |

## Quick Start

```bash
# Run the demo
python demo.py

# Run tests
python -m pytest tests/ -v
```

## Requirements

- Python 3.10+
- NumPy

```bash
pip install numpy pytest
```

## Project Structure

```
encoder-in-cuda/
├── encoder/
│   ├── __init__.py          # Public API
│   ├── attention.py         # Softmax, scaled dot-product, multi-head attention
│   ├── layernorm.py         # Layer normalization
│   ├── feedforward.py       # Position-wise FFN with GELU
│   ├── positional.py        # Sinusoidal positional encoding
│   ├── embedding.py         # Token embedding lookup
│   └── encoder.py           # Encoder block + full encoder stack
├── tests/
│   └── test_components.py   # Unit tests for every component
├── demo.py                  # End-to-end demo
└── README.md
```

## Design Decisions

- **Pre-LN architecture** (GPT-2 / modern style) — more stable than Post-LN
- **GELU activation** — standard in BERT/GPT, smoother gradients than ReLU
- **Xavier initialization** — prevents vanishing/exploding activations
- **Numerically stable softmax** — subtract-max trick, same as GPU online softmax
- **No autograd** — forward pass only, focuses on the compute graph structure

## CUDA Mapping

Every function in this codebase has inline comments explaining what the equivalent CUDA kernel would look like — thread block assignments, shared memory tiling, warp-level reductions, kernel fusion opportunities, and memory access patterns.
