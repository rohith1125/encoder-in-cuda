"""
Low-level "kernel" implementations — NumPy functions that mirror
the exact computation pattern of their CUDA counterparts.

Each function documents:
  - Grid/block dimensions
  - Shared memory usage
  - Memory access pattern
  - Warp-level primitives used

These are the building blocks that attention.py, layernorm.py, etc. call.
Separated here to make the CUDA mapping crystal clear.
"""

import numpy as np


# ═══════════════════════════════════════════════════════════════════════════════
# KERNEL 1: Tiled Matrix Multiplication
# ═══════════════════════════════════════════════════════════════════════════════
#
# CUDA kernel:
#   Grid:  (ceil(M/TILE), ceil(N/TILE))
#   Block: (TILE, TILE) — typically 16×16 or 32×32
#   Shared memory: 2 × TILE × TILE × sizeof(float)
#
#   Each thread block loads a TILE×TILE submatrix of A and B into shared memory,
#   computes partial dot products, then accumulates across tiles along the K axis.
#   This reduces global memory accesses from O(MNK) to O(MNK / TILE).
#
def tiled_matmul(A: np.ndarray, B: np.ndarray, tile_size: int = 32) -> np.ndarray:
    """Simulates tiled GEMM — processes the multiplication in blocks.

    In real CUDA:
      - Each tile is loaded into __shared__ memory
      - Threads within a block cooperatively load and compute
      - __syncthreads() between tile loads
      - Results accumulated in registers

    Args:
        A: (..., M, K)
        B: (..., K, N)
        tile_size: simulated tile dimension

    Returns:
        C: (..., M, N)
    """
    M, K = A.shape[-2], A.shape[-1]
    N = B.shape[-1]

    batch_shape = A.shape[:-2]
    C = np.zeros(batch_shape + (M, N), dtype=A.dtype)

    # Tile along K dimension (the reduction axis)
    for k_start in range(0, K, tile_size):
        k_end = min(k_start + tile_size, K)

        # In CUDA: cooperative load of A_tile and B_tile into shared memory
        A_tile = A[..., :, k_start:k_end]  # (..., M, tile)
        B_tile = B[..., k_start:k_end, :]  # (..., tile, N)

        # In CUDA: each thread computes one element of the partial product
        # using values from shared memory (fast, no bank conflicts if padded)
        C += A_tile @ B_tile  # accumulate partial products

    return C


# ═══════════════════════════════════════════════════════════════════════════════
# KERNEL 2: Online Softmax (Numerically Stable)
# ═══════════════════════════════════════════════════════════════════════════════
#
# CUDA kernel (FlashAttention-style):
#   Grid:  (batch * heads * seq_len,)
#   Block: (WARP_SIZE,) or (256,)
#
#   Three-pass approach (can be fused to two passes):
#     Pass 1: Find row max using warp-level reduction (__shfl_down_sync)
#     Pass 2: Compute exp(x - max) and sum using warp reduction
#     Pass 3: Normalize: exp(x - max) / sum
#
#   FlashAttention does this in a single pass using online softmax
#   (maintaining running max and sum, correcting previous values).
#
def online_softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """Three-pass stable softmax, mirrors the CUDA warp-reduction approach.

    Pass 1: row-wise max (warp reduction in CUDA)
    Pass 2: exp and sum (warp reduction in CUDA)
    Pass 3: normalize (element-wise)
    """
    # Pass 1: max reduction — in CUDA: __shfl_down_sync for warp-level max
    row_max = np.max(x, axis=axis, keepdims=True)

    # Pass 2: exp + sum — in CUDA: parallel exp + warp-level sum reduction
    exp_x = np.exp(x - row_max)
    row_sum = np.sum(exp_x, axis=axis, keepdims=True)

    # Pass 3: normalize — fully parallel, one element per thread
    return exp_x / row_sum


# ═══════════════════════════════════════════════════════════════════════════════
# KERNEL 3: Layer Normalization (Fused)
# ═══════════════════════════════════════════════════════════════════════════════
#
# CUDA kernel:
#   Grid:  (batch * seq_len,)
#   Block: (min(d_model, 1024),)
#
#   Single fused kernel doing:
#     1. Welford's online mean+variance (one pass, numerically stable)
#     2. Normalize in-place
#     3. Scale + shift (affine parameters)
#
#   Uses warp-level reductions (__shfl_down_sync) for mean and variance.
#   No shared memory needed if d_model <= warp_size * values_per_thread.
#
def fused_layernorm(
    x: np.ndarray,
    gamma: np.ndarray,
    beta: np.ndarray,
    eps: float = 1e-6,
) -> np.ndarray:
    """Fused LayerNorm — single pass Welford's + normalize + affine.

    In CUDA: one kernel launch, no intermediate global memory writes.
    Welford's algorithm computes mean and variance in a single pass,
    which is critical for GPU performance (halves memory bandwidth).
    """
    # Welford's online algorithm — single pass over the data
    n = x.shape[-1]
    mean = np.zeros(x.shape[:-1] + (1,), dtype=x.dtype)
    M2 = np.zeros_like(mean)

    for i in range(n):
        val = x[..., i:i+1]
        delta = val - mean
        mean = mean + delta / (i + 1)
        delta2 = val - mean
        M2 = M2 + delta * delta2

    variance = M2 / n

    # Normalize + affine — fused in the same kernel
    x_norm = (x - mean) / np.sqrt(variance + eps)
    return gamma * x_norm + beta


# ═══════════════════════════════════════════════════════════════════════════════
# KERNEL 4: GELU Activation (Fused with Bias Add)
# ═══════════════════════════════════════════════════════════════════════════════
#
# CUDA kernel:
#   Grid:  (ceil(N / 256),)
#   Block: (256,)
#
#   Element-wise kernel, typically fused with the preceding bias-add
#   as a GEMM epilogue (cuBLAS supports custom epilogues).
#
#   GELU(x) = x * Φ(x) ≈ 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x³)))
#
def fused_bias_gelu(x: np.ndarray, bias: np.ndarray) -> np.ndarray:
    """Fused bias-add + GELU — single kernel in CUDA.

    On GPU, this avoids a separate global memory write for the bias-add
    intermediate result. The GEMM output goes directly into registers,
    bias is added, GELU is applied, and the result is written once.
    """
    x = x + bias  # bias add (fused — no separate write)
    return 0.5 * x * (1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (x + 0.044715 * x ** 3)))


# ═══════════════════════════════════════════════════════════════════════════════
# KERNEL 5: Residual Add (Fused with LayerNorm)
# ═══════════════════════════════════════════════════════════════════════════════
#
# CUDA kernel:
#   In optimized implementations (e.g., Megatron-LM), the residual add
#   is fused with the subsequent LayerNorm to avoid an extra memory pass:
#
#   fused_residual_layernorm(residual, x, gamma, beta):
#     y = residual + x          ← element-wise add
#     return layernorm(y, gamma, beta)  ← immediate normalize
#
#   One kernel, one read of residual + x, one write of normalized output.
#
def fused_residual_layernorm(
    residual: np.ndarray,
    x: np.ndarray,
    gamma: np.ndarray,
    beta: np.ndarray,
    eps: float = 1e-6,
) -> np.ndarray:
    """Fused residual-add + LayerNorm — single kernel in CUDA.

    Avoids writing the intermediate (residual + x) to global memory.
    """
    y = residual + x
    mean = np.mean(y, axis=-1, keepdims=True)
    var = np.var(y, axis=-1, keepdims=True)
    return gamma * (y - mean) / np.sqrt(var + eps) + beta


# ═══════════════════════════════════════════════════════════════════════════════
# KERNEL 6: Causal Mask Generation
# ═══════════════════════════════════════════════════════════════════════════════
#
# CUDA kernel:
#   Simple element-wise kernel: mask[i][j] = (j <= i) ? 1 : 0
#   Often precomputed and stored in constant memory.
#
def causal_mask(seq_len: int) -> np.ndarray:
    """Generate causal (autoregressive) attention mask.

    Returns a lower-triangular matrix where position i can only
    attend to positions <= i.

    Shape: (1, 1, seq_len, seq_len) for broadcasting.
    """
    mask = np.tril(np.ones((seq_len, seq_len), dtype=np.float32))
    return mask[np.newaxis, np.newaxis, :, :]  # (1, 1, S, S)


def padding_mask(lengths: np.ndarray, max_len: int) -> np.ndarray:
    """Generate padding mask from sequence lengths.

    Args:
        lengths: (batch,) — actual length of each sequence
        max_len: maximum sequence length

    Returns:
        mask: (batch, 1, 1, max_len) — 1 for valid, 0 for padding
    """
    arange = np.arange(max_len)[np.newaxis, :]  # (1, max_len)
    mask = (arange < lengths[:, np.newaxis]).astype(np.float32)
    return mask[:, np.newaxis, np.newaxis, :]  # (B, 1, 1, S)
