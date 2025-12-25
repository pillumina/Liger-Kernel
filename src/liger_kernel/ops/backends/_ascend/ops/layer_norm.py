import math
import operator

import torch
import triton
import triton.language as tl
import triton.runtime.driver as driver

from liger_kernel.ops.utils import calculate_settings
from liger_kernel.ops.utils import compare_version
from liger_kernel.ops.utils import ensure_contiguous
from liger_kernel.utils import is_npu_available

from triton.language.math import rsqrt


# ============================================================================
# Ascend NPU Properties
# ============================================================================

def get_npu_properties():
    """
    Get NPU device properties.
    Compatible with both torch_npu and triton driver.
    """
    try:
        if hasattr(torch, 'npu') and torch.npu.is_available():
            device = torch.npu.current_device()
            return driver.active.utils.get_device_properties(device)
    except Exception:
        pass
    # Fallback to default
    return {"num_vectorcore": 40, "num_aicore": 40}


def get_vector_core_count():
    """
    Get the number of vector cores for the current Ascend device.
    LayerNorm is a pure vector computation, so we use num_vectorcore.
    """
    props = get_npu_properties()
    return props.get("num_vectorcore", 40)  # Default to 40 for 910B3/910B4


# ============================================================================
# Constants based on triton-ascend HighPerformanceGuide.md
# ============================================================================

# Sub-block size: single computation size within one core
# Based on 192KB on-chip memory / 4 bytes per float32 = 49152
# Reserve space for intermediate variables: 49152 * 0.25 ≈ 12288
# Conservative value: 8192 (as recommended in the guide)
DEFAULT_SUB_BLOCK_SIZE = 8192

# Maximum sub-block size (should not exceed on-chip memory)
MAX_SUB_BLOCK_SIZE = 12288


# ============================================================================
# Forward Kernels
# ============================================================================

@triton.jit
def _layer_norm_forward_kernel(
    Y_ptr,           # pointer to output
    X_ptr,           # pointer to input
    W_ptr,           # pointer to weights
    B_ptr,           # pointer to bias
    Mean_ptr,        # pointer to mean
    RSTD_ptr,        # pointer to rstd
    n_cols,          # number of columns
    xnumel,          # total number of elements
    eps,             # epsilon for numerical stability
    XBLOCK: tl.constexpr,      # inter-core data partitioning (per vector core)
    XBLOCK_SUB: tl.constexpr,  # intra-core data partitioning (per computation)
):
    """
    LayerNorm forward kernel with XBLOCK/XBLOCK_SUB pattern.

    Memory access pattern (following triton-ascend HighPerformanceGuide.md):
    1. Inter-core partitioning: each vector core processes XBLOCK elements
    2. Intra-core partitioning: each core loops over data in XBLOCK_SUB chunks

    Reference: https://github.com/yuanpeng-2022/triton-ascend/blob/main/docs/HighPerformanceGuide.md
    """
    # === Inter-core partitioning: calculate offset for this core ===
    xoffset = tl.program_id(0) * XBLOCK

    # Initialize accumulators for mean and variance
    sum_x = tl.zeros((), dtype=tl.float32)
    sum_sq = tl.zeros((), dtype=tl.float32)

    # === First pass: compute sum and sum of squares ===
    # Intra-core partitioning: loop over data in XBLOCK_SUB chunks
    for xoffset_sub in range(0, XBLOCK, XBLOCK_SUB):
        # Calculate indices for this sub-block
        x_index = xoffset + xoffset_sub + tl.arange(0, XBLOCK_SUB)[:]

        # Mask to prevent out-of-bounds access
        xmask = x_index < xnumel

        # Load input data from global memory to on-chip memory
        x = tl.load(X_ptr + x_index, xmask, other=0.0).to(tl.float32)

        # Accumulate sum and sum of squares
        sum_x += tl.sum(x, axis=0)
        sum_sq += tl.sum(x * x, axis=0)

    # Compute mean and variance from accumulated sums
    mean = sum_x / n_cols
    var = (sum_sq / n_cols) - (mean * mean)
    rstd = rsqrt(var + eps)

    # Store mean and rstd for backward pass
    row_id = tl.program_id(0)
    tl.store(Mean_ptr + row_id, mean)
    tl.store(RSTD_ptr + row_id, rstd)

    # === Second pass: apply normalization and affine transformation ===
    for xoffset_sub in range(0, XBLOCK, XBLOCK_SUB):
        x_index = xoffset + xoffset_sub + tl.arange(0, XBLOCK_SUB)[:]
        xmask = x_index < xnumel

        # Load input data
        x = tl.load(X_ptr + x_index, xmask, other=0.0).to(tl.float32)

        # Load weights and bias (column-wise, need to map to correct position)
        # Calculate column index within the row
        col_index = x_index % n_cols
        w = tl.load(W_ptr + col_index, xmask, other=0.0).to(tl.float32)
        b = tl.load(B_ptr + col_index, xmask, other=0.0).to(tl.float32)

        # Apply layer normalization
        x_hat = (x - mean) * rstd
        y = w * x_hat + b

        # Store output
        tl.store(Y_ptr + x_index, y.to(Y_ptr.dtype.element_ty), xmask)


@triton.jit
def _layer_norm_backward_kernel_dx(
    X_ptr,
    DY_ptr,
    W_ptr,
    Mean_ptr,
    RSTD_ptr,
    DX_ptr,
    n_cols,
    xnumel,
    XBLOCK: tl.constexpr,
    XBLOCK_SUB: tl.constexpr,
):
    """
    LayerNorm backward kernel for computing dX (input gradient).
    Uses XBLOCK/XBLOCK_SUB pattern.
    """
    # Inter-core partitioning
    xoffset = tl.program_id(0) * XBLOCK

    # Load mean and rstd for this row
    row_id = tl.program_id(0)
    mean = tl.load(Mean_ptr + row_id).to(tl.float32)
    rstd = tl.load(RSTD_ptr + row_id).to(tl.float32)

    # Initialize accumulators for backward terms
    c1_accum = tl.zeros((), dtype=tl.float32)
    c2_accum = tl.zeros((), dtype=tl.float32)

    # === First pass: compute c1 and c2 ===
    for xoffset_sub in range(0, XBLOCK, XBLOCK_SUB):
        x_index = xoffset + xoffset_sub + tl.arange(0, XBLOCK_SUB)[:]
        xmask = x_index < xnumel

        x = tl.load(X_ptr + x_index, xmask, other=0.0).to(tl.float32)
        dy = tl.load(DY_ptr + x_index, xmask, other=0.0).to(tl.float32)
        col_index = x_index % n_cols
        w = tl.load(W_ptr + col_index, xmask, other=0.0).to(tl.float32)

        x_hat = (x - mean) * rstd
        wdy = w * dy

        c1_accum += tl.sum(x_hat * wdy, axis=0)
        c2_accum += tl.sum(wdy, axis=0)

    c1 = c1_accum / n_cols
    c2 = c2_accum / n_cols

    # === Second pass: compute dx ===
    for xoffset_sub in range(0, XBLOCK, XBLOCK_SUB):
        x_index = xoffset + xoffset_sub + tl.arange(0, XBLOCK_SUB)[:]
        xmask = x_index < xnumel

        x = tl.load(X_ptr + x_index, xmask, other=0.0).to(tl.float32)
        dy = tl.load(DY_ptr + x_index, xmask, other=0.0).to(tl.float32)
        col_index = x_index % n_cols
        w = tl.load(W_ptr + col_index, xmask, other=0.0).to(tl.float32)

        x_hat = (x - mean) * rstd
        wdy = w * dy
        dx = (wdy - (x_hat * c1 + c2)) * rstd

        tl.store(DX_ptr + x_index, dx, xmask)


@triton.jit
def _layer_norm_backward_kernel_dw_db(
    X_ptr,
    DY_ptr,
    Mean_ptr,
    RSTD_ptr,
    DW_ptr,
    DB_ptr,
    n_cols,
    n_rows,
    YBLOCK: tl.constexpr,
    YBLOCK_SUB: tl.constexpr,
):
    """
    LayerNorm backward kernel for computing dW and dB (weight and bias gradients).
    Uses 2D tiling with YBLOCK/YBLOCK_SUB pattern.
    """
    # Inter-core partitioning
    yoffset = tl.program_id(0) * YBLOCK

    # Initialize accumulators for this core's portion
    dw_partial = tl.zeros((n_cols,), dtype=tl.float32)
    db_partial = tl.zeros((n_cols,), dtype=tl.float32)

    # Loop over rows assigned to this core
    for yoffset_sub in range(0, YBLOCK, YBLOCK_SUB):
        row_index = yoffset + yoffset_sub + tl.arange(0, YBLOCK_SUB)[:]
        row_mask = row_index < n_rows

        # Broadcast row_index to 2D for accessing column data
        row_idx_2d = row_index[:, None]
        col_idx = tl.arange(0, n_cols)[None, :]

        # Calculate flat indices
        x_index = row_idx_2d * n_cols + col_idx
        mask = row_mask[:, None] & tl.const_array([True], dtype=tl.int1)

        x = tl.load(X_ptr + x_index, mask, other=0.0).to(tl.float32)
        dy = tl.load(DY_ptr + x_index, mask, other=0.0).to(tl.float32)
        mean = tl.load(Mean_ptr + row_index, row_mask, other=0.0).to(tl.float32)[:, None]
        rstd = tl.load(RSTD_ptr + row_index, row_mask, other=0.0).to(tl.float32)[:, None]

        x_hat = (x - mean) * rstd

        # Accumulate gradients
        dw_partial += tl.sum(dy * x_hat * rstd, axis=0)
        db_partial += tl.sum(dy, axis=0)

    # Store partial results (will be summed across cores)
    core_id = tl.program_id(0)
    tl.store(DW_ptr + core_id * n_cols + tl.arange(0, n_cols), dw_partial)
    tl.store(DB_ptr + core_id * n_cols + tl.arange(0, n_cols), db_partial)


# ============================================================================
# Forward/Backward Functions
# ============================================================================

def layer_norm_forward(X, W, B, eps):
    """
    LayerNorm forward pass using triton-ascend recommended pattern.

    Args:
        X: Input tensor of shape (..., hidden_size)
        W: Weight tensor of shape (hidden_size,)
        B: Bias tensor of shape (hidden_size,)
        eps: Small constant for numerical stability

    Returns:
        Tuple of (output, input, mean, rstd, block_size)
    """
    shape = X.shape
    dim = shape[-1]
    X = X.view(-1, dim)
    n_rows, n_cols = X.shape
    xnumel = X.numel()

    # === Get vector core count ===
    # LayerNorm is pure vector computation, so use num_vectorcore
    num_cores = get_vector_core_count()

    # === Calculate XBLOCK (data per core) ===
    # Each core processes: total_elements / num_cores
    # Ensure at least 1 element per core
    XBLOCK = max(1, triton.cdiv(xnumel, num_cores))

    # === Calculate XBLOCK_SUB (sub-block size) ===
    # Based on triton-ascend guide: sub_block_size = 8192
    # This ensures data fits in on-chip memory (192KB L1/UB cache)
    XBLOCK_SUB = min(DEFAULT_SUB_BLOCK_SIZE, triton.next_power_of_2(XBLOCK))

    # Allocate output tensors
    Y = torch.empty_like(X)
    Mean = torch.empty(n_rows, dtype=torch.float32, device=X.device)
    RSTD = torch.empty(n_rows, dtype=torch.float32, device=X.device)

    # Validate input dimensions
    if X.shape[1] != W.shape[0]:
        raise ValueError(
            f"Incompatible dimensions: input feature size (X.shape[1]={X.shape[1]}) "
            f"must match weight size (W.shape[0]={W.shape[0]})"
        )

    # === Launch kernel ===
    # Grid size = number of vector cores (as recommended in triton-ascend guide)
    grid = (num_cores, 1, 1)
    _layer_norm_forward_kernel[grid](
        Y,
        X,
        W,
        B,
        Mean,
        RSTD,
        n_cols,
        xnumel,
        eps,
        XBLOCK=XBLOCK,
        XBLOCK_SUB=XBLOCK_SUB,
    )

    return Y.view(*shape), X, Mean, RSTD, XBLOCK


def layer_norm_backward(dY, X, W, B, Mean, RSTD):
    """
    LayerNorm backward pass using triton-ascend recommended pattern.

    Args:
        dY: Gradient of output
        X: Input tensor
        W: Weight tensor
        B: Bias tensor
        Mean: Pre-computed mean
        RSTD: Pre-computed reciprocal standard deviation

    Returns:
        Tuple of (input_grad, weight_grad, bias_grad)
    """
    shape = dY.shape
    dim = shape[-1]
    dY = dY.view(-1, dim)
    n_rows, n_cols = dY.shape
    xnumel = X.numel()

    # === Get vector core count ===
    num_cores = get_vector_core_count()

    # === Calculate XBLOCK for dx computation ===
    XBLOCK = max(1, triton.cdiv(xnumel, num_cores))
    XBLOCK_SUB = min(DEFAULT_SUB_BLOCK_SIZE, triton.next_power_of_2(XBLOCK))

    # === Allocate gradient tensors ===
    DX = torch.empty_like(X)

    # Partial accumulators for weight/bias gradients (one per core)
    _DW = torch.empty((num_cores, n_cols), dtype=torch.float32, device=W.device)
    _DB = torch.empty((num_cores, n_cols), dtype=torch.float32, device=W.device)

    # === Launch dx kernel ===
    grid = (num_cores, 1, 1)
    _layer_norm_backward_kernel_dx[grid](
        X,
        dY,
        W,
        Mean,
        RSTD,
        DX,
        n_cols,
        xnumel,
        XBLOCK=XBLOCK,
        XBLOCK_SUB=XBLOCK_SUB,
    )

    # === Launch dw/db kernel ===
    # For dw/db, we process rows in parallel
    YBLOCK = max(1, triton.cdiv(n_rows, num_cores))
    YBLOCK_SUB = min(256, triton.next_power_of_2(YBLOCK))

    _layer_norm_backward_kernel_dw_db[grid](
        X,
        dY,
        Mean,
        RSTD,
        _DW,
        _DB,
        n_cols,
        n_rows,
        YBLOCK=YBLOCK,
        YBLOCK_SUB=YBLOCK_SUB,
    )

    # Sum partial gradients across cores
    DX = DX.view(*shape)
    DW = _DW.sum(dim=0).to(W.dtype)
    DB = _DB.sum(dim=0).to(B.dtype)

    return DX, DW, DB


class LigerLayerNormFunction(torch.autograd.Function):
    @staticmethod
    @ensure_contiguous
    def forward(ctx, X, W, B, eps):
        Y, X, Mean, RSTD, XBLOCK = layer_norm_forward(X, W, B, eps)
        ctx.save_for_backward(X, W, B, Mean, RSTD)
        return Y

    @staticmethod
    @ensure_contiguous
    def backward(ctx, dY):
        X, W, B, Mean, RSTD = ctx.saved_tensors
        DX, DW, DB = layer_norm_backward(dY, X, W, B, Mean, RSTD)
        return DX, DW, DB, None
