import torch
import triton
import triton.language as tl


@triton.jit
def softmax_fwd_kernel(
    output_ptr,
    input_ptr,
    input_batch_stride,
    input_row_stride,
    output_batch_stride,
    n_cols,
    BLOCK_SIZE: tl.constexpr,
):
    """Performs fused softmax operation over last dimension."""

    row_pid = tl.program_id(axis=0)
    batch_pid = tl.program_id(axis=1)

    row_start_ptr = (
        input_ptr + (input_batch_stride * batch_pid) + row_pid * input_row_stride
    )
    col_offsets = tl.arange(0, BLOCK_SIZE)
    input_ptrs = row_start_ptr + col_offsets
    row = tl.load(input_ptrs, mask=col_offsets < n_cols, other=-float("inf"))
    rowmax = tl.max(row, axis=0)
    unnormalized = tl.exp(row - rowmax)
    result = unnormalized / tl.sum(unnormalized, axis=0)
    output_batch_start_ptr = (
        output_ptr + (input_batch_stride * batch_pid) + row_pid * output_batch_stride
    )
    output_ptrs = output_batch_start_ptr + col_offsets
    tl.store(output_ptrs, result, mask=col_offsets < n_cols)


def softmax_fwd(x: torch.Tensor) -> torch.Tensor:
    n_b, n_rows, n_cols = x.shape
    BLOCK_SIZE = triton.next_power_of_2(n_cols)
    output = torch.empty_like(x)
    num_warps = 4 if BLOCK_SIZE < 2048 else 8
    assert x.is_cuda and output.is_cuda

    softmax_fwd_kernel[(n_rows, n_b)](
        output,
        x,
        x.stride(0),
        x.stride(1),
        output.stride(1),
        n_cols,
        num_warps=num_warps,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    return output


@triton.jit
def softmax_bwd_kernel(
    grad_out_ptr,
    out_ptr,
    grad_in_ptr,
    batch_stride,
    row_stride,
    n_cols,
    BLOCK_SIZE: tl.constexpr,
):
    """Backward Kernel."""
    row_pid = tl.program_id(axis=0)
    batch_pid = tl.program_id(axis=1)

    g_row_start_ptr = grad_out_ptr + (batch_stride * batch_pid) + row_pid * row_stride
    out_row_start_ptr = out_ptr + (batch_stride * batch_pid) + row_pid * row_stride

    col_offsets = tl.arange(0, BLOCK_SIZE)

    grad_ptrs = g_row_start_ptr + col_offsets
    out_ptrs = out_row_start_ptr + col_offsets

    grad = tl.load(grad_ptrs, mask=col_offsets < n_cols, other=0.0)

    out = tl.load(out_ptrs, mask=col_offsets < n_cols, other=0.0)

    prod = out * grad
    tmp = grad - tl.sum(prod, axis=0)
    result = out * tmp

    grad_in_row_start_ptr = (
        grad_in_ptr + (batch_stride * batch_pid) + row_pid * row_stride
    )

    grad_in_ptrs = grad_in_row_start_ptr + col_offsets
    tl.store(grad_in_ptrs, result, mask=col_offsets < n_cols)


def softmax_bwd(grad_out: torch.Tensor, saved_out: torch.Tensor) -> torch.Tensor:
    """Performs an in-place gradient update for softmax."""
    n_b, n_rows, n_cols = grad_out.shape
    BLOCK_SIZE = triton.next_power_of_2(n_cols)
    num_warps = 4 if BLOCK_SIZE < 2048 else 8

    softmax_bwd_kernel[(n_rows, n_b)](
        grad_out,
        saved_out,
        grad_out,
        grad_out.stride(0),
        grad_out.stride(1),
        n_cols,
        num_warps=num_warps,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    return grad_out
