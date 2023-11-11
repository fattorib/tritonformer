""" 
Matrix-multiplication kernels for cases:
    1. (b,m,n) @ (n,k) -> (b,m,k): `matmul`
    2. (b,m,n) @ (n,k) -> (b,m,k) + fused activation: `matmul`
    3. (b,m,n) @ (b,n,k) -> (b,m,k): `bmm`
    4. (b,m,n) @ (n,k)-> (b,m,k) + (k,): 'gemm'     
    5. (b,m,n) @ (n,k)-> (b,m,k) + (k,) + fused activation: 'gemm'
"""

import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config(
            {
                "BLOCK_SIZE_M": 128,
                "BLOCK_SIZE_N": 32,
                "BLOCK_SIZE_K": 128,
                "GROUP_SIZE_M": 8,
            },
            num_stages=1,
            num_warps=8,
        ),
        triton.Config(
            {
                "BLOCK_SIZE_M": 128,
                "BLOCK_SIZE_N": 32,
                "BLOCK_SIZE_K": 128,
                "GROUP_SIZE_M": 8,
            },
            num_stages=2,
            num_warps=8,
        ),
        triton.Config(
            {
                "BLOCK_SIZE_M": 128,
                "BLOCK_SIZE_N": 32,
                "BLOCK_SIZE_K": 128,
                "GROUP_SIZE_M": 8,
            },
            num_stages=3,
            num_warps=8,
        ),
        triton.Config(
            {
                "BLOCK_SIZE_M": 128,
                "BLOCK_SIZE_N": 64,
                "BLOCK_SIZE_K": 128,
                "GROUP_SIZE_M": 8,
            },
            num_stages=1,
            num_warps=8,
        ),
        triton.Config(
            {
                "BLOCK_SIZE_M": 128,
                "BLOCK_SIZE_N": 64,
                "BLOCK_SIZE_K": 128,
                "GROUP_SIZE_M": 8,
            },
            num_stages=2,
            num_warps=8,
        ),
        triton.Config(
            {
                "BLOCK_SIZE_M": 128,
                "BLOCK_SIZE_N": 64,
                "BLOCK_SIZE_K": 128,
                "GROUP_SIZE_M": 8,
            },
            num_stages=3,
            num_warps=8,
        ),
    ],
    key=["dim_m", "dim_n", "dim_k"],
)
@triton.jit
def bmm_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    abs_stride,
    arow_stride,
    acol_stride,
    bbs_stride,
    brow_stride,
    bcol_stride,
    cbs_stride,
    crow_stride,
    ccol_stride,
    dim_m,
    dim_n,
    dim_k,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
):
    pid = tl.program_id(0)
    bs_pid = tl.program_id(axis=1)

    num_pid_row = tl.cdiv(dim_m, BLOCK_SIZE_M)
    num_pid_col = tl.cdiv(dim_k, BLOCK_SIZE_K)

    num_pid_in_group = GROUP_SIZE_M * num_pid_col
    group_id = pid // num_pid_in_group
    first_pid_row = group_id * GROUP_SIZE_M
    group_size_row = min(num_pid_row - first_pid_row, GROUP_SIZE_M)
    pid_row = first_pid_row + (pid % group_size_row)
    pid_col = (pid % num_pid_in_group) // group_size_row

    acc = tl.zeros([BLOCK_SIZE_M, BLOCK_SIZE_K], dtype=tl.float32)

    a_block_ptr = tl.make_block_ptr(
        a_ptr + bs_pid * abs_stride,
        shape=(dim_m, dim_n),
        strides=(arow_stride, acol_stride),
        offsets=(pid_row * BLOCK_SIZE_M, 0),
        block_shape=(BLOCK_SIZE_M, BLOCK_SIZE_N),
        order=(1, 0),
    )
    b_block_ptr = tl.make_block_ptr(
        b_ptr + bs_pid * bbs_stride,
        shape=(dim_n, dim_k),
        strides=(brow_stride, bcol_stride),
        offsets=(
            0,
            pid_col * BLOCK_SIZE_K,
        ),
        block_shape=(BLOCK_SIZE_N, BLOCK_SIZE_K),
        order=(1, 0),
    )
    c_block_ptr = tl.make_block_ptr(
        c_ptr + bs_pid * cbs_stride,
        shape=(dim_m, dim_k),
        strides=(crow_stride, ccol_stride),
        offsets=(
            pid_row * BLOCK_SIZE_M,
            pid_col * BLOCK_SIZE_K,
        ),
        block_shape=(BLOCK_SIZE_M, BLOCK_SIZE_K),
        order=(1, 0),
    )

    for n in range(0, tl.cdiv(dim_n, BLOCK_SIZE_N)):
        a_block = tl.load(a_block_ptr, boundary_check=(0, 1))
        b_block = tl.load(b_block_ptr, boundary_check=(0, 1))

        acc += tl.dot(a_block, b_block)

        a_block_ptr = tl.advance(a_block_ptr, offsets=(0, BLOCK_SIZE_N))
        b_block_ptr = tl.advance(b_block_ptr, offsets=(BLOCK_SIZE_N, 0))

    acc = acc.to(tl.float16)
    tl.store(c_block_ptr, acc, boundary_check=(0, 1))


def bmm(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    Performs a batched matrix-multiply between tensor a and b.
    - a has shape (B,M,N) and b has shape (B,N,K)
    - Returns a tensor of shape (B, M, K)
    """

    assert (
        a.shape[-1] == b.shape[-2]
    ), f"Dimension mismatch. Expected a.shape[2] ({a.shape[-1]}) to be equal to b.shape[0] ({b.shape[-2]})"
    assert a.ndim == 3 and b.ndim == 3, "Incorrect number of dimensions for LHS or RHS"

    B, M, N, K = a.shape[0], a.shape[1], a.shape[2], b.shape[2]
    c = torch.empty((B, M, K), device=a.device, dtype=a.dtype)
    assert a.is_cuda and b.is_cuda and c.is_cuda

    # this launches one kernel per output block of c
    grid = lambda META: (
        triton.cdiv(M, META["BLOCK_SIZE_M"]) * triton.cdiv(K, META["BLOCK_SIZE_K"]),
        B,
    )

    bmm_kernel[grid](
        a,
        b,
        c,
        a.stride(0),
        a.stride(1),
        a.stride(2),
        b.stride(0),
        b.stride(1),
        b.stride(2),
        c.stride(0),
        c.stride(1),
        c.stride(2),
        M,
        N,
        K,
    )
    return c


@triton.autotune(
    configs=[
        triton.Config(
            {
                "BLOCK_SIZE_M": 128,
                "BLOCK_SIZE_N": 32,
                "BLOCK_SIZE_K": 128,
                "GROUP_SIZE_M": 8,
            },
            num_stages=2,
            num_warps=8,
        ),
        triton.Config(
            {
                "BLOCK_SIZE_M": 128,
                "BLOCK_SIZE_N": 32,
                "BLOCK_SIZE_K": 128,
                "GROUP_SIZE_M": 8,
            },
            num_stages=3,
            num_warps=8,
        ),
        triton.Config(
            {
                "BLOCK_SIZE_M": 128,
                "BLOCK_SIZE_N": 32,
                "BLOCK_SIZE_K": 128,
                "GROUP_SIZE_M": 8,
            },
            num_stages=2,
            num_warps=4,
        ),
        triton.Config(
            {
                "BLOCK_SIZE_M": 128,
                "BLOCK_SIZE_N": 32,
                "BLOCK_SIZE_K": 128,
                "GROUP_SIZE_M": 8,
            },
            num_stages=3,
            num_warps=4,
        ),
        triton.Config(
            {
                "BLOCK_SIZE_M": 128,
                "BLOCK_SIZE_N": 64,
                "BLOCK_SIZE_K": 128,
                "GROUP_SIZE_M": 8,
            },
            num_stages=1,
            num_warps=8,
        ),
        triton.Config(
            {
                "BLOCK_SIZE_M": 128,
                "BLOCK_SIZE_N": 64,
                "BLOCK_SIZE_K": 128,
                "GROUP_SIZE_M": 8,
            },
            num_stages=2,
            num_warps=8,
        ),
        triton.Config(
            {
                "BLOCK_SIZE_M": 128,
                "BLOCK_SIZE_N": 64,
                "BLOCK_SIZE_K": 128,
                "GROUP_SIZE_M": 8,
            },
            num_stages=3,
            num_warps=8,
        ),
    ],
    key=["dim_m", "dim_n", "dim_k"],
)
@triton.jit
def matmul_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    abs_stride,
    arow_stride,
    acol_stride,
    brow_stride,
    bcol_stride,
    cbs_stride,
    crow_stride,
    ccol_stride,
    dim_m,
    dim_n,
    dim_k,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    fuse_relu: tl.constexpr,
):
    pid = tl.program_id(0)
    bs_pid = tl.program_id(axis=1)

    num_pid_row = tl.cdiv(dim_m, BLOCK_SIZE_M)
    num_pid_col = tl.cdiv(dim_k, BLOCK_SIZE_K)

    num_pid_in_group = GROUP_SIZE_M * num_pid_col
    group_id = pid // num_pid_in_group
    first_pid_row = group_id * GROUP_SIZE_M
    group_size_row = min(num_pid_row - first_pid_row, GROUP_SIZE_M)
    pid_row = first_pid_row + (pid % group_size_row)
    pid_col = (pid % num_pid_in_group) // group_size_row

    acc = tl.zeros([BLOCK_SIZE_M, BLOCK_SIZE_K], dtype=tl.float32)

    a_block_ptr = tl.make_block_ptr(
        a_ptr + bs_pid * abs_stride,
        shape=(dim_m, dim_n),
        strides=(arow_stride, acol_stride),
        offsets=(pid_row * BLOCK_SIZE_M, 0),
        block_shape=(BLOCK_SIZE_M, BLOCK_SIZE_N),
        order=(1, 0),
    )
    b_block_ptr = tl.make_block_ptr(
        b_ptr,
        shape=(dim_n, dim_k),
        strides=(brow_stride, bcol_stride),
        offsets=(
            0,
            pid_col * BLOCK_SIZE_K,
        ),
        block_shape=(BLOCK_SIZE_N, BLOCK_SIZE_K),
        order=(1, 0),
    )
    c_block_ptr = tl.make_block_ptr(
        c_ptr + bs_pid * cbs_stride,
        shape=(dim_m, dim_k),
        strides=(crow_stride, ccol_stride),
        offsets=(
            pid_row * BLOCK_SIZE_M,
            pid_col * BLOCK_SIZE_K,
        ),
        block_shape=(BLOCK_SIZE_M, BLOCK_SIZE_K),
        order=(1, 0),
    )

    for n in range(0, tl.cdiv(dim_n, BLOCK_SIZE_N)):
        a_block = tl.load(a_block_ptr, boundary_check=(0, 1))
        b_block = tl.load(b_block_ptr, boundary_check=(0, 1))

        acc += tl.dot(a_block, b_block, allow_tf32=False)

        a_block_ptr = tl.advance(a_block_ptr, offsets=(0, BLOCK_SIZE_N))
        b_block_ptr = tl.advance(b_block_ptr, offsets=(BLOCK_SIZE_N, 0))

    if fuse_relu:
        acc = tl.maximum(acc, 0.0)

    tl.store(c_block_ptr, acc.to(tl.float16), boundary_check=(0, 1))


def mm(a: torch.Tensor, b: torch.Tensor, fuse_relu: bool = False) -> torch.Tensor:
    """
    Performs a matrix-multiply between batched tensor a and b.
    - a has shape (B,M,N) and b has shape (N,K)
    - Returns a tensor of shape (B, M, K)

    Optionally supports fusing ReLU activation computation.
    """

    assert (
        a.shape[2] == b.shape[0]
    ), f"Dimension mismatch. Expected a.shape[2] ({a.shape[2]}) to be equal to b.shape[0] ({b.shape[0]})"
    assert a.ndim == 3 and b.ndim == 2, "Incorrect number of dimensions for LHS or RHS"

    B, M, N, K = a.shape[0], a.shape[1], a.shape[2], b.shape[1]
    c = torch.empty((B, M, K), device=a.device, dtype=a.dtype)
    assert a.is_cuda and b.is_cuda and c.is_cuda

    grid = lambda META: (
        triton.cdiv(M, META["BLOCK_SIZE_M"]) * triton.cdiv(K, META["BLOCK_SIZE_K"]),
        B,
    )

    matmul_kernel[grid](
        a,
        b,
        c,
        a.stride(0),
        a.stride(1),
        a.stride(2),
        b.stride(0),
        b.stride(1),
        c.stride(0),
        c.stride(1),
        c.stride(2),
        M,
        N,
        K,
        fuse_relu=fuse_relu,
    )
    return c


@triton.autotune(
    configs=[
        triton.Config(
            {
                "BLOCK_SIZE_M": 128,
                "BLOCK_SIZE_N": 32,
                "BLOCK_SIZE_K": 128,
                "GROUP_SIZE_M": 8,
            },
            num_stages=2,
            num_warps=8,
        ),
        triton.Config(
            {
                "BLOCK_SIZE_M": 128,
                "BLOCK_SIZE_N": 32,
                "BLOCK_SIZE_K": 128,
                "GROUP_SIZE_M": 8,
            },
            num_stages=3,
            num_warps=8,
        ),
        triton.Config(
            {
                "BLOCK_SIZE_M": 128,
                "BLOCK_SIZE_N": 32,
                "BLOCK_SIZE_K": 128,
                "GROUP_SIZE_M": 8,
            },
            num_stages=2,
            num_warps=4,
        ),
        triton.Config(
            {
                "BLOCK_SIZE_M": 128,
                "BLOCK_SIZE_N": 32,
                "BLOCK_SIZE_K": 128,
                "GROUP_SIZE_M": 8,
            },
            num_stages=3,
            num_warps=4,
        ),
        triton.Config(
            {
                "BLOCK_SIZE_M": 128,
                "BLOCK_SIZE_N": 64,
                "BLOCK_SIZE_K": 128,
                "GROUP_SIZE_M": 8,
            },
            num_stages=1,
            num_warps=8,
        ),
        triton.Config(
            {
                "BLOCK_SIZE_M": 128,
                "BLOCK_SIZE_N": 64,
                "BLOCK_SIZE_K": 128,
                "GROUP_SIZE_M": 8,
            },
            num_stages=2,
            num_warps=8,
        ),
        triton.Config(
            {
                "BLOCK_SIZE_M": 128,
                "BLOCK_SIZE_N": 64,
                "BLOCK_SIZE_K": 128,
                "GROUP_SIZE_M": 8,
            },
            num_stages=3,
            num_warps=8,
        ),
    ],
    key=["dim_m", "dim_n", "dim_k"],
)
@triton.jit
def gemm_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    bias_ptr,
    abs_stride,
    arow_stride,
    acol_stride,
    brow_stride,
    bcol_stride,
    cbs_stride,
    crow_stride,
    ccol_stride,
    dim_m,
    dim_n,
    dim_k,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    fuse_relu: tl.constexpr,
):
    pid = tl.program_id(0)
    bs_pid = tl.program_id(axis=1)

    num_pid_row = tl.cdiv(dim_m, BLOCK_SIZE_M)
    num_pid_col = tl.cdiv(dim_k, BLOCK_SIZE_K)

    num_pid_in_group = GROUP_SIZE_M * num_pid_col
    group_id = pid // num_pid_in_group
    first_pid_row = group_id * GROUP_SIZE_M
    group_size_row = min(num_pid_row - first_pid_row, GROUP_SIZE_M)
    pid_row = first_pid_row + (pid % group_size_row)
    pid_col = (pid % num_pid_in_group) // group_size_row

    acc = tl.zeros([BLOCK_SIZE_M, BLOCK_SIZE_K], dtype=tl.float32)

    a_block_ptr = tl.make_block_ptr(
        a_ptr + bs_pid * abs_stride,
        shape=(dim_m, dim_n),
        strides=(arow_stride, acol_stride),
        offsets=(pid_row * BLOCK_SIZE_M, 0),
        block_shape=(BLOCK_SIZE_M, BLOCK_SIZE_N),
        order=(1, 0),
    )
    b_block_ptr = tl.make_block_ptr(
        b_ptr,
        shape=(dim_n, dim_k),
        strides=(brow_stride, bcol_stride),
        offsets=(
            0,
            pid_col * BLOCK_SIZE_K,
        ),
        block_shape=(BLOCK_SIZE_N, BLOCK_SIZE_K),
        order=(1, 0),
    )
    c_block_ptr = tl.make_block_ptr(
        c_ptr + bs_pid * cbs_stride,
        shape=(dim_m, dim_k),
        strides=(crow_stride, ccol_stride),
        offsets=(
            pid_row * BLOCK_SIZE_M,
            pid_col * BLOCK_SIZE_K,
        ),
        block_shape=(BLOCK_SIZE_M, BLOCK_SIZE_K),
        order=(1, 0),
    )

    # compute offset for bias based on row

    bias_start = pid_col * BLOCK_SIZE_K
    offsets = bias_start + tl.arange(0, BLOCK_SIZE_K)

    for n in range(0, tl.cdiv(dim_n, BLOCK_SIZE_N)):
        a_block = tl.load(a_block_ptr, boundary_check=(0, 1))
        b_block = tl.load(b_block_ptr, boundary_check=(0, 1))

        acc += tl.dot(a_block, b_block, allow_tf32=False)

        a_block_ptr = tl.advance(a_block_ptr, offsets=(0, BLOCK_SIZE_N))
        b_block_ptr = tl.advance(b_block_ptr, offsets=(BLOCK_SIZE_N, 0))

    bias = tl.load(bias_ptr + offsets, mask=offsets < dim_k)

    acc = acc + bias

    if fuse_relu:
        acc = tl.where(acc > 0, acc, 0.0)

    tl.store(c_block_ptr, acc.to(tl.float16), boundary_check=(0, 1))


def gemm(
    a: torch.Tensor, b: torch.Tensor, bias: torch.Tensor, fuse_relu: bool = False
) -> torch.Tensor:
    """
    Performs a matrix-multiply and add between batched tensor a and b.
    - a has shape (B,M,N) and b has shape (N,K)
    - bias has shape (K,)
    - Returns a tensor of shape (B, M, K)

    Optionally supports fusing ReLU activation computation.
    """

    assert (
        a.shape[2] == b.shape[0]
    ), f"Dimension mismatch. Expected a.shape[2] ({a.shape[2]}) to be equal to b.shape[0] ({b.shape[0]})"
    assert a.ndim == 3 and b.ndim == 2, "Incorrect number of dimensions for LHS or RHS"

    B, M, N, K = a.shape[0], a.shape[1], a.shape[2], b.shape[1]
    c = torch.empty((B, M, K), device=a.device, dtype=a.dtype)
    assert a.is_cuda and b.is_cuda and c.is_cuda

    grid = lambda META: (
        triton.cdiv(M, META["BLOCK_SIZE_M"]) * triton.cdiv(K, META["BLOCK_SIZE_K"]),
        B,
    )

    gemm_kernel[grid](
        a,
        b,
        c,
        bias,
        a.stride(0),
        a.stride(1),
        a.stride(2),
        b.stride(0),
        b.stride(1),
        c.stride(0),
        c.stride(1),
        c.stride(2),
        M,
        N,
        K,
        fuse_relu=fuse_relu,
    )
    return c
