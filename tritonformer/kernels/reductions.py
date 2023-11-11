"""Multi-axis reductions in Triton."""

import math
from functools import lru_cache
from typing import Tuple

import torch
import triton
import triton.language as tl


@lru_cache
def get_optimal_split(prod: int) -> Tuple[int, int]:
    """Finds the factorization which distributes work evenly in the reduction."""
    start = int(math.sqrt(prod))
    cond = prod % start
    while cond != 0:
        start -= 1
        cond = prod % start

    return start, prod // start


@triton.jit
def _reduction(
    x_ptr,
    out_ptr,
    x_stride_b,
    numel: tl.constexpr,
    numbatch: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """Naive Reduction. One thread block processes the complete reduction.
    Kahan summation is used to reduce numerical roundoff errors.
    """
    pid_bs = tl.program_id(0)
    assert pid_bs == 0

    offsets = tl.arange(0, BLOCK_SIZE)

    mask = offsets < numel

    sum = tl.zeros([BLOCK_SIZE], dtype=tl.float16)
    compensator = tl.zeros([BLOCK_SIZE], dtype=tl.float16)

    for i in range(numbatch):
        # load one batch item from x
        block_start_ptr = x_stride_b * i
        y = tl.load(x_ptr + block_start_ptr + offsets, mask=mask) - compensator
        tmp = sum + y
        compensator = (tmp - sum) - y
        sum = tmp

    tl.store(out_ptr + offsets, sum, mask=mask)


def _unbroadcast(x: torch.Tensor) -> torch.Tensor:
    out = torch.empty((1, 1, x.shape[-1]), device=x.device, dtype=x.dtype)
    BLOCK_SIZE = triton.next_power_of_2(x.shape[-1])
    _reduction[(1,)](
        x, out, x.stride(0), x.shape[1], x.shape[0], BLOCK_SIZE, num_warps=4
    )
    return out


@triton.jit
def reduction_2(
    x_ptr,
    out_ptr,
    x_stride_b,
    o_stride_b,
    numel: tl.constexpr,
    numbatch: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    NUM_GROUP: tl.constexpr,
):
    """Parallel Reduction. Split reduction into multiple steps."""

    pid_bs = tl.program_id(0)

    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < numel

    sum = tl.zeros([BLOCK_SIZE], dtype=tl.float16)
    compensator = tl.zeros([BLOCK_SIZE], dtype=tl.float16)

    for i in range(pid_bs, numbatch, NUM_GROUP):
        # load one batch item from x
        block_start_ptr = x_stride_b * i
        y = tl.load(x_ptr + block_start_ptr + offsets, mask=mask) - compensator
        tmp = sum + y
        compensator = (tmp - sum) - y
        sum = tmp
    tl.store(out_ptr + (o_stride_b * pid_bs) + offsets, sum, mask=mask)


def unbroadcast_leading(x: torch.Tensor) -> torch.Tensor:
    """Unbroadcast (sum) over first two tensor dimensions."""

    x = x.view(-1, x.shape[-1])

    _, max_fact = get_optimal_split(x.shape[0])

    NUM_GROUP = max_fact

    tmp = torch.empty((NUM_GROUP, x.shape[-1]), device=x.device, dtype=x.dtype)

    BLOCK_SIZE = triton.next_power_of_2(x.shape[-1])

    reduction_2[(NUM_GROUP,)](
        x,
        tmp,
        x.stride(0),
        tmp.stride(0),
        x.shape[1],
        x.shape[0],
        BLOCK_SIZE,
        NUM_GROUP,
        num_warps=8,
    )

    return _unbroadcast(tmp)


@triton.jit
def reduction_tail(
    in_ptr,
    out_ptr,
    in_stride_batch,
    in_stride_row,
    out_stride_batch,
    out_stride_row,
    numel,
    BLOCK_SIZE: tl.constexpr,
):
    pid_bs = tl.program_id(0)
    pid_row = tl.program_id(1)

    in_start_ptr = in_ptr + pid_bs * in_stride_batch + pid_row * in_stride_row
    offsets = tl.arange(0, BLOCK_SIZE)

    in_row_bs = tl.load(in_start_ptr + offsets, mask=(offsets < numel))

    tmp = tl.sum(in_row_bs)
    out_start_ptr = out_ptr + pid_bs * out_stride_batch + pid_row * out_stride_row

    tl.store(out_start_ptr, value=tmp)


def unbroadcast_tailing(x: torch.Tensor) -> torch.Tensor:
    """Tensor reduction over final dimension."""

    n_b, n_r, n_d = x.shape

    grid = (n_b, n_r)

    out = torch.empty((x.shape[0], x.shape[1], 1), device=x.device)

    BLOCK_SIZE = triton.next_power_of_2(n_d)

    reduction_tail[grid](
        x,
        out,
        x.stride(0),
        x.stride(1),
        out.stride(0),
        out.stride(1),
        BLOCK_SIZE=BLOCK_SIZE,
        numel=n_d,
    )

    return out
