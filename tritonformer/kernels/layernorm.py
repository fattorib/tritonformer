import torch
import triton
import triton.language as tl

from .reductions import _unbroadcast, get_optimal_split


@triton.jit
def layernorm_fwd_kernel(
    output_ptr,
    input_ptr,
    alpha_ptr,
    beta_ptr,
    d_embed,
    input_batch_stride,
    input_row_stride,
    output_row_stride,
    BLOCK_SIZE: tl.constexpr,
    eps,
):
    """
    Perform Layer Normalization on a 3-dimensional tensor along the last dimension.
    """

    pid = tl.program_id(0)
    batch_pid = tl.program_id(1)

    in_t_start_ptr = (
        input_ptr + (batch_pid * input_batch_stride) + pid * input_row_stride
    )
    offsets = tl.arange(0, BLOCK_SIZE)

    input_ptrs = in_t_start_ptr + offsets

    x = tl.load(input_ptrs, mask=offsets < d_embed, other=0.0).to(tl.float32)

    alpha = tl.load(
        alpha_ptr + offsets,
        mask=offsets < d_embed,
    )
    beta = tl.load(
        beta_ptr + offsets,
        mask=offsets < d_embed,
    )

    x_mean = tl.sum(x, axis=0) / d_embed
    centered = tl.where(offsets < d_embed, x - x_mean, 0.0)

    x_var = tl.sum((centered) * (centered), axis=0) / d_embed

    rstd = 1.0 / (tl.sqrt(x_var + eps))
    centered = x - x_mean
    norm = centered * rstd
    affine = alpha * norm + beta

    out_t_start_ptr = (
        output_ptr + (batch_pid * input_batch_stride) + pid * output_row_stride
    )
    output_ptrs = out_t_start_ptr + offsets

    tl.store(output_ptrs, affine, mask=offsets < d_embed)


def layernorm_fwd(
    x: torch.Tensor, alpha: torch.Tensor, beta: torch.Tensor, eps=1e-05
) -> torch.Tensor:
    output = torch.empty_like(x)
    n_b, n_ctx, d_embed = x.shape
    BLOCK_SIZE = triton.next_power_of_2(d_embed)

    assert x.is_cuda and output.is_cuda and x.dtype == torch.float16

    layernorm_fwd_kernel[(n_ctx, n_b)](
        output,
        x,
        alpha,
        beta,
        d_embed,
        x.stride(0),
        x.stride(1),
        output.stride(1),
        eps=eps,
        num_warps=8,
        BLOCK_SIZE=BLOCK_SIZE,
    )

    return output


@triton.jit
def layernorm_dx_kernel(
    grad_out_ptr,
    input_ptr,
    alpha_ptr,
    grad_x_ptr,
    grad_out_stride_b,
    grad_out_stride_r,
    input_stride_b,
    input_stride_r,
    numel,
    eps,
    BLOCK_SIZE: tl.constexpr,
):
    """dx kernel assuming other grads have been computed."""

    # one program instance per (bs x row)
    pid_bs = tl.program_id(0)
    pid_row = tl.program_id(1)

    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < numel

    g_out_start = (
        grad_out_ptr + pid_bs * grad_out_stride_b + pid_row * grad_out_stride_r
    )

    alpha = tl.load(alpha_ptr + offsets, mask=mask)
    grad_out = tl.load(g_out_start + offsets, mask=mask)

    input_start = input_ptr + pid_bs * input_stride_b + pid_row * input_stride_r
    input = tl.load(input_start + offsets, mask=mask, other=0.0).to(tl.float32)

    # NOTE: tl.where is needed here, load mask is not sufficient...
    x_mean = tl.sum(input, axis=0) / numel
    x_mu = tl.where(mask, input - x_mean, 0.0)
    x_var = tl.sum(x_mu * x_mu) / numel

    rstd = 1.0 / (tl.sqrt(x_var + eps))
    d_out = alpha * grad_out
    c1 = tl.sum(d_out * x_mu) / numel

    c2 = rstd * (d_out - (c1 * tl.math.pow(rstd, 2.0) * x_mu))
    grad_in = c2 - (1.0 / numel) * (tl.sum(c2))

    g_x_start = grad_x_ptr + pid_bs * grad_out_stride_b + pid_row * grad_out_stride_r

    tl.store(g_x_start + offsets, value=grad_in, mask=mask)


def dx_layernorm(grad_output, input, alpha, eps):
    n_bs, n_sq, n_d = grad_output.shape
    grad_input = torch.empty_like(grad_output)

    grid = (n_bs, n_sq)
    BLOCK_SIZE = triton.next_power_of_2(n_d)

    layernorm_dx_kernel[grid](
        grad_output,
        input,
        alpha,
        grad_input,
        grad_output.stride(0),
        grad_output.stride(1),
        input.stride(0),
        input.stride(1),
        n_d,
        eps,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=8,
    )

    return grad_input


@triton.jit
def layernorm_da_db_kernel(
    grad_out_ptr,
    grad_alpha_ptr,
    grad_beta_ptr,
    input_ptr,
    grad_out_stride_b,
    grad_alpha_stride_b,
    grad_beta_stride_b,
    input_stride_b,
    numel,
    numbatch,
    eps,
    BLOCK_SIZE: tl.constexpr,
    NUM_GROUP: tl.constexpr,
):
    # compute dalpha and dbeta in one pass
    pid_bs = tl.program_id(0)
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < numel

    sum_alpha = tl.zeros([BLOCK_SIZE], dtype=tl.float16)
    compensator_alpha = tl.zeros([BLOCK_SIZE], dtype=tl.float16)

    sum_beta = tl.zeros([BLOCK_SIZE], dtype=tl.float16)
    compensator_beta = tl.zeros([BLOCK_SIZE], dtype=tl.float16)

    for i in range(pid_bs, numbatch, NUM_GROUP):
        # load one batch item from x
        grad_out_start_ptr = grad_out_stride_b * i
        input_start_ptr = input_stride_b * i

        x = tl.load(input_ptr + input_start_ptr + offsets, mask=mask).to(tl.float32)
        grad_out = tl.load(grad_out_ptr + grad_out_start_ptr + offsets, mask=mask)

        x_mean = tl.sum(x, axis=0) / numel
        x_mu = tl.where(offsets < numel, x - x_mean, 0.0)

        x_var = tl.sum(x_mu * x_mu) / numel
        rstd = 1.0 / (tl.sqrt(x_var + eps))

        grad_alpha_accum = (grad_out * rstd * x_mu).to(tl.float16)
        grad_beta_accum = grad_out

        y_alpha = grad_alpha_accum - compensator_alpha
        y_beta = grad_beta_accum - compensator_beta

        tmp_alpha = sum_alpha + y_alpha
        compensator_alpha = (tmp_alpha - sum_alpha) - y_alpha
        sum_alpha = tmp_alpha

        tmp_beta = sum_beta + y_beta
        compensator_beta = (tmp_beta - sum_beta) - y_beta
        sum_beta = tmp_beta

    tl.store(
        grad_alpha_ptr + grad_alpha_stride_b * pid_bs + offsets, sum_alpha, mask=mask
    )
    tl.store(grad_beta_ptr + grad_beta_stride_b * pid_bs + offsets, sum_beta, mask=mask)


def layernorm_da_db(grad_output, input, eps):
    input = input.view(-1, input.shape[-1])
    _, max_fact = get_optimal_split(input.shape[0])

    NUM_GROUP = max_fact

    grad_alpha = torch.empty(
        (NUM_GROUP, grad_output.shape[-1]),
        device=grad_output.device,
        dtype=grad_output.dtype,
    )
    grad_beta = torch.empty_like(grad_alpha)

    grad_output = grad_output.view(-1, grad_output.shape[-1])
    BLOCK_SIZE = triton.next_power_of_2(grad_output.shape[-1])

    layernorm_da_db_kernel[(NUM_GROUP,)](
        grad_output,
        grad_alpha,
        grad_beta,
        input,
        grad_output.stride(0),
        grad_alpha.stride(0),
        grad_beta.stride(0),
        input.stride(0),
        grad_output.shape[-1],
        grad_output.shape[0],
        eps,
        BLOCK_SIZE,
        NUM_GROUP,
        num_warps=8,
    )

    return _unbroadcast(grad_alpha), _unbroadcast(grad_beta)
