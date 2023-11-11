import torch
import triton
import triton.language as tl


@triton.jit
def mask_grad_kernel(
    grad_ptr, out_act_ptr, bs_stride, sq_stride, numel, BLOCK_SIZE: tl.constexpr
):
    bs_pid = tl.program_id(0)
    sq_pid = tl.program_id(1)

    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < numel

    grad_start_ptr = grad_ptr + bs_pid * bs_stride + sq_pid * sq_stride
    act_start_ptr = out_act_ptr + bs_pid * bs_stride + sq_pid * sq_stride

    grad = tl.load(grad_start_ptr + offsets, mask=mask)
    act = tl.load(act_start_ptr + offsets, mask=mask)

    grad = tl.where(act > 0.0, grad, 0.0)

    tl.store(grad_start_ptr + offsets, grad.to(tl.float16), mask=mask)


def mask_grad(grad: torch.Tensor, out_act: torch.Tensor) -> torch.Tensor:
    """Fused masking kernel used to process ReLU backward pass."""

    n_b, n_sq, d_model = grad.shape

    grid = (n_b, n_sq)

    num_warps = 4 if d_model <= 2048 else 8

    mask_grad_kernel[grid](
        grad,
        out_act,
        grad.stride(0),
        grad.stride(1),
        d_model,
        BLOCK_SIZE=triton.next_power_of_2(d_model),
        num_warps=num_warps,
    )
