import torch
import triton
import triton.language as tl


@triton.jit
def cross_entropy_fwd_kernel(
    logits_ptr,
    labels_ptr,
    act_ptr,
    loss_ptr,
    logits_stride_b,
    labels_stride_b,
    act_stride_b,
    loss_stride_b,
    num_probs,
    BLOCK_SIZE: tl.constexpr,
):
    pid_bs = tl.program_id(0)
    offsets = tl.arange(0, BLOCK_SIZE)

    logits_start_ptr = logits_ptr + (pid_bs * logits_stride_b)
    labels_start_ptr = labels_ptr + (pid_bs * labels_stride_b)
    act_start_ptr = act_ptr + (pid_bs * act_stride_b)

    logits = tl.load(
        logits_start_ptr + offsets, mask=offsets < num_probs, other=-float("inf")
    )

    logits = logits.to(tl.float32)
    shifted_logits = logits - tl.max(logits, axis=0)
    neglogprobs = (
        tl.math.log(tl.sum(tl.math.exp(shifted_logits), axis=0)) - shifted_logits
    )

    tl.store(
        act_start_ptr + offsets, neglogprobs.to(tl.float16), mask=offsets < num_probs
    )

    label_offset = tl.load(labels_start_ptr)

    tmp = tl.load(act_start_ptr + label_offset)

    tl.store(loss_ptr + pid_bs * loss_stride_b, tmp.to(tl.float16))


def cross_entropy_fwd(logits: torch.Tensor, labels: torch.Tensor):
    """Function wrapping cross-entropy forward pass kernel."""
    n_b, n_probs = logits.shape

    loss = torch.empty((logits.shape[0],), device=logits.device, dtype=torch.float16)

    softmax_act = torch.empty_like(logits, dtype=torch.float16)

    grid = (logits.shape[0],)

    BLOCK_SIZE = triton.next_power_of_2(n_probs)

    num_warps = 4 if BLOCK_SIZE <= 32768 else 8

    cross_entropy_fwd_kernel[grid](
        logits,
        labels,
        softmax_act,
        loss,
        logits.stride(0),
        labels.stride(0),
        softmax_act.stride(0),
        loss.stride(0),
        num_probs=n_probs,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=num_warps,
    )

    return loss, softmax_act.to(torch.float16)


@triton.jit
def cross_entropy_bwd_kernel(
    activation_ptr,
    labels_ptr,
    logits_stride_b,
    labels_stride_b,
    num_probs,
    BLOCK_SIZE: tl.constexpr,
):
    pid_bs = tl.program_id(0)
    offsets = tl.arange(0, BLOCK_SIZE)

    logprobs_start_ptr = activation_ptr + (pid_bs * logits_stride_b)
    labels_start_ptr = labels_ptr + (pid_bs * labels_stride_b)

    logprobs = tl.load(
        logprobs_start_ptr + offsets, mask=offsets < num_probs, other=-float("inf")
    )
    logprobs = logprobs.to(tl.float32)

    probs = tl.math.exp(-1.0 * logprobs)
    tl.store(
        logprobs_start_ptr + offsets, probs.to(tl.float16), mask=offsets < num_probs
    )
    label_offset = tl.load(labels_start_ptr)
    probs = tl.load(logprobs_start_ptr + label_offset)

    # discount gt labels
    probs -= 1.0
    tl.store(logprobs_start_ptr + label_offset, probs)


def cross_entropy_bwd(activation: torch.Tensor, labels: torch.Tensor):
    """Function wrapping the backward cross-entropy kernel.
    Performs an in-place update of the `activation` tensor
    and returns this as the gradient."""
    n_b, n_probs = activation.shape

    grid = (n_b,)

    cross_entropy_bwd_kernel[grid](
        activation,
        labels,
        activation.stride(0),
        labels.stride(0),
        num_probs=n_probs,
        BLOCK_SIZE=triton.next_power_of_2(n_probs),
    )

    return activation
