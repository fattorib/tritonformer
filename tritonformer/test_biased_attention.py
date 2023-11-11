import math
import os
from typing import List

import numpy as np
import pytest
import torch

from .attention import BiasedCausalAttention

torch.manual_seed(0)
np.random.seed(0)

try:
    NUM_TEST = int(os.environ["NUM_TEST"])
except Exception:
    NUM_TEST = 5

from typing import Tuple


def make_qkv(
    bs: int, nh: int, sq: int, hd: int
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    q = torch.randn(
        (bs, nh, sq, hd), device="cuda:0", dtype=torch.float16
    ).requires_grad_(True)
    k = torch.randn(
        (bs, nh, sq, hd), device="cuda:0", dtype=torch.float16
    ).requires_grad_(True)
    v = torch.randn(
        (bs, nh, sq, hd), device="cuda:0", dtype=torch.float16
    ).requires_grad_(True)

    return (q, k, v)


def get_slopes(n: int) -> List[float]:
    """Create slopes for ALiBi attention bias."""

    def get_slopes_power_of_2(n):
        start = 2 ** (-(2 ** -(math.log2(n) - 3)))
        ratio = start
        return [start * ratio**i for i in range(n)]

    if math.log2(n).is_integer():
        return get_slopes_power_of_2(n)
    else:
        closest_power_of_2 = 2 ** math.floor(math.log2(n))
        return (
            get_slopes_power_of_2(closest_power_of_2)
            + get_slopes(2 * closest_power_of_2)[0::2][: n - closest_power_of_2]
        )


def generate_alibi_mask(seq_len: int, num_heads: int) -> torch.Tensor:
    """Create ALiBi attention bias."""
    a = -torch.tril(
        torch.arange(seq_len).view(seq_len, 1).repeat(1, seq_len)
        + torch.arange(0, -seq_len, -1)
    )

    a = a.to("cuda:0")
    a = a.to(torch.float16)

    slopes = torch.tensor(get_slopes(num_heads), device=a.device, dtype=a.dtype)
    alibi_mask = a * slopes[..., None, None]

    mask = torch.tril(
        torch.ones((1, 1, seq_len, seq_len), dtype=torch.bool, device="cuda:0")
    )

    alibi_mask = alibi_mask.masked_fill(
        mask[:, :, :seq_len, :seq_len] == 0, float("-inf")
    )
    return alibi_mask


def torch_ref(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, attn_bias: torch.Tensor
) -> torch.Tensor:
    """Reference torch implementation with optional attention bias."""
    head_scale = 1.0 / (q.shape[-1] ** (0.5))
    attention_scores = q @ k.transpose(-2, -1) * head_scale

    if attn_bias is None:
        attn_bias = torch.zeros(
            q.shape[-2], q.shape[-2], dtype=q.dtype, device=q.device
        )
        temp_mask = torch.ones(
            q.shape[-2], q.shape[-2], dtype=torch.bool, device=q.device
        ).tril(diagonal=0)
        attn_bias.masked_fill_(temp_mask.logical_not(), float("-inf"))
        attn_bias.to(q.dtype)

    attention_scores += attn_bias

    attention_probs = torch.nn.functional.softmax(
        attention_scores.float(), dim=-1
    ).half()

    return attention_probs @ v


@torch.no_grad()
@pytest.mark.parametrize(
    "dims",
    [
        (b, nh, sq, hd)
        for b in np.random.randint(1, 4, size=NUM_TEST)
        for nh in np.random.randint(4, 32, size=NUM_TEST)
        for sq in 64 * np.random.randint(1, 32, size=NUM_TEST)
        for hd in np.random.choice([32, 64], size=NUM_TEST, replace=True)
    ],
)
def test_attn_fwd(dims):
    bs, nh, sq, hd = dims

    q, k, v = make_qkv(bs, nh, sq, hd)

    attn_bias = generate_alibi_mask(sq, nh)

    out_torch = torch_ref(q, k, v, attn_bias=attn_bias)

    out_triton = BiasedCausalAttention.apply(q, k, v, attn_bias[0, ...])

    assert torch.allclose(out_torch, out_triton, atol=1e-2, rtol=0)


@pytest.mark.parametrize(
    "dims",
    [
        (b, nh, sq, hd)
        for b in np.random.randint(1, 4, size=NUM_TEST)
        for nh in np.random.randint(4, 32, size=NUM_TEST)
        for sq in 64 * np.random.randint(1, 32, size=NUM_TEST)
        for hd in np.random.choice([32, 64], size=NUM_TEST, replace=True)
    ],
)
def test_attn_bwd_dV(dims):
    bs, nh, sq, hd = dims
    q, k, v = make_qkv(bs, nh, sq, hd)

    attn_bias = generate_alibi_mask(sq, nh)

    out_torch = torch_ref(q, k, v, attn_bias=attn_bias)
    out_triton = BiasedCausalAttention.apply(q, k, v, attn_bias[0, ...])

    dy = torch.randn_like(out_torch)

    out_torch.backward(dy, retain_graph=True)

    dq_torch, dk_torch, dv_torch = [_.grad.clone() for _ in [q, k, v]]
    q.grad = None
    k.grad = None
    v.grad = None

    out_triton.backward(dy, retain_graph=True)
    dq_triton, dk_triton, dv_triton = [_.grad.clone() for _ in [q, k, v]]

    assert torch.allclose(dv_triton, dv_torch, rtol=0, atol=1e-2)


@pytest.mark.parametrize(
    "dims",
    [
        (b, nh, sq, hd)
        for b in np.random.randint(1, 4, size=NUM_TEST)
        for nh in np.random.randint(4, 32, size=NUM_TEST)
        for sq in 64 * np.random.randint(1, 32, size=NUM_TEST)
        for hd in np.random.choice([32, 64], size=NUM_TEST, replace=True)
    ],
)
def test_attn_bwd_dK(dims):
    bs, nh, sq, hd = dims
    q, k, v = make_qkv(bs, nh, sq, hd)

    attn_bias = generate_alibi_mask(sq, nh)

    out_torch = torch_ref(q, k, v, attn_bias=attn_bias)
    out_triton = BiasedCausalAttention.apply(q, k, v, attn_bias[0, ...])

    dy = torch.randn_like(out_torch)

    out_torch.backward(dy, retain_graph=True)

    dq_torch, dk_torch, dv_torch = [_.grad.clone() for _ in [q, k, v]]
    q.grad = None
    k.grad = None
    v.grad = None

    out_triton.backward(dy, retain_graph=True)
    dq_triton, dk_triton, dv_triton = [_.grad.clone() for _ in [q, k, v]]

    assert torch.allclose(dk_triton, dk_torch, rtol=0, atol=1e-2)


@pytest.mark.parametrize(
    "dims",
    [
        (b, nh, sq, hd)
        for b in np.random.randint(1, 4, size=NUM_TEST)
        for nh in np.random.randint(4, 32, size=NUM_TEST)
        for sq in 64 * np.random.randint(1, 32, size=NUM_TEST)
        for hd in np.random.choice([32, 64], size=NUM_TEST, replace=True)
    ],
)
def test_attn_bwd_dQ(dims):
    bs, nh, sq, hd = dims
    q, k, v = make_qkv(bs, nh, sq, hd)

    attn_bias = generate_alibi_mask(sq, nh)

    out_torch = torch_ref(q, k, v, attn_bias=attn_bias)
    out_triton = BiasedCausalAttention.apply(q, k, v, attn_bias[0, ...])

    dy = torch.randn_like(out_torch)

    out_torch.backward(dy, retain_graph=True)

    dq_torch, dk_torch, dv_torch = [_.grad.clone() for _ in [q, k, v]]
    q.grad = None
    k.grad = None
    v.grad = None

    out_triton.backward(dy, retain_graph=True)
    dq_triton, dk_triton, dv_triton = [_.grad.clone() for _ in [q, k, v]]

    assert torch.allclose(dq_triton, dq_torch, rtol=0, atol=1e-2)
