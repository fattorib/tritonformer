import os

import numpy as np
import pytest
import torch
from einops import rearrange

from .attention import CausalAttention

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


def generate_mask(seq_len: int) -> torch.Tensor:
    return torch.tril(
        torch.ones((1, 1, seq_len, seq_len), dtype=torch.bool, device="cuda:0")
    )


def torch_ref(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    causal_attn: bool,
    attn_mask: torch.Tensor,
) -> torch.Tensor:
    """Reference torch implementation with optional causal masking."""
    k = rearrange(k, "bs nh sk hd -> bs nh hd sk")
    head_scale = 1.0 / (q.shape[-1] ** (0.5))
    q = q * head_scale
    attention_scores = torch.matmul(q, k)
    if causal_attn:
        attention_scores.masked_fill_(
            ~attn_mask, torch.tensor(torch.finfo(torch.float16).min)
        )
    attention_probs = torch.nn.functional.softmax(
        attention_scores.float(), dim=-1
    ).half()
    return torch.matmul(attention_probs, v)


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

    attention_mask = generate_mask(sq)

    with torch.no_grad():
        out_torch = torch_ref(q, k, v, causal_attn=True, attn_mask=attention_mask)

    with torch.no_grad():
        out_triton = CausalAttention.apply(q, k, v)

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

    attention_mask = generate_mask(sq)

    out_torch = torch_ref(q, k, v, causal_attn=True, attn_mask=attention_mask)
    out_triton = CausalAttention.apply(q, k, v)

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

    attention_mask = generate_mask(sq)

    out_torch = torch_ref(q, k, v, causal_attn=True, attn_mask=attention_mask)
    out_triton = CausalAttention.apply(q, k, v)

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

    attention_mask = generate_mask(sq)

    out_torch = torch_ref(q, k, v, causal_attn=True, attn_mask=attention_mask)
    out_triton = CausalAttention.apply(q, k, v)

    dy = torch.randn_like(out_torch)

    out_torch.backward(dy, retain_graph=True)

    dq_torch, dk_torch, dv_torch = [_.grad.clone() for _ in [q, k, v]]
    q.grad = None
    k.grad = None
    v.grad = None

    out_triton.backward(dy, retain_graph=True)
    dq_triton, dk_triton, dv_triton = [_.grad.clone() for _ in [q, k, v]]

    assert torch.allclose(dq_triton, dq_torch, rtol=0, atol=1e-2)
