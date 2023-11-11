import os
from math import log10

import numpy as np
import pytest
import torch

from .layernorm import LayerNorm

torch.manual_seed(0)
np.random.seed(0)

try:
    NUM_TEST = int(os.environ["NUM_TEST"])
except Exception:
    NUM_TEST = 5


def layer_norm(
    x: torch.Tensor, alpha: torch.Tensor, beta: torch.Tensor
) -> torch.Tensor:
    """Reference torch implementation."""
    return torch.nn.functional.layer_norm(x, torch.Size([x.shape[-1]]), alpha, beta)


def relative_error(x: torch.Tensor, y: torch.Tensor) -> float:
    """Compute relative error between x and y."""
    return (torch.linalg.norm(x - y) / torch.linalg.norm(y)).item()


@pytest.mark.parametrize(
    "dims",
    [
        (b, i, j)
        for b in np.random.randint(1, 32, size=NUM_TEST)
        for i in np.random.randint(64, 1024, size=NUM_TEST)
        for j in np.random.randint(64, 1024, size=NUM_TEST)
    ],
)
def test_ln_fwd(dims):
    b, sq, d = dims

    x = 3.14 + 0.4 * torch.randn((b, sq, d), device="cuda:0", dtype=torch.float16)
    alpha = torch.randn((d,), requires_grad=True, device="cuda:0", dtype=torch.float16)
    beta = torch.randn((d,), requires_grad=True, device="cuda:0", dtype=torch.float16)

    x.requires_grad_(True)

    out_torch = layer_norm(x, alpha, beta)
    out_triton = LayerNorm.apply(x, alpha, beta, 1e-05)

    assert torch.allclose(out_torch, out_triton, atol=1e-2, rtol=0)


@pytest.mark.parametrize(
    "dims",
    [
        (b, i, j)
        for b in np.random.randint(1, 32, size=NUM_TEST)
        for i in np.random.randint(64, 1024, size=NUM_TEST)
        for j in np.random.randint(64, 1024, size=NUM_TEST)
    ],
)
def test_ln_bwd_dx(dims):
    b, sq, d = dims

    x = 3.14 + 0.4 * torch.randn((b, sq, d), device="cuda:0", dtype=torch.float16)
    alpha = torch.randn((d,), requires_grad=True, device="cuda:0", dtype=torch.float16)
    beta = torch.randn((d,), requires_grad=True, device="cuda:0", dtype=torch.float16)

    x.requires_grad_(True)

    out_torch = layer_norm(x, alpha, beta)
    out_triton = LayerNorm.apply(x, alpha, beta, 1e-05)

    dy = torch.randn((b, sq, d), device="cuda:0", dtype=torch.float16)

    out_torch.backward(dy, retain_graph=True)
    dx_torch, da_torch, db_torch = [_.grad.clone() for _ in [x, alpha, beta]]
    x.grad = None
    alpha.grad = None
    beta.grad = None

    out_triton.backward(dy, retain_graph=True)
    dx_triton, da_triton, db_triton = [_.grad.clone() for _ in [x, alpha, beta]]

    assert torch.allclose(dx_torch, dx_triton, atol=1e-2, rtol=0)


@pytest.mark.parametrize(
    "dims",
    [
        (b, i, j)
        for b in np.random.randint(1, 32, size=NUM_TEST)
        for i in np.random.randint(64, 1024, size=NUM_TEST)
        for j in np.random.randint(64, 1024, size=NUM_TEST)
    ],
)
def test_ln_bwd_dalpha(dims):
    b, sq, d = dims

    x = 3.14 + 0.4 * torch.randn((b, sq, d), device="cuda:0", dtype=torch.float16)
    alpha = torch.randn((d,), requires_grad=True, device="cuda:0", dtype=torch.float16)
    beta = torch.randn((d,), requires_grad=True, device="cuda:0", dtype=torch.float16)

    x.requires_grad_(True)

    out_torch = layer_norm(x, alpha, beta)
    out_triton = LayerNorm.apply(x, alpha, beta, 1e-05)

    dy = torch.randn((b, sq, d), device="cuda:0", dtype=torch.float16)

    out_torch.backward(dy, retain_graph=True)
    dx_torch, da_torch, db_torch = [_.grad.clone() for _ in [x, alpha, beta]]
    x.grad = None
    alpha.grad = None
    beta.grad = None

    out_triton.backward(dy, retain_graph=True)
    dx_triton, da_triton, db_triton = [_.grad.clone() for _ in [x, alpha, beta]]

    # assert torch.allclose(da_torch, da_triton, atol=1e-2, rtol=0)

    # relative error is a better metric here
    assert -1 * log10(relative_error(da_triton, da_torch)) > 3


@pytest.mark.parametrize(
    "dims",
    [
        (b, i, j)
        for b in np.random.randint(1, 32, size=NUM_TEST)
        for i in np.random.randint(64, 1024, size=NUM_TEST)
        for j in np.random.randint(64, 1024, size=NUM_TEST)
    ],
)
def test_ln_bwd_dbeta(dims):
    b, sq, d = dims

    x = 3.14 + 0.4 * torch.randn((b, sq, d), device="cuda:0", dtype=torch.float16)
    alpha = torch.randn((d,), requires_grad=True, device="cuda:0", dtype=torch.float16)
    beta = torch.randn((d,), requires_grad=True, device="cuda:0", dtype=torch.float16)

    x.requires_grad_(True)

    out_torch = layer_norm(x, alpha, beta)
    out_triton = LayerNorm.apply(x, alpha, beta, 1e-05)

    dy = torch.randn((b, sq, d), device="cuda:0", dtype=torch.float16)

    out_torch.backward(dy, retain_graph=True)
    dx_torch, da_torch, db_torch = [_.grad.clone() for _ in [x, alpha, beta]]
    x.grad = None
    alpha.grad = None
    beta.grad = None

    out_triton.backward(dy, retain_graph=True)
    dx_triton, da_triton, db_triton = [_.grad.clone() for _ in [x, alpha, beta]]

    # relative error is a better metric here
    assert -1 * log10(relative_error(db_triton, db_torch)) > 3
