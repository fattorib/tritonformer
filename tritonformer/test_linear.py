import os
from math import log10

import numpy as np
import pytest
import torch

from .linear import LinearBias, LinearNoBias

torch.manual_seed(0)
np.random.seed(0)

try:
    NUM_TEST = int(os.environ["NUM_TEST"])
except Exception:
    NUM_TEST = 5


def torch_linear(
    x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor = None
) -> torch.Tensor:
    if bias is None:
        return torch.matmul(x, weight)
    else:
        return torch.matmul(x, weight) + bias


def relative_error(x: torch.Tensor, y: torch.Tensor) -> float:
    """Compute relative error between x and y."""
    return (torch.linalg.norm(x - y) / torch.linalg.norm(y)).item()


@pytest.mark.parametrize(
    "dims",
    [
        (b, i, j, k)
        for b in np.random.randint(1, 32, size=NUM_TEST)
        for i in np.random.randint(64, 4096, size=NUM_TEST)
        for j in np.random.randint(64, 4096, size=NUM_TEST)
        for k in np.random.randint(64, 4096, size=NUM_TEST)
    ],
)
def test_linear_nobias_fwd(dims):
    b, m, n, k = dims

    a: torch.Tensor = torch.randn(
        size=(b, m, n), device="cuda:0", dtype=torch.float16, requires_grad=True
    )
    weight: torch.Tensor = torch.randn(
        size=(n, k), device="cuda:0", dtype=torch.float16, requires_grad=True
    ) / (n**0.5)

    expected_torch = torch_linear(a, weight)

    out_triton = LinearNoBias.apply(a, weight)

    assert torch.allclose(out_triton, expected_torch, rtol=0.0, atol=1e-2)


@pytest.mark.parametrize(
    "dims",
    [
        (b, i, j, k)
        for b in np.random.randint(1, 32, size=NUM_TEST)
        for i in np.random.randint(64, 4096, size=NUM_TEST)
        for j in np.random.randint(64, 4096, size=NUM_TEST)
        for k in np.random.randint(64, 4096, size=NUM_TEST)
    ],
)
def test_linear_nobias_bwd(dims):
    b, m, n, k = dims

    a: torch.Tensor = torch.randn(size=(b, m, n), device="cuda:0", dtype=torch.float16)
    weight: torch.Tensor = torch.randn(
        size=(n, k), device="cuda:0", dtype=torch.float16
    ) / (n**0.5)

    a.requires_grad_(True)
    weight.requires_grad_(True)

    expected_torch = torch_linear(a, weight)
    out_triton = LinearNoBias.apply(a, weight)

    dy = torch.randn_like(expected_torch)

    expected_torch.backward(dy, retain_graph=True)
    dx_torch, dw_torch = [_.grad.clone() for _ in [a, weight]]
    a.grad, weight.grad = None, None

    out_triton.backward(dy, retain_graph=True)
    dx_triton, dw_triton = [_.grad.clone() for _ in [a, weight]]

    assert torch.allclose(dx_torch, dx_triton, atol=1e-2, rtol=0)
    assert torch.allclose(dw_torch, dw_triton, atol=1e-2, rtol=0)


@pytest.mark.parametrize(
    "dims",
    [
        (b, i, j, k)
        for b in np.random.randint(1, 32, size=NUM_TEST)
        for i in np.random.randint(64, 4096, size=NUM_TEST)
        for j in np.random.randint(64, 4096, size=NUM_TEST)
        for k in np.random.randint(64, 4096, size=NUM_TEST)
    ],
)
def test_linear_bias_fwd(dims):
    b, m, n, k = dims

    a: torch.Tensor = torch.randn(
        size=(b, m, n), device="cuda:0", dtype=torch.float16, requires_grad=True
    )
    weight: torch.Tensor = torch.randn(
        size=(n, k), device="cuda:0", dtype=torch.float16, requires_grad=True
    ) / (n**0.5)

    bias: torch.Tensor = torch.randn(
        (k,), device="cuda:0", dtype=torch.float16, requires_grad=True
    )

    expected_torch = torch_linear(a, weight, bias)

    out_triton = LinearBias.apply(a, weight, bias)

    assert torch.allclose(out_triton, expected_torch, rtol=0.0, atol=1e-2)


@pytest.mark.parametrize(
    "dims",
    [
        (b, i, j, k)
        for b in np.random.randint(1, 32, size=NUM_TEST)
        for i in np.random.randint(64, 4096, size=NUM_TEST)
        for j in np.random.randint(64, 4096, size=NUM_TEST)
        for k in np.random.randint(64, 4096, size=NUM_TEST)
    ],
)
def test_linear_bias_bwd(dims):
    b, m, n, k = dims

    a: torch.Tensor = torch.randn(size=(b, m, n), device="cuda:0", dtype=torch.float16)
    weight: torch.Tensor = torch.randn(
        size=(n, k), device="cuda:0", dtype=torch.float16
    ) / (n**0.5)

    bias: torch.Tensor = torch.randn(
        (k,), device="cuda:0", dtype=torch.float16, requires_grad=True
    )

    a.requires_grad_(True)
    weight.requires_grad_(True)

    expected_torch = torch_linear(a, weight, bias)
    out_triton = LinearBias.apply(a, weight, bias)

    dy = torch.randn_like(expected_torch)

    expected_torch.backward(dy, retain_graph=True)
    dx_torch, dw_torch, db_torch = [_.grad.clone() for _ in [a, weight, bias]]
    a.grad, weight.grad, bias.grad = None, None, None

    out_triton.backward(dy, retain_graph=True)
    dx_triton, dw_triton, db_triton = [_.grad.clone() for _ in [a, weight, bias]]

    assert torch.allclose(dx_torch, dx_triton, atol=1e-2, rtol=0)
    assert torch.allclose(dw_torch, dw_triton, atol=1e-2, rtol=0)

    # relative error is a better metric here
    assert -1 * log10(relative_error(db_triton, db_torch)) > 3
