import os

import numpy as np
import pytest
import torch

from .softmax import Softmax

torch.manual_seed(0)
np.random.seed(0)

try:
    NUM_TEST = int(os.environ["NUM_TEST"])
except Exception:
    NUM_TEST = 5


def softmax(x: torch.Tensor) -> torch.Tensor:
    """PyTorch reference softmax."""
    return torch.nn.functional.softmax(x, dim=-1)


@pytest.mark.parametrize(
    "dims",
    [
        (b, i, j)
        for b in np.random.randint(1, 32, size=NUM_TEST)
        for i in np.random.randint(256, 1024, size=NUM_TEST)
        for j in np.random.randint(64, 1024, size=NUM_TEST)
    ],
)
def test_softmax_fwd_bwd_fp32(dims):
    b, sq, d = dims
    x = torch.randn(
        (b, sq, d), device="cuda:0", dtype=torch.float32, requires_grad=True
    )
    dy = 0.1 * torch.randn_like(x, device="cuda:0", dtype=torch.float32)

    out_torch = softmax(x)

    out_triton = Softmax.apply(x)

    # check forward
    assert torch.allclose(out_torch, out_triton, atol=1e-4, rtol=0)

    out_torch.backward(dy, retain_graph=True)
    (dx_torch,) = [_.grad.clone() for _ in [x]]
    x.grad = None

    out_triton.backward(dy, retain_graph=True)
    (dx_triton,) = [_.grad.clone() for _ in [x]]

    assert torch.allclose(dx_torch, dx_triton, atol=1e-4, rtol=0)
