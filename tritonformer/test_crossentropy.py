import os

import numpy as np
import pytest
import torch

from .crossentropy import CrossEntropyLoss

torch.manual_seed(0)
np.random.seed(0)

try:
    NUM_TEST = int(os.environ["NUM_TEST"])
except Exception:
    NUM_TEST = 5


def softmax_xentropy(logits, labels):
    log_probs = -1 * torch.nn.functional.log_softmax(logits, dim=-1)
    return torch.sum(log_probs * labels) / logits.shape[0]


@pytest.mark.parametrize(
    "dims",
    [
        (b, d)
        for b in np.random.randint(1, 32, size=NUM_TEST)
        for d in np.random.randint(256, 64000, size=NUM_TEST)
    ],
)
def test_cross_entropy_fwd(dims):
    b, logit_dim = dims
    logits = torch.randn(
        (b, logit_dim), device="cuda:0", dtype=torch.float16, requires_grad=True
    )
    labels = torch.randint(size=(b,), high=logit_dim, low=0, device="cuda:0").long()

    out_torch = torch.nn.functional.cross_entropy(logits, labels)
    out_triton = CrossEntropyLoss.apply(logits, labels)

    assert torch.allclose(out_torch, out_triton, atol=1e-2, rtol=0)


@pytest.mark.parametrize(
    "dims",
    [
        (b, d)
        for b in np.random.randint(1, 32, size=NUM_TEST)
        for d in np.random.randint(256, 64000, size=NUM_TEST)
    ],
)
def test_cross_entropy_bwd(dims):
    b, logit_dim = dims
    logits = torch.randn(
        (b, logit_dim), device="cuda:0", dtype=torch.float16, requires_grad=True
    )
    labels = torch.randint(size=(b,), high=logit_dim, low=0, device="cuda:0").long()
    dy = torch.tensor(1.0)

    out_torch = torch.nn.functional.cross_entropy(logits, labels)
    out_triton = CrossEntropyLoss.apply(logits, labels)

    out_torch.backward(dy, retain_graph=True)
    (dx_torch,) = [_.grad.clone() for _ in [logits]]
    logits.grad = None

    out_triton.backward(dy, retain_graph=True)
    (dx_triton,) = [_.grad.clone() for _ in [logits]]

    assert torch.allclose(dx_torch, dx_triton, atol=1e-2, rtol=0)
