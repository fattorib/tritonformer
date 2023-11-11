from typing import Any, Tuple

import torch
from torch.autograd import Function

from .kernels.crossentropy import cross_entropy_bwd, cross_entropy_fwd


class CrossEntropyLoss(Function):
    """Softmax Cross Entropy."""

    @staticmethod
    @torch.cuda.amp.custom_fwd(cast_inputs=torch.float16)
    def forward(ctx: Any, input: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        output, softmax_act = cross_entropy_fwd(input, labels)
        ctx.save_for_backward(labels, softmax_act)
        return output

    @staticmethod
    @torch.cuda.amp.custom_bwd
    def backward(ctx: Any, grad_output: torch.Tensor) -> Tuple[torch.Tensor, None]:
        saved_labels, saved_act = ctx.saved_tensors
        grad_input = cross_entropy_bwd(saved_act, saved_labels)
        return grad_output[..., None] * grad_input, None
