import torch
from torch.autograd import Function

from .kernels.layernorm import dx_layernorm, layernorm_da_db, layernorm_fwd


class LayerNorm(Function):
    @staticmethod
    @torch.cuda.amp.custom_fwd(cast_inputs=torch.float16)
    def forward(ctx, input, alpha, beta, eps):
        x_ln = layernorm_fwd(input, alpha, beta, eps)
        ctx.save_for_backward(input, alpha)
        ctx.eps = eps

        return x_ln

    @staticmethod
    @torch.cuda.amp.custom_bwd
    def backward(ctx, grad_output):
        input, alpha = ctx.saved_tensors

        grad_alpha, grad_beta = layernorm_da_db(grad_output, input, ctx.eps)

        grad_input = dx_layernorm(grad_output, input, alpha, ctx.eps)

        return grad_input, grad_alpha, grad_beta, None
