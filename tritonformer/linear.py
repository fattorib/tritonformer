import torch
from einops import rearrange
from torch.autograd import Function

from .kernels.activation import mask_grad
from .kernels.gemm import bmm, gemm, mm
from .kernels.reductions import unbroadcast_leading


class LinearNoBias(Function):
    """MM(X,A) kernel call."""

    @staticmethod
    @torch.cuda.amp.custom_fwd(cast_inputs=torch.float16)
    def forward(ctx, input, weight):
        ctx.save_for_backward(input, weight)
        return mm(input, weight)

    @staticmethod
    @torch.cuda.amp.custom_bwd
    def backward(ctx, grad_output):
        inputs, weights = ctx.saved_tensors
        grad_input = mm(grad_output, weights.T)
        grad_weights = bmm(rearrange(inputs, "b sq d -> b d sq"), grad_output)
        return grad_input, torch.sum(grad_weights, keepdim=True, dim=(0,))


class LinearBias(Function):
    """Fused GEMM(X,A,B) into a single kernel call."""

    @staticmethod
    @torch.cuda.amp.custom_fwd(cast_inputs=torch.float16)
    def forward(ctx, input, weight, bias):
        ctx.save_for_backward(input, weight)
        return gemm(input, weight, bias, False)

    @staticmethod
    @torch.cuda.amp.custom_bwd
    def backward(ctx, grad_output):
        inputs, weights = ctx.saved_tensors
        grad_input = mm(grad_output, weights.T)
        grad_weights = bmm(rearrange(inputs, "b sq d -> b d sq"), grad_output)
        grad_bias = unbroadcast_leading(grad_output)
        return grad_input, torch.sum(grad_weights, keepdim=True, dim=(0,)), grad_bias


class LinearNoBiasReLU(Function):
    """Fused ReLU(MM(X,A)) into a single kernel call."""

    @staticmethod
    @torch.cuda.amp.custom_fwd(cast_inputs=torch.float16)
    def forward(ctx, input, weight):
        out_w_act = mm(input, weight, True)
        ctx.save_for_backward(input, weight, out_w_act)
        return out_w_act

    @staticmethod
    @torch.cuda.amp.custom_bwd
    def backward(ctx, grad_output):
        inputs, weights, out_w_act = ctx.saved_tensors

        mask_grad(grad_output, out_w_act)
        grad_input = mm(grad_output, weights.T)
        grad_weights = bmm(rearrange(inputs, "b sq d -> b d sq"), grad_output)

        return grad_input, torch.sum(
            grad_weights,
            dim=(0),
        )


class LinearBiasReLU(Function):
    """Fused ReLU(GEMM(X,A,bias)) into a single kernel call."""

    @staticmethod
    @torch.cuda.amp.custom_fwd(cast_inputs=torch.float16)
    def forward(ctx, input, weight, bias):
        out_w_act = gemm(input, weight, bias, True)
        ctx.save_for_backward(input, weight, out_w_act)
        return out_w_act

    @staticmethod
    @torch.cuda.amp.custom_bwd
    def backward(ctx, grad_output):
        inputs, weights, out_w_act = ctx.saved_tensors

        mask_grad(grad_output, out_w_act)

        grad_input = mm(grad_output, weights.T)

        grad_weights = bmm(rearrange(inputs, "b sq d -> b d sq"), grad_output)

        grad_bias = unbroadcast_leading(grad_output)

        return (
            grad_input,
            torch.sum(
                grad_weights,
                dim=(0,),
            ),
            grad_bias,
        )
