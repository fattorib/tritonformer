from typing import Any, Tuple

import torch
from torch.autograd import Function

from .kernels.attention import flash_wrapper_bwd, flash_wrapper_fwd
from .kernels.biased_attention import (
    flash_wrapper_bwd_attn_bias,
    flash_wrapper_fwd_attn_bias,
)


class CausalAttention(Function):
    @staticmethod
    @torch.cuda.amp.custom_fwd(cast_inputs=torch.float16)
    def forward(
        ctx: Any, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor
    ) -> torch.Tensor:
        out, m, l = flash_wrapper_fwd(query, key, value)
        ctx.save_for_backward(m, l, query, key, value, out)
        return out

    @staticmethod
    @torch.cuda.amp.custom_bwd
    def backward(
        ctx: Any, grad_output: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        m, l, query, key, value, out = ctx.saved_tensors

        grad_query, grad_key, grad_value = flash_wrapper_bwd(
            grad_output, out, query, key, value, m, l
        )

        return grad_query, grad_key, grad_value


class BiasedCausalAttention(Function):
    @staticmethod
    def forward(
        ctx: Any,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_bias: torch.Tensor,
    ) -> torch.Tensor:
        out, m, l = flash_wrapper_fwd_attn_bias(query, key, value, attn_bias)
        ctx.save_for_backward(m, l, query, key, value, out, attn_bias)
        return out

    @staticmethod
    def backward(
        ctx: Any, grad_output: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, None]:
        m, l, query, key, value, out, attn_bias = ctx.saved_tensors

        grad_query, grad_key, grad_value = flash_wrapper_bwd_attn_bias(
            grad_output, out, query, key, value, m, l, attn_bias
        )

        return grad_query, grad_key, grad_value, None
