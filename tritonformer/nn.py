"""Overloaded torch.nn modules"""
import torch
import torch.nn as nn

from . import (
    layer_norm,
    linear_bias,
    linear_fused_bias_relu,
    linear_fused_relu,
    linear_no_bias,
)


class LayerNorm(nn.Module):
    """Triton LayerNorm with elementwise affine."""

    def __init__(
        self, normalized_shape: int, eps: float = 1e-5, device=None, dtype=None
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.normalized_shape = normalized_shape
        self.eps = eps
        self.weight = nn.Parameter(torch.empty(self.normalized_shape, **factory_kwargs))
        self.bias = nn.Parameter(torch.empty(self.normalized_shape, **factory_kwargs))

        self.reset_parameters()

    def reset_parameters(self) -> None:
        pass

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return layer_norm.apply(input, self.weight, self.bias, self.eps)

    def extra_repr(self) -> str:
        return (
            "{normalized_shape}, eps={eps}, "
            "elementwise_affine={elementwise_affine}".format(**self.__dict__)
        )


class Linear(nn.Module):
    """Linear layer with optional bias and activation fusions."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        fuse_activation: bool = False,
        device=None,
        dtype=None,
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(
            torch.empty((in_features, out_features), **factory_kwargs)
        )

        if bias:
            self.bias = nn.Parameter(torch.empty(out_features, **factory_kwargs))
        else:
            self.register_parameter("bias", None)

        self.fuse_activation = fuse_activation

        self.reset_parameters()

    def reset_parameters(self) -> None:
        pass

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if self.bias is None:
            if self.fuse_activation:
                return linear_fused_relu.apply(input, self.weight)
            else:
                return linear_no_bias.apply(input, self.weight)

        else:
            if self.fuse_activation:
                return linear_fused_bias_relu.apply(input, self.weight, self.bias)
            else:
                return linear_bias.apply(input, self.weight, self.bias)

    def extra_repr(self) -> str:
        return "in_features={}, out_features={}, bias={}".format(
            self.in_features, self.out_features, self.bias is not None
        )
