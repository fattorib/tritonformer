from torch.autograd import Function

from .kernels.softmax import softmax_bwd, softmax_fwd


class Softmax(Function):
    """Fused Softmax over last dimension of a 3-d input tensor."""

    @staticmethod
    def forward(input):
        output = softmax_fwd(input)
        return output

    @staticmethod
    def setup_context(ctx, inputs, output):
        ctx.save_for_backward(output)

    @staticmethod
    def backward(ctx, grad_output):
        (saved_out,) = ctx.saved_tensors  # (b, sq, dim)
        return softmax_bwd(grad_output, saved_out)
