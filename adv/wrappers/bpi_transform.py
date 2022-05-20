from torch.autograd import Function


class BPITransform(Function):

    # Note that both forward and backward are @staticmethods
    @staticmethod
    def forward(ctx, input, transform):
        # ctx.save_for_backward(input)
        output = transform(input)
        return output

    # This function has only a single output, so it gets only one gradient
    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None
