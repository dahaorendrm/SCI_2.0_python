import torch
from torch.nn.modules.module import Module
from torch.autograd import Function
import correlation_cuda

class CorrelationFunction(Function):

#    def __init__(self, pad_size=3, kernel_size=3, max_displacement=20, stride1=1, stride2=2, corr_multiply=1):
#        super(CorrelationFunction, self).__init__()
#        self.pad_size = pad_size
#        self.kernel_size = kernel_size
#        self.max_displacement = max_displacement
#        self.stride1 = stride1
#        self.stride2 = stride2
#        self.corr_multiply = corr_multiply
        # self.out_channel = ((max_displacement/stride2)*2 + 1) * ((max_displacement/stride2)*2 + 1)
    @staticmethod
    def forward(ctx, input1, input2,pad_size,kernel_size,max_displacement,stride1,stride2,corr_multiply):
        ctx.save_for_backward(input1, input2,pad_size,kernel_size,max_displacement,stride1,stride2,corr_multiply)

        with torch.cuda.device_of(input1):
            rbot1 = input1.new()
            rbot2 = input2.new()
            output = input1.new()

            correlation_cuda.forward(input1, input2, rbot1, rbot2, output,
                pad_size, kernel_size, max_displacement,stride1, stride2, corr_multiply)

        return output
    @staticmethod
    def backward(ctx, grad_output):
        input1, input2,pad_size,kernel_size,max_displacement,stride1,stride2,corr_multiply = ctx.saved_tensors

        with torch.cuda.device_of(input1):
            rbot1 = input1.new()
            rbot2 = input2.new()

            grad_input1 = input1.new()
            grad_input2 = input2.new()

            correlation_cuda.backward(input1, input2, rbot1, rbot2, grad_output, grad_input1, grad_input2,
                int(pad_size), int(kernel_size), int(max_displacement),int(stride1), int(stride2), int(corr_multiply))

        return grad_input1, grad_input2,None,None,None,None,None,None


class Correlation(Module):
    def __init__(self, pad_size=0, kernel_size=0, max_displacement=0, stride1=1, stride2=2, corr_multiply=1):
        super(Correlation, self).__init__()
        self.pad_size = torch.tensor(pad_size)
        self.kernel_size = torch.tensor(kernel_size)
        self.max_displacement = torch.tensor(max_displacement)
        self.stride1 = torch.tensor(stride1)
        self.stride2 = torch.tensor(stride2)
        self.corr_multiply = torch.tensor(corr_multiply)

    def forward(self, input1, input2):

        result = CorrelationFunction.apply(input1, input2,self.pad_size, self.kernel_size, self.max_displacement,self.stride1, self.stride2, self.corr_multiply)

        return result
