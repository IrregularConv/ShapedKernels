import torch
import torch.nn as nn
import torch.nn.functional as F

class BrainDamage(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, weights, mask, padding=1, stride=1, groups=1):
        if torch.cuda.is_available():
            mask = mask.to(input.device)
        if mask.size() == weights.size():
            weights = weights * mask
        output = F.conv2d(input, weights, stride=stride, padding=padding, groups=groups, dilation=padding)
        ctx.save_for_backward(input, weights, mask, torch.tensor(stride), torch.tensor(padding), torch.tensor(groups))
        return output

    @staticmethod
    def backward(ctx, grad_output):
        x, w, mask, stride, padding, groups = ctx.saved_variables
        x_grad = w_grad = None
        if ctx.needs_input_grad[0]:
            x_grad = torch.nn.grad.conv2d_input(x.shape, w, grad_output, stride=stride.item(), padding=padding.item(), groups=groups.item(), dilation=padding.item())
        if ctx.needs_input_grad[1]:
            w_grad = torch.nn.grad.conv2d_weight(x, w.shape, grad_output, stride=stride.item(), padding=padding.item(), groups=groups.item(), dilation=padding.item())
            w_grad *= mask
        return x_grad, w_grad, None, None, None, None

class IrreguConv(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=1, groups=1, dilation=1, mask = None):
        super(IrreguConv, self).__init__(in_channels, out_channels, kernel_size)
        
        if mask == None:
            self.mask = torch.ones(out_channels, in_channels, kernel_size, kernel_size)
        else:
            self.mask = torch.tensor(out_channels*[in_channels*[mask]])

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.groups = groups
        self.dilation = dilation

        # Set up weights given in nn.Conv2d
        #weights = torch.zeros((out_channels, in_channels, kernel_size, kernel_size), requires_grad = True)
        #nn.init.kaiming_normal(weights)
        weights = self.weight
        self.weight = torch.nn.Parameter(weights*self.mask)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        # if torch.cuda.is_available():
        #     self.mask = self.mask.to(input.device)
        # weights = self.weight.data * self.mask
        # output = F.conv2d(input, weights, stride=self.stride, padding=self.padding, groups=self.groups, dilation=self.padding)
        output = BrainDamage.apply(input, self.weight, self.mask, self.padding, self.stride, self.groups)
        
        return output