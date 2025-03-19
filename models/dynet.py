####
# 原文: https://0809zheng.github.io/2022/12/20/dyconv.html
####

import functools
import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.modules.conv import _ConvNd
from torch.nn.modules.utils import _pair
from torch.nn.parameter import Parameter

def default_conv(in_channels, out_channels, kernel_size, bias=True):
    return nn.Conv2d(
        in_channels, out_channels, kernel_size,
        padding=(kernel_size//2), bias=bias)
    
class _coefficient(nn.Module):
    def __init__(self, in_channels, num_experts, out_channels, dropout_rate):
        super(_coefficient, self).__init__()
        self.num_experts = num_experts
        self.dropout = nn.Dropout(dropout_rate)
        self.fc = nn.Linear(in_channels, num_experts*out_channels)

    def forward(self, x):
        x = torch.flatten(x)
        x = self.dropout(x)
        x = self.fc(x)
        x = x.view(self.num_experts, -1)
        return torch.softmax(x, dim=0)
    
    
class DyNet2D(_ConvNd):
    r"""
    in_channels (int): Number of channels in the input image
    out_channels (int): Number of channels produced by the convolution
    kernel_size (int or tuple): Size of the convolving kernel
    stride (int or tuple, optional): Stride of the convolution. Default: 1
    padding (int or tuple, optional): Zero-padding added to both sides of the input. Default: 0
    padding_mode (string, optional): ``'zeros'``, ``'reflect'``, ``'replicate'`` or ``'circular'``. Default: ``'zeros'``
    dilation (int or tuple, optional): Spacing between kernel elements. Default: 1
    groups (int, optional): Number of blocked connections from input channels to output channels. Default: 1
    bias (bool, optional): If ``True``, adds a learnable bias to the output. Default: ``True``
    num_experts (int): Number of experts per layer 
    """

    # def __init__(self, in_channels, out_channels, kernel_size, stride=1,
    #              padding=0, dilation=1, groups=1,
    #              bias=True, padding_mode='zeros', num_experts=3, dropout_rate=0.2):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1,
                 bias=True, padding_mode='reflect', num_experts=3, dropout_rate=0.2):
        kernel_size = _pair(kernel_size)
        stride = _pair(stride)
        padding = _pair(padding)
        dilation = _pair(dilation)
        super(DyNet2D, self).__init__(
            in_channels, out_channels, kernel_size, stride, padding, dilation,
            False, _pair(0), groups, bias, padding_mode)

        # 全局平均池化
        self._avg_pooling = functools.partial(F.adaptive_avg_pool2d, output_size=(1, 1))
        # 注意力全连接层
        self._coefficient_fn = _coefficient(in_channels, num_experts, out_channels, dropout_rate)
        # 多套卷积层的权重
        self.weight = Parameter(torch.Tensor(
            num_experts, out_channels, in_channels // groups, *kernel_size))
        self.reset_parameters()
        
    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=nn.init.calculate_gain('relu'))
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def _conv_forward(self, input, weight):
        if self.padding_mode != 'zeros':
            # print("why padding_mode != zeros")
            return F.conv2d(F.pad(input, (1,1,1,1), mode=self.padding_mode),
                            weight, self.bias, self.stride,
                            _pair(0), self.dilation, self.groups)
        return F.conv2d(input, weight, self.bias, self.stride,
                        self.padding, self.dilation, self.groups)
    
    def forward(self, inputs): # [b, c, h, w]
        res = []
        for input in inputs:
            # import pdb; pdb.set_trace()
            # print("why input: ", input.shape)
            input = input.unsqueeze(0) # [1, c, h, w]
            pooled_inputs = self._avg_pooling(input) # [1, c, 1, 1]
            # print("why pooled_inputs: ", pooled_inputs.shape)              #torch.Size([1, 64, 1, 1])
            routing_weights = self._coefficient_fn(pooled_inputs) # [k,]    #torch.Size([3, 64])
            # print("why routing_weights: ", routing_weights.shape)
            kernels = torch.sum(routing_weights[: , :, None, None, None] * self.weight, 0)
            # print("why kernels: ", kernels.shape)
            out = self._conv_forward(input, kernels)
            # print("why out: ", out.shape)
            res.append(out)
        return torch.cat(res, dim=0)
    
    # 太慢了这个for
    