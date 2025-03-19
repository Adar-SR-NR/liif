# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from torch.nn.modules.utils import _pair



# class DyNet2D(nn.Module):
#     def __init__(self, in_channels, out_channels, kernel_size, stride=1,
#                  padding=0, dilation=1, groups=1,
#                  bias=True, padding_mode='reflect', num_experts=3, dropout_rate=0.2):
#         super(DyNet2D, self).__init__()
#         self.in_channels = in_channels
#         self.out_channels = out_channels
#         self.kernel_size = _pair(kernel_size)
#         self.stride = _pair(stride)
#         self.padding = _pair(padding)
#         self.dilation = _pair(dilation)
#         self.groups = groups
#         self.bias = bias
#         self.padding_mode = padding_mode
#         self.num_experts = num_experts
#         self.dropout_rate = dropout_rate

#         # 全局平均池化
#         self._avg_pooling = lambda x: F.adaptive_avg_pool2d(x, output_size=(1, 1))

#         # 注意力全连接层
#         self._coefficient_fn = self._create_coefficient_fn(in_channels, num_experts, dropout_rate)

#         # 多套卷积层的权重
#         self.weight = nn.Parameter(torch.Tensor(num_experts, out_channels, in_channels // groups, *self.kernel_size))
#         if bias:
#             self.bias = nn.Parameter(torch.Tensor(out_channels))
#         else:
#             self.register_parameter('bias', None)

#         self.reset_parameters()

#     def _create_coefficient_fn(self, in_channels, num_experts, dropout_rate):
#         return nn.Sequential(
#             nn.Linear(in_channels, num_experts),
#             nn.Softmax(dim=1)
#         )

#     def reset_parameters(self):
#         nn.init.kaiming_uniform_(self.weight, a=nn.init.calculate_gain('relu'))
#         if self.bias is not None:
#             nn.init.zeros_(self.bias)

#     def _conv_forward(self, input, weight):
#         if self.padding_mode != 'zeros':
#             return F.conv2d(F.pad(input, (1, 1, 1, 1), mode=self.padding_mode),
#                             weight, self.bias, self.stride,
#                             _pair(0), self.dilation, self.groups)
#         return F.conv2d(input, weight, self.bias, self.stride,
#                         self.padding, self.dilation, self.groups)

#     def forward(self, inputs): # [b, c, h, w]
#         # 全局平均池化
#         pooled_inputs = self._avg_pooling(inputs) # [b, c, 1, 1]
#         pooled_inputs = pooled_inputs.view(-1, self.in_channels) # [b, c]

#         # 注意力权重计算
#         routing_weights = self._coefficient_fn(pooled_inputs) # [b, num_experts]
#         import pdb; pdb.set_trace()

#         # 卷积核加权求和
#         kernels = torch.einsum('bn, nchw->bchw', routing_weights, self.weight)

#         # 批量卷积操作
#         b, c, h, w = inputs.shape
#         outputs = []
#         for i in range(b):
#             out = self._conv_forward(inputs[i].unsqueeze(0), kernels[i])
#             outputs.append(out)
#         return torch.cat(outputs, dim=0)

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.utils import _pair


class DyNet2D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1,
                 bias=True, padding_mode='reflect', num_experts=3, dropout_rate=0.2):
        super(DyNet2D, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = _pair(kernel_size)
        self.stride = _pair(stride)
        self.padding = _pair(padding)
        self.dilation = _pair(dilation)
        self.groups = groups
        self.num_experts = num_experts

        # 全局平均池化
        self._avg_pooling = lambda x: F.adaptive_avg_pool2d(x, output_size=(1, 1))

        # 注意力计算
        self._coefficient_fn = nn.Sequential(
            nn.Linear(in_channels, num_experts),
            nn.Softmax(dim=1)
        )

        # 动态卷积权重
        self.weight = nn.Parameter(torch.Tensor(num_experts, out_channels, in_channels // groups, *self.kernel_size))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=nn.init.calculate_gain('relu'))
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, inputs):  # [b, c, h, w]
        b, c, h, w = inputs.shape

        # **1. 计算注意力权重**
        pooled_inputs = self._avg_pooling(inputs)  # [b, c, 1, 1]       torch.Size([16, 3, 1, 1])
        # import pdb; pdb.set_trace()
        pooled_inputs = pooled_inputs.view(b, c)  # [b, c]      torch.Size([16, 3])
        routing_weights = self._coefficient_fn(pooled_inputs)  # [b, num_experts]   torch.Size([16, 3])

        # **2. 计算动态卷积核**
        kernels = torch.einsum('bn, nochw -> bochw', routing_weights, self.weight)  # [b, out_channels, in_channels, kH, kW] torch.Size([16, 64, 3, 3, 3])

        # **3. 进行批量卷积**
        inputs = inputs.reshape(1, b * c, h, w)  # 变换成 `[1, batch * c, h, w]`  torch.Size([1, 48, 48, 48])
        kernels = kernels.reshape(b * self.out_channels, self.in_channels // self.groups, *self.kernel_size)  # `[b*out_c, in_c, kH, kW]` torch.Size([1024, 3, 3, 3])
        outputs = F.conv2d(F.pad(inputs, (1,1,1,1), mode="reflect"), kernels, self.bias.repeat(b), self.stride, self.padding, self.dilation, self.groups * b)
        
        return outputs.reshape(b, self.out_channels, outputs.shape[-2], outputs.shape[-1])
