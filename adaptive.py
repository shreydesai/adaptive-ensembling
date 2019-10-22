import torch
import torch.nn as nn
import torch.nn.functional as F


class _AdaptiveConv(nn.Module):
    def init_params(self, in_channels, out_channels, kernel_size, num):
        self.w_consts = nn.Parameter(torch.Tensor(
            out_channels, in_channels, kernel_size
        ))
        self.b_consts = nn.Parameter(torch.Tensor(out_channels))
        self.num = num

        self.init_weights()

    def init_weights(self, c=0.9):
        nn.init.constant_(self.w_consts, c)
        nn.init.constant_(self.b_consts, c)
    
    def transform_weight(self, student):
        if student:
            w_t = self.weight
            w_s = student.conv_layers[self.num].weight
            return torch.mul(w_t, self.w_consts) + \
                   torch.mul(w_s, 1.-self.w_consts)
        return self.weight
    
    def transform_bias(self, student):
        if student:
            b_t = self.bias
            b_s = student.conv_layers[self.num].bias
            return torch.mul(b_t, self.b_consts) + \
                   torch.mul(b_s, 1.-self.b_consts)
        return self.bias


class AdaptiveConv1d(nn.Conv1d, _AdaptiveConv):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        padding=0,
        dilation=1,
        bias=True,
        num=0
    ):
        super().__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride=1,
            padding=padding,
            dilation=dilation,
            bias=bias,
        )
        self.init_params(in_channels, out_channels, kernel_size, num)
    
    def forward(self, input, student):
        return F.conv1d(
            input,
            self.transform_weight(student),
            self.transform_bias(student),
            self.stride,
            self.padding,
            self.dilation,
        )


class _AdaptiveLinear(nn.Module):
    def init_params(self, in_feats, out_feats, num):
        self.w_consts = nn.Parameter(torch.Tensor(out_feats, in_feats))
        self.b_consts = nn.Parameter(torch.Tensor(out_feats))
        self.num = num

        self.init_weights()
    
    def init_weights(self, c=0.9):
        nn.init.constant_(self.w_consts, c)
        nn.init.constant_(self.b_consts, c)
    
    def transform_weight(self, student):
        if student:
            w_t = self.weight.data
            w_s = student.linear_layers[self.num].weight.data
            return torch.mul(w_t, self.w_consts.data) + \
                   torch.mul(w_s, 1. - self.w_consts.data)
        return self.weight.data
    
    def transform_bias(self, student):
        if student:
            b_t = self.bias.data
            b_s = student.linear_layers[self.num].bias.data
            return torch.mul(b_t, self.b_consts.data) + \
                   torch.mul(b_s, 1. - self.b_consts.data)
        return self.bias.data


class AdaptiveLinear(nn.Linear, _AdaptiveLinear):
    def __init__(self, in_feats, out_feats, bias=True, num=0):
        super().__init__(in_feats, out_feats, bias)
        self.init_params(in_feats, out_feats, num)
    
    def forward(self, input, student):
        return F.linear(
            input,
            self.transform_weight(student),
            self.transform_bias(student),
        )
