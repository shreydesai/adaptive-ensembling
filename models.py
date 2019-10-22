import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm
import numpy as np

from adaptive import AdaptiveConv1d, AdaptiveLinear


class CNN(nn.Module):
    def __init__(
        self,
        input_size,
        vocab_size,
        hidden_size,
        output_size,
        max_len,
        kernel_size,
        dropout,
        pretrained,
    ):
        super().__init__()

        self.input_size = input_size
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.max_len = max_len
        self.kernel_size = kernel_size

        self.embedding = nn.Embedding.from_pretrained(pretrained, freeze=False)
        self.width = self._width()

        self.conv = nn.Sequential(*self._conv_layers())
        self.dropout = nn.Dropout(dropout)

    def _conv_layers(self):
        layers = []
        c_in, c_out = self.input_size, self.hidden_size
        n = int(np.log2(self.max_len)) 
        for i in range(n):
            layers.append(
                nn.Conv1d(c_in, c_out, self.kernel_size, dilation=2 ** i)
            )
            layers.append(nn.ReLU(True))
            c_in, c_out = c_out, self.hidden_size
        layers.append(nn.Conv1d(self.hidden_size, self.output_size, 1))
        return layers
    
    def _width(self):
        width = 0
        for i in range(int(np.log2(self.max_len))):
            width += (2 ** i) * (self.kernel_size - 1)
        return width - self.max_len + 1
    
    def add_noise(self, x, mean=0., std=0.1):
        B, E, T = x.size()
        noise = torch.FloatTensor(np.random.normal(mean, std, (B, E, T)))
        noise.requires_grad_(False)
        if torch.cuda.is_available():
            noise = noise.cuda()
        return x + noise

    def forward(self, x):
        x = F.pad(x, (self.width, 0))
        x = self.dropout(self.embedding(x)).permute(0, 2, 1)  # [B,E,T]
        if self.training:
            x = self.add_noise(x)
        res = self.conv(x).squeeze(-1)
        return res


class AdaptiveCNN(nn.Module):
    def __init__(
        self,
        input_size,
        vocab_size,
        hidden_size,
        output_size,
        max_len,
        kernel_size,
        dropout,
        pretrained,
    ):
        super().__init__()

        self.input_size = input_size
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.max_len = max_len
        self.kernel_size = kernel_size

        self.embedding = nn.Embedding.from_pretrained(pretrained, freeze=False)
        self.dropout = nn.Dropout(dropout)
        self.width = self._width()

        ks = kernel_size
        self.conv_layers = []

        c_in, c_out = self.input_size, self.hidden_size
        for i in range(7):
            conv_layers.append(
                AdaptiveConv1d(c_in, c_out, ks, dilation=2 ** i, num=i)
            )
            c_in, c_out = c_out, self.hidden_size
        conv_layers.append(
            AdaptiveConv1d(c_out, output_size, 1, num=7)
        )

    def _width(self):
        width = 0
        for i in range(int(np.log2(self.max_len))):
            width += (2 ** i) * (self.kernel_size - 1)
        return width - self.max_len + 1
    
    def add_noise(self, x, mean=0., std=0.1):
        B, E, T = x.size()
        noise = torch.FloatTensor(np.random.normal(mean, std, (B, E, T)))
        noise.requires_grad_(False)
        if torch.cuda.is_available():
            noise = noise.cuda()
        return x + noise
    
    def forward(self, x, student=None):
        x = F.pad(x, (self.width, 0))
        x = self.dropout(self.embedding(x)).permute(0, 2, 1)  # [B, E, T]
        if self.training:
            x = self.add_noise(x)
        for conv in self.conv_layers[:-1]:
            x = self.relu(conv(x))
        x = self.conv8(x, student).squeeze(-1)
        return x
