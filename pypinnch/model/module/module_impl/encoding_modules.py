




import torch.nn as nn
from torch import randn

from .rff.functional import (
    basic_encoding,
    positional_encoding,
    gaussian_encoding,
)


class BasicEncodingModule(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, x):
        return basic_encoding(x)



class PositionalEncodingModule(nn.Module):

    def __init__(self, sigma, m):
        super().__init__()
        self.sigma = sigma
        self.m = m

    def forward(self, x):
        return positional_encoding(x, self.sigma, self.m)



class GaussianEncodingModule(nn.Module):

    def __init__(self, indim, sigma, m, b, dtype):
        super().__init__()
        b_ = randn(size=(m, indim), dtype=dtype)*sigma if b is None else b
        self.register_buffer('b', b_)

    def forward(self, x):
        return gaussian_encoding(x, self.b)



