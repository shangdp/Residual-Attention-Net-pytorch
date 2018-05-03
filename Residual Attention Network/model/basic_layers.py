import torch
import torch.nn as nn
from torch.nn import init
import functools
from torch.autograd import Variable
import numpy as np


class ResidualBlockSample(nn.Module):
    def __init__(self, input_channels, output_channels):
        super(ResidualBlockSample, self).__init__()
        self.conv_res = nn.Sequential(
            nn.Conv2d(input_channels, input_channels, 1, 1, bias=False),
            nn.BatchNorm2d(input_channels),
            nn.ReLU(),
            nn.Conv2d(input_channels, input_channels, 3, 1, padding=1, bias = False),
            nn.BatchNorm2d(input_channels),
            nn.ReLU(),
            nn.Conv2d(input_channels, output_channels, 1, 1, bias=False)
        )
        self.conv_identity = nn.Conv2d(input_channels, output_channels, 1, 1, bias=False)

    def forward(self, x):
        res = self.conv_res(x)
        identity = self.conv_identity(x)
        out = identity + res
        return out


class ResidualBlockDeep(nn.Module):
    def __init__(self, input_channels, output_channels):
        super(ResidualBlockDeep, self).__init__()
        self.conv_res = nn.Sequential(
            nn.Conv2d(input_channels, int(input_channels / 4), 1, 1, bias=False),
            nn.BatchNorm2d(int(input_channels / 4)),
            nn.ReLU(),
            nn.Conv2d(int(input_channels / 4), int(input_channels / 4), 3, 1, padding=1, bias = False),
            nn.BatchNorm2d(int(input_channels / 4)),
            nn.ReLU(),
            nn.Conv2d(int(input_channels / 4), output_channels, 1, 1, bias=False)
        )

    def forward(self, x):
        res = self.conv_res(x)
        out = x + res
        return out


class ResidualBlockDeepBN(nn.Module):
    def __init__(self, input_channels, output_channels):
        super(ResidualBlockDeepBN, self).__init__()
        self.conv_res = nn.Sequential(
            nn.BatchNorm2d(input_channels),
            nn.ReLU(),
            nn.Conv2d(input_channels, int(input_channels / 4), 1, 1, bias=False),
            nn.BatchNorm2d(int(input_channels / 4)),
            nn.ReLU(),
            nn.Conv2d(int(input_channels / 4), int(input_channels / 4), 3, 1, padding=1, bias = False),
            nn.BatchNorm2d(int(input_channels / 4)),
            nn.ReLU(),
            nn.Conv2d(int(input_channels / 4), output_channels, 1, 1, bias=False)
        )

    def forward(self, x):
        res = self.conv_res(x)
        out = x + res
        return out


class ResidualBlockDeeper(nn.Module):
    def __init__(self, input_channels, output_channels):
        super(ResidualBlockDeeper, self).__init__()
        self.conv_res = nn.Sequential(
            nn.Conv2d(input_channels, int(input_channels / 2), 1, 1, bias=False),
            nn.BatchNorm2d(int(input_channels / 2)),
            nn.ReLU(),
            nn.Conv2d(int(input_channels / 2), int(input_channels / 2), 3, 2, padding=1, bias = False),
            nn.BatchNorm2d(int(input_channels / 2)),
            nn.ReLU(),
            nn.Conv2d(int(input_channels / 2), output_channels, 1, 1, bias=False)
        )
        self.conv_identity = nn.Conv2d(input_channels, output_channels, 1, 2, bias=False)

    def forward(self, x):
        res = self.conv_res(x)
        identity = self.conv_identity(x)
        out = identity + res
        return out


