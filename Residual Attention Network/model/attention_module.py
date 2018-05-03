import torch
import torch.nn as nn
from torch.nn import init
import functools
from torch.autograd import Variable
import numpy as np

from .basic_layers import *


class AttentionModule_pre(nn.Module):

    def __init__(self, in_channels, out_channels, size1, size2, size3):
        super(AttentionModule_pre, self).__init__()
        self.first_residual_blocks = ResidualBlock(in_channels, out_channels)

        self.trunk_branches = nn.Sequential(
            ResidualBlock(in_channels, out_channels),
            ResidualBlock(in_channels, out_channels)
         )

        self.mpool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.softmax1_blocks = ResidualBlock(in_channels, out_channels)

        self.skip1_connection_residual_block = ResidualBlock(in_channels, out_channels)

        self.mpool2 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.softmax2_blocks = ResidualBlock(in_channels, out_channels)

        self.skip2_connection_residual_block = ResidualBlock(in_channels, out_channels)

        self.mpool3 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.softmax3_blocks = nn.Sequential(
            ResidualBlock(in_channels, out_channels),
            ResidualBlock(in_channels, out_channels)
        )

        self.interpolation3 = nn.UpsamplingBilinear2d(size=size3)

        self.softmax4_blocks = ResidualBlock(in_channels, out_channels)

        self.interpolation2 = nn.UpsamplingBilinear2d(size=size2)

        self.softmax5_blocks = ResidualBlock(in_channels, out_channels)

        self.interpolation1 = nn.UpsamplingBilinear2d(size=size1)

        self.softmax6_blocks = nn.Sequential(
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels , kernel_size = 1, stride = 1, bias = False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels , kernel_size = 1, stride = 1, bias = False),
            nn.Sigmoid()
        )

        self.last_blocks = ResidualBlock(in_channels, out_channels)

    def forward(self, x):
        x = self.first_residual_blocks(x)
        out_trunk = self.trunk_branches(x)
        out_mpool1 = self.mpool1(x)
        out_softmax1 = self.softmax1_blocks(out_mpool1)
        out_skip1_connection = self.skip1_connection_residual_block(out_softmax1)
        out_mpool2 = self.mpool2(out_softmax1)
        out_softmax2 = self.softmax2_blocks(out_mpool2)
        out_skip2_connection = self.skip2_connection_residual_block(out_softmax2)
        out_mpool3 = self.mpool3(out_softmax2)
        out_softmax3 = self.softmax3_blocks(out_mpool3)
        #
        out_interp3 = self.interpolation3(out_softmax3)
        # print(out_skip2_connection.data)
        # print(out_interp3.data)
        out = out_interp3 + out_skip2_connection
        out_softmax4 = self.softmax4_blocks(out)
        out_interp2 = self.interpolation2(out_softmax4)
        out = out_interp2 + out_skip1_connection
        out_softmax5 = self.softmax5_blocks(out)
        out_interp1 = self.interpolation1(out_softmax5)
        out_softmax6 = self.softmax6_blocks(out_interp1)
        out = (1 + out_softmax6) * out_trunk
        out_last = self.last_blocks(out)

        return out_last


class AttentionModule_stage0(nn.Module):
    # input size is 112*112
    def __init__(self, in_channels, out_channels, size1=(112, 112), size2=(56, 56), size3=(28, 28), size4=(14, 14)):
        super(AttentionModule_stage0, self).__init__()
        self.first_residual_blocks = ResidualBlock(in_channels, out_channels)

        self.trunk_branches = nn.Sequential(
            ResidualBlock(in_channels, out_channels),
            ResidualBlock(in_channels, out_channels)
         )

        self.mpool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        # 56*56
        self.softmax1_blocks = ResidualBlock(in_channels, out_channels)

        self.skip1_connection_residual_block = ResidualBlock(in_channels, out_channels)

        self.mpool2 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        # 28*28
        self.softmax2_blocks = ResidualBlock(in_channels, out_channels)

        self.skip2_connection_residual_block = ResidualBlock(in_channels, out_channels)

        self.mpool3 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        # 14*14
        self.softmax3_blocks = ResidualBlock(in_channels, out_channels)
        self.skip3_connection_residual_block = ResidualBlock(in_channels, out_channels)
        self.mpool4 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        # 7*7
        self.softmax4_blocks = nn.Sequential(
            ResidualBlock(in_channels, out_channels),
            ResidualBlock(in_channels, out_channels)
        )
        self.interpolation4 = nn.UpsamplingBilinear2d(size=size4)
        self.softmax5_blocks = ResidualBlock(in_channels, out_channels)
        self.interpolation3 = nn.UpsamplingBilinear2d(size=size3)
        self.softmax6_blocks = ResidualBlock(in_channels, out_channels)
        self.interpolation2 = nn.UpsamplingBilinear2d(size=size2)
        self.softmax7_blocks = ResidualBlock(in_channels, out_channels)
        self.interpolation1 = nn.UpsamplingBilinear2d(size=size1)

        self.softmax8_blocks = nn.Sequential(
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=1, stride=1, bias = False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels , kernel_size=1, stride=1, bias = False),
            nn.Sigmoid()
        )

        self.last_blocks = ResidualBlock(in_channels, out_channels)

    def forward(self, x):
        # 112*112
        x = self.first_residual_blocks(x)
        out_trunk = self.trunk_branches(x)
        out_mpool1 = self.mpool1(x)
        # 56*56
        out_softmax1 = self.softmax1_blocks(out_mpool1)
        out_skip1_connection = self.skip1_connection_residual_block(out_softmax1)
        out_mpool2 = self.mpool2(out_softmax1)
        # 28*28
        out_softmax2 = self.softmax2_blocks(out_mpool2)
        out_skip2_connection = self.skip2_connection_residual_block(out_softmax2)
        out_mpool3 = self.mpool3(out_softmax2)
        # 14*14
        out_softmax3 = self.softmax3_blocks(out_mpool3)
        out_skip3_connection = self.skip3_connection_residual_block(out_softmax3)
        out_mpool4 = self.mpool4(out_softmax3)
        # 7*7
        out_softmax4 = self.softmax4_blocks(out_mpool4)
        out_interp4 = self.interpolation4(out_softmax4) + out_softmax3
        out = out_interp4 + out_skip3_connection
        out_softmax5 = self.softmax5_blocks(out)
        out_interp3 = self.interpolation3(out_softmax5) + out_softmax2
        # print(out_skip2_connection.data)
        # print(out_interp3.data)
        out = out_interp3 + out_skip2_connection
        out_softmax6 = self.softmax6_blocks(out)
        out_interp2 = self.interpolation2(out_softmax6) + out_softmax1
        out = out_interp2 + out_skip1_connection
        out_softmax7 = self.softmax7_blocks(out)
        out_interp1 = self.interpolation1(out_softmax7) + out_trunk
        out_softmax8 = self.softmax8_blocks(out_interp1)
        out = (1 + out_softmax8) * out_trunk
        out_last = self.last_blocks(out)

        return out_last


class AttentionModuleStage1(nn.Module):
    # input size is 56*56
    def __init__(self, in_channels, out_channels, size=(56, 56)):
        super(AttentionModuleStage1, self).__init__()
        self.down_sample_pool = nn.MaxPool2d(3, 2, padding=1)
        self.residual_block = ResidualBlockDeepBN(in_channels, out_channels)
        self.up_sample = nn.Upsample(scale_factor=2, mode='bilinear')
        #self.up_sample2 = nn.UpsamplingBilinear2d(size=(size[0] / 2, size[1] / 2))
        #self.up_sample3 = nn.UpsamplingBilinear2d(size=size)
        self.mask = nn.Sequential(
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, 1, 1, 0, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, 1, 1, 0, bias=False),
            nn.Sigmoid()
        )
    def forward(self, x):
        attention = self.down_sample_pool(x)
        attention = self.residual_block(attention)
        attention_trunk = self.down_sample_pool(attention)
        attention_trunk = self.residual_block(attention_trunk)
        attention_trunk_branch1 = self.down_sample_pool(attention_trunk)
        attention_trunk_branch1 = self.residual_block(attention_trunk_branch1)
        attention_trunk_branch1 = self.residual_block(attention_trunk_branch1)
        attention_trunk_branch2 = self.residual_block(attention_trunk)
        attention_trunk = self.up_sample(attention_trunk_branch1) + attention_trunk + attention_trunk_branch2
        attention_trunk = self.residual_block(attention_trunk)
        attention_branch1 = self.residual_block(attention)
        attention = self.up_sample(attention_trunk) + attention_branch1 + attention
        attention = self.residual_block(attention)
        out = self.residual_block(x)
        out = self.residual_block(out)
        attention = out + self.up_sample(attention)
        attention = self.mask(attention)
        attention = attention * out
        out = attention + out
        return out


class AttentionModuleStage2(nn.Module):
    # input image size is 28*28
    def __init__(self, in_channels, out_channels, size=(28, 28)):
        super(AttentionModuleStage2, self).__init__()

        self.down_sample_pool = nn.MaxPool2d(3, 2, padding=1)
        self.residual_block = ResidualBlockDeepBN(in_channels, out_channels)
        self.up_sample = nn.Upsample(scale_factor=2, mode='bilinear')
        #self.up_sample1 = nn.UpsamplingBilinear2d(size=(size[0] / 2, size[1] / 2))
        #self.up_sample2 = nn.UpsamplingBilinear2d(size=size)
        self.mask = nn.Sequential(
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, 1, 1, 0, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, 1, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        attention = self.down_sample_pool(x)
        attention = self.residual_block(attention)
        attention_trunk_branch1 = self.down_sample_pool(attention)
        attention_trunk_branch1 = self.residual_block(attention_trunk_branch1)
        attention_trunk_branch1 = self.residual_block(attention_trunk_branch1)
        attention_trunk_branch2 = self.residual_block(attention)
        attention_trunk = self.up_sample(attention_trunk_branch1) + attention + attention_trunk_branch2
        attention = self.residual_block(attention_trunk)
        out = self.residual_block(x)
        out = self.residual_block(out)
        attention = out + self.up_sample(attention)
        attention = self.mask(attention)
        attention = attention * out
        out = attention + out
        return out


class AttentionModuleStage3(nn.Module):
    # input image size is 14*14
    def __init__(self, in_channels, out_channels, size=(14, 14)):
        super(AttentionModuleStage3, self).__init__()
        self.down_sample_pool = nn.MaxPool2d(3, 2, padding=1)
        self.residual_block = ResidualBlockDeepBN(in_channels, out_channels)
        self.up_sample = nn.Upsample(scale_factor=2, mode='bilinear')
        #self.up_sample = nn.UpsamplingBilinear2d(size=size)
        self.mask = nn.Sequential(
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, 1, 1, 0, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, 1, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        attention = self.down_sample_pool(x)
        attention = self.residual_block(attention)
        attention = self.residual_block(attention)
        out = self.residual_block(x)
        out = self.residual_block(out)
        attention = out + self.up_sample(attention)
        attention = self.mask(attention)
        attention = attention * out
        out = attention + out

        return out


class AttentionModule_stage1_cifar(nn.Module):
    # input size is 16*16
    def __init__(self, in_channels, out_channels, size1=(16, 16), size2=(8, 8)):
        super(AttentionModule_stage1_cifar, self).__init__()
        self.first_residual_blocks = ResidualBlock(in_channels, out_channels)

        self.trunk_branches = nn.Sequential(
            ResidualBlock(in_channels, out_channels),
            ResidualBlock(in_channels, out_channels)
         )

        self.mpool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)  # 8*8

        self.down_residual_blocks1 = ResidualBlock(in_channels, out_channels)

        self.skip1_connection_residual_block = ResidualBlock(in_channels, out_channels)

        self.mpool2 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)  # 4*4

        self.middle_2r_blocks = nn.Sequential(
            ResidualBlock(in_channels, out_channels),
            ResidualBlock(in_channels, out_channels)
        )

        self.interpolation1 = nn.UpsamplingBilinear2d(size=size2)  # 8*8

        self.up_residual_blocks1 = ResidualBlock(in_channels, out_channels)

        self.interpolation2 = nn.UpsamplingBilinear2d(size=size1)  # 16*16

        self.conv1_1_blocks = nn.Sequential(
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=1, stride=1, bias = False),
            nn.Sigmoid()
        )

        self.last_blocks = ResidualBlock(in_channels, out_channels)

    def forward(self, x):
        x = self.first_residual_blocks(x)
        out_trunk = self.trunk_branches(x)
        out_mpool1 = self.mpool1(x)
        out_down_residual_blocks1 = self.down_residual_blocks1(out_mpool1)
        out_skip1_connection = self.skip1_connection_residual_block(out_down_residual_blocks1)
        out_mpool2 = self.mpool2(out_down_residual_blocks1)
        out_middle_2r_blocks = self.middle_2r_blocks(out_mpool2)
        #
        out_interp = self.interpolation1(out_middle_2r_blocks) + out_down_residual_blocks1
        # print(out_skip2_connection.data)
        # print(out_interp3.data)
        out = out_interp + out_skip1_connection
        out_up_residual_blocks1 = self.up_residual_blocks1(out)
        out_interp2 = self.interpolation2(out_up_residual_blocks1) + out_trunk
        out_conv1_1_blocks = self.conv1_1_blocks(out_interp2)
        out = (1 + out_conv1_1_blocks) * out_trunk
        out_last = self.last_blocks(out)

        return out_last


class AttentionModule_stage2_cifar(nn.Module):
    # input size is 8*8
    def __init__(self, in_channels, out_channels, size=(8, 8)):
        super(AttentionModule_stage2_cifar, self).__init__()
        self.first_residual_blocks = ResidualBlock(in_channels, out_channels)

        self.trunk_branches = nn.Sequential(
            ResidualBlock(in_channels, out_channels),
            ResidualBlock(in_channels, out_channels)
         )

        self.mpool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)  # 4*4

        self.middle_2r_blocks = nn.Sequential(
            ResidualBlock(in_channels, out_channels),
            ResidualBlock(in_channels, out_channels)
        )

        self.interpolation1 = nn.UpsamplingBilinear2d(size=size)  # 8*8

        self.conv1_1_blocks = nn.Sequential(
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=1, stride=1, bias = False),
            nn.Sigmoid()
        )

        self.last_blocks = ResidualBlock(in_channels, out_channels)

    def forward(self, x):
        x = self.first_residual_blocks(x)
        out_trunk = self.trunk_branches(x)
        out_mpool1 = self.mpool1(x)
        out_middle_2r_blocks = self.middle_2r_blocks(out_mpool1)
        #
        out_interp = self.interpolation1(out_middle_2r_blocks) + out_trunk
        # print(out_skip2_connection.data)
        # print(out_interp3.data)
        out_conv1_1_blocks = self.conv1_1_blocks(out_interp)
        out = (1 + out_conv1_1_blocks) * out_trunk
        out_last = self.last_blocks(out)

        return out_last


class AttentionModule_stage3_cifar(nn.Module):
    # input size is 4*4
    def __init__(self, in_channels, out_channels, size=(8, 8)):
        super(AttentionModule_stage3_cifar, self).__init__()
        self.first_residual_blocks = ResidualBlock(in_channels, out_channels)

        self.trunk_branches = nn.Sequential(
            ResidualBlock(in_channels, out_channels),
            ResidualBlock(in_channels, out_channels)
         )

        self.middle_2r_blocks = nn.Sequential(
            ResidualBlock(in_channels, out_channels),
            ResidualBlock(in_channels, out_channels)
        )

        self.conv1_1_blocks = nn.Sequential(
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=1, stride=1, bias = False),
            nn.Sigmoid()
        )

        self.last_blocks = ResidualBlock(in_channels, out_channels)

    def forward(self, x):
        x = self.first_residual_blocks(x)
        out_trunk = self.trunk_branches(x)
        out_middle_2r_blocks = self.middle_2r_blocks(x)
        #
        out_conv1_1_blocks = self.conv1_1_blocks(out_middle_2r_blocks)
        out = (1 + out_conv1_1_blocks) * out_trunk
        out_last = self.last_blocks(out)

        return out_last