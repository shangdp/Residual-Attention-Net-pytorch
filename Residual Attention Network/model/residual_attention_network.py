import torch
import torch.nn as nn
from torch.nn import init
import functools
from torch.autograd import Variable
import numpy as np
from .basic_layers import *
from .attention_module import AttentionModuleStage1, AttentionModuleStage2, AttentionModuleStage3, AttentionModule_stage0
from .attention_module import AttentionModule_stage1_cifar, AttentionModule_stage2_cifar, AttentionModule_stage3_cifar
'''

class ResidualAttentionModel_448input(nn.Module):
    # for input size 448
    def __init__(self):
        super(ResidualAttentionModel_448input, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias = False),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        self.mpool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        # tbq add
        # 112*112
        self.residual_block0 = ResidualBlock(64, 128)
        self.attention_module0 = AttentionModule_stage0(128, 128)
        # tbq add end
        self.residual_block1 = ResidualBlock(128, 256, 2)
        # 56*56
        self.attention_module1 = AttentionModule_stage1(256, 256)
        self.residual_block2 = ResidualBlock(256, 512, 2)
        self.attention_module2 = AttentionModule_stage2(512, 512)
        self.attention_module2_2 = AttentionModule_stage2(512, 512)  # tbq add
        self.residual_block3 = ResidualBlock(512, 1024, 2)
        self.attention_module3 = AttentionModule_stage3(1024, 1024)
        self.attention_module3_2 = AttentionModule_stage3(1024, 1024)  # tbq add
        self.attention_module3_3 = AttentionModule_stage3(1024, 1024)  # tbq add
        self.residual_block4 = ResidualBlock(1024, 2048, 2)
        self.residual_block5 = ResidualBlock(2048, 2048)
        self.residual_block6 = ResidualBlock(2048, 2048)
        self.mpool2 = nn.Sequential(
            nn.BatchNorm2d(2048),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=7, stride=1)
        )
        self.fc = nn.Linear(2048,10)

    def forward(self, x):
        out = self.conv1(x)
        out = self.mpool1(out)
        out = self.residual_block0(out)
        out = self.attention_module0(out)
        # print(out.data)
        out = self.residual_block1(out)
        out = self.attention_module1(out)
        out = self.residual_block2(out)
        out = self.attention_module2(out)
        out = self.attention_module2_2(out)
        out = self.residual_block3(out)
        # print(out.data)
        out = self.attention_module3(out)
        out = self.attention_module3_2(out)
        out = self.attention_module3_3(out)
        out = self.residual_block4(out)
        out = self.residual_block5(out)
        out = self.residual_block6(out)
        out = self.mpool2(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)

        return out


class ResidualAttentionModel_92(nn.Module):
    # for input size 224
    def __init__(self):
        super(ResidualAttentionModel_92, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias = False),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        self.mpool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.residual_block1 = `(64, 256)
        self.attention_module1 = AttentionModule_stage1(256, 256)
        self.residual_block2 = ResidualBlock(256, 512, 2)
        self.attention_module2 = AttentionModule_stage2(512, 512)
        self.attention_module2_2 = AttentionModule_stage2(512, 512)  # tbq add
        self.residual_block3 = ResidualBlock(512, 1024, 2)
        self.attention_module3 = AttentionModule_stage3(1024, 1024)
        self.attention_module3_2 = AttentionModule_stage3(1024, 1024)  # tbq add
        self.attention_module3_3 = AttentionModule_stage3(1024, 1024)  # tbq add
        self.residual_block4 = ResidualBlock(1024, 2048, 2)
        self.residual_block5 = ResidualBlock(2048, 2048)
        self.residual_block6 = ResidualBlock(2048, 2048)
        self.mpool2 = nn.Sequential(
            nn.BatchNorm2d(2048),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=7, stride=1)
        )
        self.fc = nn.Linear(2048,10)

    def forward(self, x):
        out = self.conv1(x)
        out = self.mpool1(out)
        # print(out.data)
        out = self.residual_block1(out)
        out = self.attention_module1(out)
        out = self.residual_block2(out)
        out = self.attention_module2(out)
        out = self.attention_module2_2(out)
        out = self.residual_block3(out)
        # print(out.data)
        out = self.attention_module3(out)
        out = self.attention_module3_2(out)
        out = self.attention_module3_3(out)
        out = self.residual_block4(out)
        out = self.residual_block5(out)
        out = self.residual_block6(out)
        out = self.mpool2(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)

        return out

'''
class ResidualAttentionModel_56(nn.Module):
    # for input size 224
    def __init__(self):
        super(ResidualAttentionModel_56, self).__init__()
        self.upsample = nn.UpsamplingBilinear2d(size=(224, 224))
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias = False),#input 224*224  output 112*112
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        self.pool_stride2 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, padding=1, stride=2),  # 56*56
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        self.residual_block64to256 = ResidualBlockSample(64, 256)
        self.residual_block256to256 = ResidualBlockDeep(256, 256)
        self.residual_block256to256_bn = ResidualBlockDeepBN(256, 256)
        self.residual_block256to512 = ResidualBlockDeeper(256, 512)
        self.residual_block512to512_bn = ResidualBlockDeepBN(512, 512)
        self.residual_block512to1024 = ResidualBlockDeeper(512, 1024)
        self.residual_block1024to1024_bn = ResidualBlockDeepBN(1024, 1024)
        self.residual_block1024to2048 = ResidualBlockDeeper(1024, 2048)
        self.residual_block2048to2048_bn = ResidualBlockDeepBN(2048, 2048)
        self.bn256 = nn.Sequential(
            nn.BatchNorm2d(256),
            nn.ReLU()
        )
        self.bn512 = nn.Sequential(
            nn.BatchNorm2d(512),
            nn.ReLU()
        )
        self.bn1024 = nn.Sequential(
            nn.BatchNorm2d(1024),
            nn.ReLU()
        )
        self.bn2048 = nn.Sequential(
            nn.BatchNorm2d(2048),
            nn.ReLU()
        )
        self.attention_module1 = AttentionModuleStage1(256, 256)
        self.attention_module2 = AttentionModuleStage2(512, 512)
        self.attention_module3 = AttentionModuleStage3(1024, 1024)
        self.ave_pool = nn.AvgPool2d(7, 1)
        self.fc = nn.Linear(2048, 10)
        self.softmax = nn.Softmax()

    def forward(self, x):
        out = self.upsample(x)
        out = self.conv1(out)
        out = self.pool_stride2(out)
        out = self.residual_block64to256(out)
        out = self.residual_block256to256_bn(out)
        out = self.attention_module1(out)
        out = self.residual_block256to256_bn(out)
        out = self.bn256(out)
        out = self.residual_block256to512(out)
        out = self.residual_block512to512_bn(out)
        out = self.attention_module2(out)
        out = self.residual_block512to512_bn(out)
        out = self.bn512(out)
        out = self.residual_block512to1024(out)
        out = self.residual_block1024to1024_bn(out)
        out = self.attention_module3(out)
        out = self.residual_block1024to1024_bn(out)
        out = self.bn1024(out)
        out = self.residual_block1024to2048(out)
        out = self.residual_block2048to2048_bn(out)
        out = self.residual_block2048to2048_bn(out)
        out = self.bn2048(out)
        out = self.ave_pool(out)
        out = out.view(out.size(0),-1)
        out = self.fc(out)
        out = self.softmax(out)
        return out

'''
class ResidualAttentionModel_92_32input(nn.Module):
    # for input size 32
    def __init__(self):
        super(ResidualAttentionModel_92_32input, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=5, stride=1, padding=2, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU()
        )  # 32*32
        self.mpool1 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)  # 16*16
        self.residual_block1 = ResidualBlock(32, 128)  # 16*16
        self.attention_module1 = AttentionModule_stage1_cifar(128, 128)  # 16*16
        self.residual_block2 = ResidualBlock(128, 256, 2)  # 8*8
        self.attention_module2 = AttentionModule_stage2_cifar(256, 256)  # 8*8
        self.attention_module2_2 = AttentionModule_stage2_cifar(256, 256)  # 8*8 # tbq add
        self.residual_block3 = ResidualBlock(256, 512, 2)  # 4*4
        self.attention_module3 = AttentionModule_stage3_cifar(512, 512)  # 4*4
        self.attention_module3_2 = AttentionModule_stage3_cifar(512, 512)  # 4*4 # tbq add
        self.attention_module3_3 = AttentionModule_stage3_cifar(512, 512)  # 4*4 # tbq add
        self.residual_block4 = ResidualBlock(512, 1024)  # 4*4
        self.residual_block5 = ResidualBlock(1024, 1024)  # 4*4
        self.residual_block6 = ResidualBlock(1024, 1024)  # 4*4
        self.mpool2 = nn.Sequential(
            nn.BatchNorm2d(1024),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=4, stride=1)
        )
        self.fc = nn.Linear(1024,10)

    def forward(self, x):
        out = self.conv1(x)
        out = self.mpool1(out)
        # print(out.data)
        out = self.residual_block1(out)
        out = self.attention_module1(out)
        out = self.residual_block2(out)
        out = self.attention_module2(out)
        out = self.attention_module2_2(out)
        out = self.residual_block3(out)
        # print(out.data)
        out = self.attention_module3(out)
        out = self.attention_module3_2(out)
        out = self.attention_module3_3(out)
        out = self.residual_block4(out)
        out = self.residual_block5(out)
        out = self.residual_block6(out)
        out = self.mpool2(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)

        return out
'''