import sys
sys.path.append('..')

import torch
import torch.nn as nn

import helpers.utils as utils

class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)

class ResidualBlock(nn.Module):
    def __init__(self,
                 in_channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        out = nn.ReLU()(x)
        out = self.conv1(out)
        out = nn.ReLU()(out)
        out = self.conv2(out)
        return out + x

class ImpalaBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ImpalaBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1)
        self.res1 = ResidualBlock(out_channels)
        self.res2 = ResidualBlock(out_channels)

    def forward(self, x):
        x = self.conv(x)
        x = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)(x)
        x = self.res1(x)
        x = self.res2(x)
        return x

class ImpalaModel(nn.Module):
    def __init__(self,
                 in_channels, flatten=True, depth=4,
                 **kwargs):
        super(ImpalaModel, self).__init__()
        self.flatten = flatten
        self.block1 = ImpalaBlock(in_channels=in_channels, out_channels=16)
        self.block2 = ImpalaBlock(in_channels=16, out_channels=32)
        self.block3 = ImpalaBlock(in_channels=32, out_channels=64)
        self.block4 = ImpalaBlock(in_channels=64, out_channels=128)
        
        if self.flatten is True:
            self.fc = nn.Linear(in_features=32 * 8 * 8, out_features=256)

        self.output_dim = 256
        self.apply(utils.xavier_uniform_init)

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = nn.ReLU()(x)
        
        if self.flatten is True:
            x = Flatten()(x)
            x = self.fc(x)
            x = nn.ReLU()(x)
        return x