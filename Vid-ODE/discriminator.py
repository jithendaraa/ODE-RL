import torch
import torch.nn as nn
import torch.nn.functional as F

class Discriminator_IMG():
    def __init__(self, h, w, c):
        super(Discriminator_IMG, self).__init__()

        self.conv1 = nn.Conv2d(c, 64, 4, 2, 1) 
        self.downsample1 = nn.Upsample(scale_factor=0.5, mode='bilinear', align_corners=True) # apply leaky ReLU
        self.conv2 = nn.Conv2d(64, 128, 4, 2, 1)
        self.downsample2 = nn.Upsample(scale_factor=0.5, mode='bilinear', align_corners=True) # apply leaky ReLU
        self.conv3 = nn.Conv2d(128, 256, 4, 2, 1)
        self.downsample3 = nn.Upsample(scale_factor=0.5, mode='bilinear', align_corners=True) # apply leaky ReLU
        self.conv4 = nn.Conv2d(256, 512, 4, 1, 2)   # apply leaky ReLU
        self.conv5 = nn.Conv2d(512, 64, 4, 1, 2)
    
    def forward(self, x):
        x = self.downsample1(self.conv1(x))
        x = F.leaky_relu_(x, negative_slope=0.2)
        x = self.downsample2(self.conv2(x))
        x = F.leaky_relu_(x, negative_slope=0.2)
        x = self.downsample3(self.conv3(x))
        x = F.leaky_relu_(x, negative_slope=0.2)
        x = self.conv4(x)
        x = F.leaky_relu_(x, negative_slope=0.2)
        x = self.conv5(x)
        return x

