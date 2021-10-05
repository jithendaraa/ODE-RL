import torch
import torch.nn as nn
import torch.nn.functional as F

class DS2VAE_ED(nn.Module):
    def __init__(self):
        super(DS2VAE_ED, self).__init__()


class C3DEncoder(nn.Module):
    def __init__(self, in_channels, out_channels, mode='', instance_norm=False, kernel_size=(3, 4, 4), stride=(1, 2, 2), padding=(0, 1, 1), affine=False):
        super(C3DEncoder, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.mode = mode
        self.instance_norm = instance_norm
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.affine = affine

        self.c3d_net = self.init_conv_3d_net()

        if mode in ['static', 'dynamic']:
            self.mu_net = nn.Linear(self.out_channels, self.out_channels)
            self.logvar_net = nn.Linear(self.out_channels, self.out_channels)
    
    def init_conv_3d_net(self):
        if self.instance_norm is True:
            c3d_net = nn.Sequential(
                nn.Conv3d(self.in_channels, 64, self.kernel_size, self.stride, self.padding), nn.InstanceNorm3d(64, affine=self.affine), nn.LeakyReLU(0.2),
                nn.Conv3d(64, 128, self.kernel_size, self.stride, self.padding), nn.InstanceNorm3d(128, affine=self.affine), nn.LeakyReLU(0.2),
                nn.Conv3d(128, 256, self.kernel_size, self.stride, self.padding), nn.InstanceNorm3d(256, affine=self.affine), nn.LeakyReLU(0.2),
                nn.Conv3d(256, 512, self.kernel_size, self.stride, self.padding), nn.InstanceNorm3d(512, affine=self.affine), nn.LeakyReLU(0.2),
                nn.Conv3d(512, self.out_channels, self.kernel_size, self.stride, self.padding), nn.InstanceNorm3d(self.out_channels, affine=self.affine), nn.Tanh())
                
        else:
            c3d_net = nn.Sequential(
                nn.Conv3d(self.in_channels, 64, self.kernel_size, self.stride, self.padding), nn.LeakyReLU(0.2),
                nn.Conv3d(64, 128, self.kernel_size, self.stride, self.padding), nn.LeakyReLU(0.2),
                nn.Conv3d(128, 256, self.kernel_size, self.stride, self.padding), nn.LeakyReLU(0.2),
                nn.Conv3d(256, 512, self.kernel_size, self.stride, self.padding), nn.LeakyReLU(0.2),
                nn.Conv3d(512, self.out_channels, self.kernel_size, self.stride, self.padding), nn.Tanh())

        return c3d_net
    
    def forward(self, x):
        enc_x = self.c3d_net(x)
        return enc_x
        


