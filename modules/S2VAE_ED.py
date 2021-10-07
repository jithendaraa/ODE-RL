import torch
import torch.nn as nn
import torch.nn.functional as F

class CNNDecoder(nn.Module):
    def __init__(self, in_channels, out_channels, model='s2vae', unmasked=True):
        super(CNNDecoder, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.model = model
        self.unmasked = unmasked

        if unmasked is False: self.out_channels += 1 # Include mask channel
        self.init_deconv_net()

    def init_deconv_net(self):
        if self.model in ['s2vae']:
            self.deconv_net = nn.Sequential(
                    nn.ConvTranspose2d(self.in_channels, 256, 4, 1, 0), nn.BatchNorm2d(256), nn.LeakyReLU(0.2),
                    nn.Upsample(scale_factor=2),
                    nn.Conv2d(256, 128, 3, 1, 1), nn.BatchNorm2d(128), nn.LeakyReLU(0.2),
                    nn.Upsample(scale_factor=2),
                    nn.Conv2d(128, 64, 3, 1, 1), nn.BatchNorm2d(64), nn.LeakyReLU(0.2),
                    nn.Upsample(scale_factor=2),
                    nn.Conv2d(64, 32, 3, 1, 1), nn.BatchNorm2d(32), nn.LeakyReLU(0.2),
                    nn.Upsample(scale_factor=2),
                    nn.Conv2d(32, 16, 3, 1, 1), nn.BatchNorm2d(16), nn.LeakyReLU(0.2),
                    nn.Conv2d(16, self.out_channels, 1, 1, 0), nn.Sigmoid())
        
        elif self.model in ['cs2vae']:
            self.deconv_net = nn.Sequential(
                    nn.ConvTranspose2d(self.in_channels, 256, 3, 1, 1), nn.BatchNorm2d(256), nn.LeakyReLU(0.2),
                    nn.Upsample(scale_factor=2),
                    nn.Conv2d(256, 128, 3, 1, 1), nn.BatchNorm2d(128), nn.LeakyReLU(0.2),
                    nn.Upsample(scale_factor=2),
                    nn.Conv2d(128, 64, 3, 1, 1), nn.BatchNorm2d(64), nn.LeakyReLU(0.2),
                    nn.Upsample(scale_factor=2),
                    nn.Conv2d(64, 32, 3, 1, 1), nn.BatchNorm2d(32), nn.LeakyReLU(0.2),
                    nn.Upsample(scale_factor=2),
                    nn.Conv2d(32, 16, 3, 1, 1), nn.BatchNorm2d(16), nn.LeakyReLU(0.2),
                    nn.Conv2d(16, self.out_channels, 1, 1, 0), nn.Sigmoid())


    def forward(self, x):
        print("x", x.size())
        res = self.deconv_net(x)
        print("res", res.size())
        return res
