import torch.nn as nn
from collections import OrderedDict
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

def make_layers(block):
    layers = []
    for layer_name, v in block.items():
        if 'pool' in layer_name:
            layer = nn.MaxPool2d(kernel_size=v[0], stride=v[1], padding=v[2])
            layers.append((layer_name, layer))
            
        elif 'deconv' in layer_name:
            if 'sample' in layer_name: # *downsample?h,w|
                if 'upsample' in layer_name:
                    layers.append(('upsample_' + layer_name, nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)))
                else:
                    layers.append(('downsample_' + layer_name, nn.Upsample(scale_factor=0.5, mode='bilinear', align_corners=True)))
                    
            transposeConv2d = nn.ConvTranspose2d(in_channels=v[0],
                                                 out_channels=v[1],
                                                 kernel_size=v[2],
                                                 stride=v[3],
                                                 padding=v[4])
            layers.append((layer_name, transposeConv2d))
            
            if 'batchnorm(' in layer_name:
                start_idx = layer_name.find("batchnorm(")
                end_idx = layer_name.find(")")
                channels = int(layer_name[start_idx+10:end_idx])
                layers.append(('batchnorm_' + layer_name, nn.BatchNorm2d(channels)))
            if 'relu' in layer_name:
                layers.append(('relu_' + layer_name, nn.ReLU(inplace=True)))
            elif 'leaky' in layer_name:
                layers.append(('leaky_' + layer_name,
                               nn.LeakyReLU(negative_slope=0.2, inplace=True)))
        elif 'conv' in layer_name:
            conv2d = nn.Conv2d(in_channels=v[0],
                               out_channels=v[1],
                               kernel_size=v[2],
                               stride=v[3],
                               padding=v[4])
            layers.append((layer_name, conv2d))
            if 'sample' in layer_name: # *downsample?h,w|
                if 'upsample' in layer_name:
                    layers.append(('upsample_' + layer_name, nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)))
                else:
                    layers.append(('downsample_' + layer_name, nn.Upsample(scale_factor=0.5, mode='bilinear', align_corners=True)))
            if 'batchnorm(' in layer_name:
                start_idx = layer_name.find("batchnorm(")
                end_idx = layer_name.find(")")
                channels = int(layer_name[start_idx+10:end_idx])
                layers.append(('batchnorm_' + layer_name, nn.BatchNorm2d(channels)))
            if 'relu' in layer_name:
                layers.append(('relu_' + layer_name, nn.ReLU(inplace=True)))
            elif 'leaky' in layer_name:
                layers.append(('leaky_' + layer_name,
                               nn.LeakyReLU(negative_slope=0.2, inplace=True)))
        else:
            raise NotImplementedError
    return nn.Sequential(OrderedDict(layers))

def get_batch(data_size, batch_size, dataloader, batch_time=1, seq=10):
    # t = tqdm(dataloader, leave=False, total=len(dataloader))
    for i, (input_frames, output_frames, rollout_results) in enumerate(dataloader):
        # input_frames  --> batch_size x input_frames x c x 64 x 64; input first input_frames frames
        # output_frames --> batch_size x output_frames x c x 64 x 64; output first input_frames frames
        yield (input_frames, i, output_frames)