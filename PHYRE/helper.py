import torch
import torch.nn as nn
from collections import OrderedDict
import numpy as np
from tqdm import tqdm
import matplotlib
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

class PhyreRolloutDataset(torch.utils.data.Dataset):
    def __init__(self, start=0, end=0, batch_size=4, data_dir='/', INPUT_FRAMES=3, TOTAL_FRAMES=17, height=256, width=256, channels=3, device=None):
        self.start = start
        self.end = end
        self.batch_size = batch_size
        self.data_dir = data_dir
        self.input_frames = INPUT_FRAMES
        self.total_frames = TOTAL_FRAMES
        self.height, self.width, self.channels = height, width, channels
        self.device = device
        if end is None: self.rollout_results = np.load('rollout_results.npy')[start:]
        else: self.rollout_results = np.load('rollout_results.npy')[start:end]
        
    def __len__(self):
        return self.end - self.start
    
    def __getitem__(self, idx):
        rollout_data = np.array([])
        
        rollout_folder = self.data_dir + '/rollout_' + str(np.arange(self.start+1, self.end+1)[idx])
        sequence = np.array([])

        for i in range(1, self.total_frames+1):
            filepath = rollout_folder + '/frame_' + str(i) + '.jpg'
            img = matplotlib.image.imread(filepath)
            if len(sequence) == 0:    sequence = img.reshape(1, self.height, self.width, self.channels)
            else:                     sequence = np.append(sequence, img.reshape(1, self.height, self.width, self.channels), axis=0)
        
        if len(sequence) == self.total_frames: 
            if len(rollout_data) == 0: rollout_data = sequence.reshape(1, self.total_frames, self.height, self.width, self.channels)
            else: rollout_data = np.append(rollout_data, sequence.reshape(1, self.total_frames, self.height, self.width, self.channels), axis=0)
        
        true_inputs = rollout_data[:, :self.input_frames]
        true_outputs = rollout_data[:, self.input_frames:]

        x_data = torch.tensor(true_inputs, dtype=torch.float32).to(self.device)/255.
        y_data = torch.tensor(true_outputs, dtype=torch.float32).to(self.device)/255.
        x_data = x_data.permute(0, 1, 4, 2, 3)
        y_data = y_data.permute(0, 1, 4, 2, 3)
        
        return x_data[0], y_data[0], self.rollout_results[idx]

def load_rollout_data(INPUT_FRAMES=3, TOTAL_FRAMES=17, train_test_split=0.8, BATCH_SIZE=10, rollout_nums=1, data_dir='/', height=256, width=256, channels=3, device=None):
    OUTPUT_FRAMES = TOTAL_FRAMES - INPUT_FRAMES
    train_idx = int(train_test_split * rollout_nums)
    
    train_dataset = PhyreRolloutDataset(start=0, end=train_idx, batch_size=BATCH_SIZE, data_dir=data_dir, TOTAL_FRAMES=TOTAL_FRAMES, INPUT_FRAMES=INPUT_FRAMES, height=height, width=width, channels=channels, device=device)
    test_dataset = PhyreRolloutDataset(start=train_idx, end=rollout_nums-1, batch_size=BATCH_SIZE, data_dir=data_dir, TOTAL_FRAMES=TOTAL_FRAMES, INPUT_FRAMES=INPUT_FRAMES, height=height, width=width, channels=channels, device=device)
    
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    return train_loader, test_loader

def get_batch(data_size, dataloader, batch_time=1, seq=10):
    for i, (input_frames, output_frames, rollout_results) in enumerate(dataloader): yield (input_frames, i, output_frames)
