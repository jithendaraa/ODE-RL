import sys
import os
sys.path.append('../')

import torch
import torch.optim as optim
import torch.nn as nn
from torch.optim import lr_scheduler

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from collections import OrderedDict
from tqdm import tqdm 

from earlystopping import EarlyStopping
from encoder import Encoder
from decoder import Decoder
from encoder_decoder import ED
from ConvGRUCell import ConvGRU
from helper import get_batch

TOTAL_FRAMES = 17
height = 256
width = 256
channels = 3

EPOCHS          = 5
INPUT_FRAMES    = 3
OUTPUT_FRAMES   = TOTAL_FRAMES - INPUT_FRAMES
LR              = 1e-3
BATCH_SIZE      = 6
LAMBDA_DIFF     = 1.0
LAMBDA_IMG      = 0.003
LAMBDA_SEQ      = 0.003

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class PhyreRolloutDataset(torch.utils.data.Dataset):
    def __init__(self, rollout_data, rollout_results):
        self.rollout_data = rollout_data
        self.rollout_results = rollout_results
        true_inputs = rollout_data[:, :INPUT_FRAMES]
        true_outputs = rollout_data[:, INPUT_FRAMES:]
        self.x_data = torch.tensor(true_inputs, dtype=torch.float32).to(device)
        self.y_data = torch.tensor(true_outputs, dtype=torch.float32).to(device)
    
    def __len__(self):
        return len(self.x_data)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):    idx = idx.tolist()
        return self.x_data[idx], self.y_data[idx], self.rollout_results[idx]


def load_rollout_data(INPUT_FRAMES=3, TOTAL_FRAMES=17, train_test_split=0.6):
    OUTPUT_FRAMES = TOTAL_FRAMES - INPUT_FRAMES
    data_dir = 'rollout_data'
    rollout_nums = len(os.listdir(data_dir))
    rollout_data = np.array([])
    train_idx = int(train_test_split * rollout_nums)

    for num in range(1, rollout_nums+1):
        rollout_folder = data_dir + '/rollout_' + str(num)
        sequence = np.array([])
        for i in range(1, TOTAL_FRAMES+1):
            filepath = rollout_folder + '/frame_' + str(i) + '.jpg'
            img = matplotlib.image.imread(filepath)
            if len(sequence) == 0:    sequence = img.reshape(1, height, width, channels)
            else:                     sequence = np.append(sequence, img.reshape(1, height, width, channels), axis=0)
            
        if len(sequence) == TOTAL_FRAMES: 
            if len(rollout_data) == 0: rollout_data = sequence.reshape(1, TOTAL_FRAMES, height, width, channels)
            else: rollout_data = np.append(rollout_data, sequence.reshape(1, TOTAL_FRAMES, height, width, channels), axis=0)

    rollout_results = np.load('rollout_results.npy')
    train_rollout_data, train_rollout_results = rollout_data[:train_idx], rollout_results[:train_idx]
    test_rollout_data, test_rollout_results = rollout_data[train_idx:], rollout_results[train_idx:]

    train_dataset = PhyreRolloutDataset(train_rollout_data, train_rollout_results)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

    test_dataset = PhyreRolloutDataset(test_rollout_data, test_rollout_results)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True)

    return train_loader, test_loader


train_loader, test_loader = load_rollout_data()
predict_timesteps = np.arange(INPUT_FRAMES+1., TOTAL_FRAMES+1., 1.0)
early_stopping = EarlyStopping(patience=20, verbose=True)

# # Encoder params

encoder_ode_specs = [
    torch.nn.Conv2d(128, 128, 3, 1, 1),
    torch.nn.Conv2d(128, 128, 3, 1, 1),
    torch.nn.Conv2d(128, 128, 3, 1, 1),
    torch.nn.Conv2d(128, 128, 3, 1, 1)
]

encoder_params = [
    # Conv Encoder E
    [
        OrderedDict({'conv1_batchnorm(32)_relu_1': [channels, 32, 4, 2, 1]}),
        OrderedDict({'conv2_batchnorm(64)_relu_2': [32, 64, 3, 1, 1]}),
        OrderedDict({'conv3_batchnorm(128)_relu_3': [64, 128, 4, 2, 1]}),
    ],
    # ConvGRU cells
    [
        ConvGRU(shape=(int(height/4), int(width/4)), input_channels=128, filter_size=3, num_features=128, ode_specs=encoder_ode_specs, feed='encoder')
    ]
]

decoder_ode_specs = [
    torch.nn.Conv2d(128, 128, 3, 1, 1),
    torch.nn.Conv2d(128, 128, 3, 1, 1),
    torch.nn.Conv2d(128, 128, 3, 1, 1),
    torch.nn.Conv2d(128, 128, 3, 1, 1)
]

decoder_params = [
    # Conv Decoder G: CNN's Deconvs
    [
        OrderedDict({'deconv1_upsample_batchnorm(64)_relu_1': [128, 64, 3, 1, 1]}),
        OrderedDict({'deconv2_upsample_relu_2': [64, channels, 3, 1, 1]}),
    ], []
]

encoder = Encoder(encoder_params[0], encoder_params[1]).cuda()
decoder = Decoder(decoder_params[0], decoder_params[1], decoder_ode_specs, predict_timesteps, decoder_ode_specs)
net = ED(encoder, decoder)

# If we have multiple GPUs
if torch.cuda.device_count() > 1:   net = nn.DataParallel(net)
net.to(device)

cur_epoch = 0

lossfunction = nn.MSELoss().cuda()
optimizer = optim.Adamax(net.parameters(), lr=LR)
pla_lr_scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.99, patience=4, verbose=True)

# to track the training loss, validation loss, and their averages as the model trains
train_losses = []
valid_losses = []
avg_train_losses = []
avg_valid_losses = []
train_data_length = train_loader.__len__()

for range in range(cur_epoch, cur_epoch + EPOCHS):
    losses = []
    for (inputs, i, labels) in get_batch(train_data_length, BATCH_SIZE, train_loader, seq=3):
        if i >= 20:
            break
        # inputs and labels -> B, S, H, W, C to B, S, C, H, W
        inputs = inputs.to(device).transpose(2, 4).transpose(3, 4)
        labels = labels.to(device).transpose(2, 4).transpose(3, 4)
        optimizer.zero_grad()
        net.train()
        pred = net(inputs)
        # .transpose(0, 1)   # S,B,C,H,W
        # loss = lossfunction(pred, labels)
        # loss_aver = loss.item() / BATCH_SIZE
        # losses.append(loss.item())
        # train_losses.append(loss_aver)
        # loss.backward()
        # torch.nn.utils.clip_grad_value_(net.parameters(), clip_value=10.0)
        # optimizer.step()
    break

