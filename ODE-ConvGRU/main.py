import sys
import os
sys.path.append('../')

import numpy as np
from collections import OrderedDict
from tensorboardX import SummaryWriter
from tqdm import tqdm
import logging
import matplotlib.pyplot as plt

import torch
import torch.optim as optim
import torch.nn as nn
from torch.optim import lr_scheduler
from torchdiffeq import odeint
from conv_encoder import Encoder
from encoder_decoder import ED

from generate_moving_mnist import MovingMNIST
from earlystopping import EarlyStopping
from ConvGRUCell import ConvGRU
from helper import get_batch

EPOCHS          = 3     # FIXME: 500 epochs in the paper
INPUT_FRAMES    = 10
OUTPUT_FRAMES   = 10
LR              = 1e-3
BATCH_SIZE      = 4     # FIXME: Batch size 8 in the paper
TIMESTAMP       = "2020-03-09T00-00-00"
LAMBDA_DIFF     = 1.0
LAMBDA_IMG      = 0.003
LAMBDA_SEQ      = 0.003

# For MNIST dataset
h = 64
w = 64
c = 1

save_dir = './save_model/' + TIMESTAMP
run_dir = './runs/' + TIMESTAMP
if not os.path.isdir(run_dir):
    os.makedirs(run_dir)

tb = SummaryWriter(run_dir)
early_stopping = EarlyStopping(patience=20, verbose=True)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

trainFolder = MovingMNIST(is_train=True, root='../data/', n_frames_input=INPUT_FRAMES, n_frames_output=OUTPUT_FRAMES, num_objects=[3])
validFolder = MovingMNIST(is_train=False, root='../data/', n_frames_input=INPUT_FRAMES, n_frames_output=OUTPUT_FRAMES, num_objects=[3])

train_data_length = trainFolder.__len__()
valid_data_length = validFolder.__len__()
print("Train data length: ", train_data_length)
print("Valid data length: ", valid_data_length)

trainLoader = torch.utils.data.DataLoader(trainFolder, batch_size=BATCH_SIZE, shuffle=False)
validLoader = torch.utils.data.DataLoader(validFolder, batch_size=BATCH_SIZE, shuffle=False)

encoder_ode_specs = [
    torch.nn.Conv2d(64, 64, 3, 1, 1),
    torch.nn.Conv2d(64, 64, 3, 1, 1),
    torch.nn.Conv2d(64, 64, 3, 1, 1),
    torch.nn.Conv2d(64, 64, 3, 1, 1)
]

encoder_params = [
    # Conv Encoder E
    [
        OrderedDict({'conv1_downsample?64,64|_relu_1': [c, 32, 3, 1, 1]}),
        OrderedDict({'conv2_relu_2': [32, 64, 3, 1, 1]}),
        OrderedDict({'conv3_relu_3': [64, 128, 4, 2, 1]}),
    ],
    # ODE-ConvGRU
    [
        ConvGRU(shape=(int(h/4), int(w/4)), input_channels=128, filter_size=3, num_features=64, ode_specs=encoder_ode_specs)
    ]
]

decoder_ode_specs = [
    torch.nn.Conv2d(64, 64, 3, 1, 1),
    torch.nn.Conv2d(64, 64, 3, 1, 1),
    torch.nn.Conv2d(64, 64, 3, 1, 1),
    torch.nn.Conv2d(64, 64, 3, 1, 1)
]

decoder_params = [
    # Conv Decoder G
    [
        OrderedDict({'conv1_downsample?64,64|_relu_1': [c, 32, 3, 1, 1]}),
        OrderedDict({'conv2_relu_2': [32, 64, 3, 1, 1]}),
        OrderedDict({'conv3_relu_3': [64, 128, 4, 2, 1]}),
    ],
    # ODE-ConvGRU
    [
        ConvGRU(shape=(int(h/4), int(w/4)), input_channels=128, filter_size=3, num_features=64)
    ]
]

encoder = Encoder(encoder_params[0], encoder_params[1]).cuda()
decoder = None
net = ED(encoder, decoder)

# If we have multiple GPUs
if torch.cuda.device_count() > 1:   net = nn.DataParallel(net)
net.to(device)

if os.path.exists(os.path.join(save_dir, 'checkpoint.pth.tar')):
    # load existing model
    print('==> loading existing model')
    model_info = torch.load(os.path.join(save_dir, 'checkpoin.pth.tar'))
    net.load_state_dict(model_info['state_dict'])
    optimizer = torch.optim.Adam(net.parameters())
    optimizer.load_state_dict(model_info['optimizer'])
    cur_epoch = model_info['epoch'] + 1
else:
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    cur_epoch = 0

# FIXME Loss: As given in paper; Adamax optimizer; exponential LR deacy -> 0.99 per epoch
lossfunction = nn.MSELoss().cuda()
optimizer = optim.Adamax(net.parameters(), lr=LR)
pla_lr_scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=4, verbose=True)

# to track the training loss, validation loss, and their averages as the model trains
train_losses = []
valid_losses = []
avg_train_losses = []
avg_valid_losses = []

t = tqdm(trainLoader, leave=False, total=len(trainLoader))

for (inputs, i, labels) in get_batch(train_data_length, BATCH_SIZE, trainLoader, seq=10):
    inputs = inputs.to(device)
    labels = labels.to(device)
    print(inputs.size(), i, labels.size())
    optimizer.zero_grad()
    net.train()
    pred = net(inputs)  # B,S,C,H,W
    print(type(pred), len(pred))

    break
#         # print(type(pred))

    #     break
    # break
