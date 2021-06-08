import sys
sys.path.append('../')

import numpy as np
from collections import OrderedDict
import os
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import logging
import matplotlib.pyplot as plt

import torch
import torch.optim as optim
import torch.nn as nn
from torch.optim import lr_scheduler
from encoder_decoder import Encoder, Decoder, ED

from generate_moving_mnist import MovingMNIST
from earlystopping import EarlyStopping
from ConvRNN import ConvGRU

EPOCHS          = 3 
INPUT_FRAMES    = 12
OUTPUT_FRAMES   = 8
LR              = 1e-4
BATCH_SIZE      = 4
TIMESTAMP       = "2020-03-09T00-00-00"

save_dir = './save_model/' + TIMESTAMP
run_dir = './runs/' + TIMESTAMP
if not os.path.isdir(run_dir):
    os.makedirs(run_dir)

tb = SummaryWriter(run_dir)
early_stopping = EarlyStopping(patience=20, verbose=True)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

trainFolder = MovingMNIST(is_train=True, root='../data/', n_frames_input=INPUT_FRAMES, n_frames_output=OUTPUT_FRAMES, num_objects=[3])
validFolder = MovingMNIST(is_train=False, root='../data/', n_frames_input=INPUT_FRAMES, n_frames_output=OUTPUT_FRAMES, num_objects=[3])

trainLoader = torch.utils.data.DataLoader(trainFolder, batch_size=BATCH_SIZE, shuffle=False)
validLoader = torch.utils.data.DataLoader(validFolder, batch_size=BATCH_SIZE, shuffle=False)
print(trainLoader.__len__(), validLoader.__len__())

encoder_params = [
    [
        OrderedDict({'conv1_leaky_1': [1, 16, 3, 1, 1]}),
        OrderedDict({'conv2_leaky_1': [64, 64, 3, 2, 1]}),
        OrderedDict({'conv3_leaky_1': [96, 96, 3, 2, 1]}),
    ],

    [
        ConvGRU(shape=(64,64), input_channels=16, filter_size=5, num_features=64),
        ConvGRU(shape=(32,32), input_channels=64, filter_size=5, num_features=96),
        ConvGRU(shape=(16,16), input_channels=96, filter_size=5, num_features=96)
    ]
]

decoder_params = [
    [
        OrderedDict({'deconv1_leaky_1': [96, 96, 4, 2, 1]}),
        OrderedDict({'deconv2_leaky_1': [96, 96, 4, 2, 1]}),
        OrderedDict({
            'conv3_leaky_1': [64, 16, 3, 1, 1],
            'conv4_leaky_1': [16, 1, 1, 1, 0]
        }),
    ],

    [
        ConvGRU(shape=(16,16), input_channels=96, filter_size=5, num_features=96),
        ConvGRU(shape=(32,32), input_channels=96, filter_size=5, num_features=96),
        ConvGRU(shape=(64,64), input_channels=96, filter_size=5, num_features=64),
    ]
]

encoder = Encoder(encoder_params[0], encoder_params[1]).cuda()
decoder = Decoder(decoder_params[0], decoder_params[1], OUTPUT_FRAMES).cuda()
net = ED(encoder, decoder)
print("ED Done")

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

lossfunction = nn.MSELoss().cuda()
optimizer = optim.Adam(net.parameters(), lr=LR)
pla_lr_scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=4, verbose=True)

# to track the training loss, validation loss, and their averages as the model trains
train_losses = []
valid_losses = []
avg_train_losses = []
avg_valid_losses = []

for epoch in range(cur_epoch, EPOCHS + 1):
    ###################
    # train the model #
    ###################
    t = tqdm(trainLoader, leave=False, total=len(trainLoader))
    for i, (idx, targetVar, inputVar, _, _) in enumerate(t):
        
        # inputVar  --> batch_size x input_frames x 1 x 64 x 64; input first input_frames frames
        # targetVar --> batch_size x output_frames x 1 x 64 x 64; output first input_frames frames
        inputs = inputVar.to(device)  # B,S,C,H,W
        label = targetVar.to(device)  # B,S,C,H,W
        print(inputs.size())
        
        optimizer.zero_grad()
        net.train()
        pred = net(inputs)  # B,S,C,H,W
        loss = lossfunction(pred, label)
        loss_aver = loss.item() / BATCH_SIZE
        train_losses.append(loss_aver)
        loss.backward()
        torch.nn.utils.clip_grad_value_(net.parameters(), clip_value=10.0)
        optimizer.step()
        t.set_postfix({
            'trainloss': '{:.6f}'.format(loss_aver),
            'epoch': '{:02d}'.format(epoch)
        })
        tb.add_scalar('TrainLoss', loss_aver, epoch)
        
        ######################
        # validate the model #
        ######################
        with torch.no_grad():
            net.eval()
            t = tqdm(validLoader, leave=False, total=len(validLoader))
            for i, (idx, targetVar, inputVar, _, _) in enumerate(t):
                if i == 3000:
                    break
                inputs = inputVar.to(device)
                label = targetVar.to(device)
                pred = net(inputs)
                loss = lossfunction(pred, label)
                loss_aver = loss.item() / BATCH_SIZE
                # record validation loss
                valid_losses.append(loss_aver)
                print ("validloss: {:.6f},  epoch : {:02d}".format(loss_aver,epoch),end = '\r', flush=True)
        
                t.set_postfix({
                'validloss': '{:.6f}'.format(loss_aver),
                'epoch': '{:02d}'.format(epoch)
                })

                tb.add_scalar('ValidLoss', loss_aver, epoch)
                torch.cuda.empty_cache()
                # print training/validation statistics
                # calculate average loss over an epoch
                train_loss = np.average(train_losses)
                valid_loss = np.average(valid_losses)
                avg_train_losses.append(train_loss)
                avg_valid_losses.append(valid_loss)

                epoch_len = len(str(EPOCHS))

                print_msg = (f'[{epoch:>{epoch_len}}/{EPOCHS:>{epoch_len}}] ' +
                                 f'train_loss: {train_loss:.6f} ' +
                                 f'valid_loss: {valid_loss:.6f}')

                print(print_msg)
                # clear lists to track next epoch
                train_losses = []
                valid_losses = []
                pla_lr_scheduler.step(valid_loss)  # lr_scheduler
                model_dict = {
                    'epoch': epoch,
                    'state_dict': net.state_dict(),
                    'optimizer': optimizer.state_dict()
                }
                early_stopping(valid_loss.item(), model_dict, epoch, save_dir)
                if early_stopping.early_stop:
                    print("Early stopping")
                    break
                
                with open("avg_train_losses.txt", 'wt') as f:
                    for i in avg_train_losses:
                        print(i, file=f)

                with open("avg_valid_losses.txt", 'wt') as f:
                    for i in avg_valid_losses:
                        print(i, file=f)
        break
    break