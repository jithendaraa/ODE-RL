import sys
import os
sys.path.append('../')

import numpy as np
from collections import OrderedDict
from tensorboardX import SummaryWriter
from tqdm import tqdm
import logging
import matplotlib.pyplot as plt
import pickle
import argparse

import torch
import torch.optim as optim
import torch.nn as nn
from torch.optim import lr_scheduler
from torchvision.utils import make_grid, save_image
from conv_encoder import Encoder
from conv_decoder import Decoder
from encoder_decoder import ED

from generate_moving_mnist import MovingMNIST
from earlystopping import EarlyStopping
from ConvGRUCell import ConvGRU
from helper import get_batch, plot_images
from utils import write_video

os.system('./launch.sh')

def arg_parser():
    # Hyperparameters
    parser = argparse.ArgumentParser()
    parser.add_argument('--algo', type=str, default='ode_convgru', help='ode_convgru')
    parser.add_argument('--id', type=str, default='default', help='Experiment ID')
    parser.add_argument('--seed', type=int, default=1, metavar='S', help='Random seed')
    parser.add_argument('--disable-cuda', action='store_true', help='Disable CUDA')
    parser.add_argument('--epochs', default=2000, type=int)
    parser.add_argument('--input-frames', default=10, type=int)
    parser.add_argument('--output-frames', default=10, type=int)  
    parser.add_argument('--learning-rate', default=3e-4, type=float)  
    parser.add_argument('--batch-size', default=10, type=int) 
    parser.add_argument('--timestamp', default="2020-03-09T00-00-00")
    parser.add_argument('-d', '--dataset', default="MNIST")  

    parser.add_argument('--lambda-diff', default=1.0, type=float)
    parser.add_argument('--lambda-img', default=3e-3, type=float)
    parser.add_argument('--lambda-seq', default=3e-3, type=float)
    
    args = parser.parse_args()
    return args


# Setup
args = arg_parser()

if args.dataset == 'MNIST':
    h, w, c = 64, 64, 1
    predict_timesteps = [11., 12., 13., 14., 15., 16., 17., 18., 19., 20.]
elif args.dataset == 'phyre':
    orig_h, orig_w, c = 256, 256, 3
    h, w = 64, 64
    predict_timesteps = np.arange(2.0, 18.0)

save_dir = './save_model/' + args.timestamp
run_dir = './runs/' + args.timestamp
if not os.path.isdir(run_dir):
    os.makedirs(run_dir)

tb = SummaryWriter(run_dir)
early_stopping = EarlyStopping(patience=20, verbose=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

trainFolder = MovingMNIST(is_train=True, root='../data/', n_frames_input=args.input_frames, n_frames_output=args.output_frames, num_objects=[3])
validFolder = MovingMNIST(is_train=False, root='../data/', n_frames_input=args.input_frames, n_frames_output=args.output_frames, num_objects=[3])

train_data_length = trainFolder.__len__()
valid_data_length = validFolder.__len__()
print("Train data length: ", train_data_length)
print("Valid data length: ", valid_data_length)

trainLoader = torch.utils.data.DataLoader(trainFolder, batch_size=args.batch_size, shuffle=False)
validLoader = torch.utils.data.DataLoader(validFolder, batch_size=args.batch_size, shuffle=False)

encoder_ode_specs = [
    torch.nn.Conv2d(64, 64, 3, 1, 1),
    torch.nn.Conv2d(64, 64, 3, 1, 1),
    torch.nn.Conv2d(64, 64, 3, 1, 1),
    torch.nn.Conv2d(64, 64, 3, 1, 1)
]

encoder_params = [
    # Conv Encoder E
    [
        OrderedDict({'conv1_downsample?64,64|_batchnorm(32)_relu_1': [c, 32, 3, 1, 1]}),
        OrderedDict({'conv2_batchnorm(64)_relu_1': [32, 64, 3, 1, 1]}),
        OrderedDict({'conv3_batchnorm(128)_relu_1': [64, 128, 4, 2, 1]}),
    ],
    # ConvGRU cells
    [
        ConvGRU(shape=(int(h/4), int(w/4)), input_channels=128, filter_size=3, num_features=64, ode_specs=encoder_ode_specs, feed='encoder')
    ]
]

decoder_ode_specs = [
    torch.nn.Conv2d(64, 64, 3, 1, 1),
    torch.nn.Conv2d(64, 64, 3, 1, 1),
    torch.nn.Conv2d(64, 64, 3, 1, 1),
    torch.nn.Conv2d(64, 64, 3, 1, 1)
]

decoder_params = [
    # Conv Decoder G: CNN's
    [
        OrderedDict({'deconv1_upsample?16,16|batchnorm(128)_relu_1': [64, 128, 3, 1, 1]}),
        OrderedDict({'deconv2_upsample?16,16|batchnorm(64)_relu_1': [128, 64, 3, 1, 1]}),
        OrderedDict({'deconv3_relu_1': [64, c, 3, 1, 1]}),
    ],
    # ConvGRU cells
    []
]

encoder = Encoder(encoder_params[0], encoder_params[1]).cuda()
decoder = Decoder(decoder_params[0], decoder_params[1], decoder_ode_specs, predict_timesteps)
net = ED(encoder, decoder)

# If we have multiple GPUs
# if torch.cuda.device_count() > 1:   
#     print("HMM")
#     net = nn.DataParallel(net)
net.to(device)

cur_epoch = 0
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

# FIXME Loss: As given in paper; Adamax optimizer; exponential LR deacy -> 0.99 per epoch
lossfunction = nn.MSELoss().cuda()
optimizer = optim.Adamax(net.parameters(), lr=args.learning_rate)
pla_lr_scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.99, patience=4, verbose=True)

# to track the training loss, validation loss, and their averages as the model trains
train_losses = []
valid_losses = []
avg_train_losses = []
avg_valid_losses = []

t = tqdm(trainLoader, leave=False, total=len(trainLoader))

for epoch in range(cur_epoch, cur_epoch + args.epochs):
    losses = []
    train_losses = []

    for (inputs, i, labels) in get_batch(train_data_length, args.batch_size, trainLoader, seq=10):
        inputs = inputs.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        net.train()
        pred = net(inputs).transpose(0, 1)   # Convert to S,B,C,H,W
        loss = lossfunction(pred, labels)
        loss_aver = loss.item() / args.batch_size
        losses.append(loss.item())
        train_losses.append(loss_aver)
        loss.backward()
        torch.nn.utils.clip_grad_value_(net.parameters(), clip_value=10.0)
        optimizer.step()

    video_frames = []
    for frame_num in range(args.output_frames):
        true_output_frame = labels.permute(1,0,2,3,4)[frame_num]
        pred_output_frame = pred.permute(1,0,2,3,4)[frame_num]
        video_frames.append(make_grid(torch.cat([true_output_frame.cpu(), pred_output_frame.cpu()], dim=3), nrow=5).detach().numpy())  # Decentre
    write_video(video_frames, 'test_episode_' + str(epoch), 'results') 

    with torch.no_grad():
        net.eval()
        for (inputs_, i_, labels_) in get_batch(valid_data_length, args.batch_size, validLoader, seq=10):
            inputs_ = inputs_.to(device)
            labels_ = labels_.to(device)
            pred_ = net(inputs_).transpose(0, 1)
            loss = lossfunction(pred_, labels_)
            loss_aver = loss.item() / args.batch_size
            valid_losses.append(loss_aver)
            torch.cuda.empty_cache()
            train_loss = np.average(train_losses)
            valid_loss = np.average(valid_losses)
            avg_train_losses.append(train_loss)
            avg_valid_losses.append(valid_loss)
            epoch_len = len(str(args.epochs))
            print_msg = (f'[{epoch:>{epoch_len}}/{args.epochs:>{epoch_len}}] ' +
                             f'train_loss: {train_loss:.6f} ' +
                             f'valid_loss: {valid_loss:.6f}')
            print(print_msg)
            valid_losses = []
            pla_lr_scheduler.step(valid_loss)  # lr_scheduler
            model_dict = {
                'epoch': epoch,
                'state_dict': net.state_dict(),
                'optimizer': optimizer.state_dict()
            }
            with open('params.pickle', 'wb') as handle:
                pickle.dump(model_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
            early_stopping(valid_loss.item(), model_dict, epoch, save_dir)
            if early_stopping.early_stop:
                # print("Early stopping")
                break
    
    print("End of epoch", epoch)