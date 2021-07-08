#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   main.py
@Time    :   2020/03/09
@Author  :   jhhuang96
@Mail    :   hjh096@126.com
@Version :   1.0
@Description:   
'''

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from encoder import Encoder
from decoder import Decoder
from model import ED
from net_params import convlstm_encoder_params, convlstm_decoder_params, convgru_encoder_params, convgru_decoder_params
from data.mm import MovingMNIST
import torch
from torch import nn
from torch.optim import lr_scheduler
import torch.optim as optim
import sys
from earlystopping import EarlyStopping
from tqdm import tqdm
import numpy as np
from tensorboardX import SummaryWriter
import argparse
import wandb

TIMESTAMP = "2020-03-09T00-00-00"
parser = argparse.ArgumentParser()
parser.add_argument('-clstm',
                    '--convlstm',
                    help='use convlstm as base cell',
                    action='store_true')
parser.add_argument('-cgru',
                    '--convgru',
                    help='use convgru as base cell',
                    action='store_true')
parser.add_argument('--batch_size',
                    default=4,
                    type=int,
                    help='mini-batch size')
parser.add_argument('-lr', default=1e-4, type=float, help='G learning rate')
parser.add_argument('-frames_input',
                    default=10,
                    type=int,
                    help='sum of input frames')
parser.add_argument('-frames_output',
                    default=10,
                    type=int,
                    help='sum of predict frames')
parser.add_argument('-epochs', default=1000, type=int, help='sum of epochs')
args = parser.parse_args()

# random_seed = 1996
# np.random.seed(random_seed)
# torch.manual_seed(random_seed)
# if torch.cuda.device_count() > 1:
#     torch.cuda.manual_seed_all(random_seed)
# else:
#     torch.cuda.manual_seed(random_seed)
# torch.backends.cudnn.deterministic = True
# torch.backends.cudnn.benchmark = False

save_dir = './save_model/' + TIMESTAMP

trainFolder = MovingMNIST(is_train=True,
                          root='data/',
                          n_frames_input=args.frames_input,
                          n_frames_output=args.frames_output,
                          num_objects=[3])
validFolder = MovingMNIST(is_train=False,
                          root='data/',
                          n_frames_input=args.frames_input,
                          n_frames_output=args.frames_output,
                          num_objects=[3])
trainLoader = torch.utils.data.DataLoader(trainFolder,
                                          batch_size=args.batch_size,
                                          shuffle=False)
validLoader = torch.utils.data.DataLoader(validFolder,
                                          batch_size=args.batch_size,
                                          shuffle=False)

if args.convlstm:
    encoder_params = convlstm_encoder_params
    decoder_params = convlstm_decoder_params
if args.convgru:
    encoder_params = convgru_encoder_params
    decoder_params = convgru_decoder_params
else:
    encoder_params = convgru_encoder_params
    decoder_params = convgru_decoder_params


def train():
    '''
    main function to run the training
    '''
    encoder = Encoder(encoder_params[0], encoder_params[1]).cuda()
    decoder = Decoder(decoder_params[0], decoder_params[1]).cuda()
    net = ED(encoder, decoder)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net.to(device)

    lossfunction = nn.MSELoss().cuda()
    optimizer = optim.Adamax(net.parameters(), lr=args.lr)

    print("START")
    wandb.init(project='ODE-RL', entity='jithendaraa', config=vars(args))
    config = wandb.config
    wandb.watch(net)
    step = 0

    for epoch in range(0, args.epochs + 1):
        t = tqdm(trainLoader, leave=False, total=len(trainLoader))
        for i, (idx, targetVar, inputVar, _, _) in enumerate(t):
            inputs = inputVar.to(device)  # B,S,C,H,W range [0, 1]
            label = targetVar.to(device)  # B,S,C,H,W range [0, 1]
            b, t, c, h, w = inputs.size()

            pred = net(inputs)  # B,S,C,H,W range [0, 1]
            
            loss = lossfunction(pred.view(b*t, c, h, w), label.view(b*t, c, h, w))
            pred_gt = torch.cat((pred.detach().cpu() * 255.0, label.cpu() * 255.0), dim=0)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if step % 100 == 0:
                wandb.log({"Train loss": loss.item() / args.batch_size, "Pred_GT": wandb.Video(pred_gt)})
            step += 1


if __name__ == "__main__":
    train()
