import sys
import os
sys.path.append('../')
from twilio.rest import Client

import torch
import torch.optim as optim
import torch.nn as nn
from torch.optim import lr_scheduler
import torchvision
from torchvision.utils import make_grid, save_image

import numpy as np
import cv2
import matplotlib
import matplotlib.pyplot as plt
from PIL import Image
from collections import OrderedDict
from tqdm import tqdm 
import argparse

from earlystopping import EarlyStopping
from encoder import Encoder
from decoder import Decoder
from encoder_decoder import ED
from ConvGRUCell import ConvGRU
from helper import *

def arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=int, default=os.system('./launch.sh'), help='GPU number')
    parser.add_argument('--seed', type=int, default=1, metavar='S', help='Random seed')
    parser.add_argument('--disable-cuda', action='store_true', help='Disable CUDA')
    parser.add_argument('--input-frames', default=3, type=int)
    parser.add_argument('--output-frames', default=14, type=int)
    parser.add_argument("-e", "--epochs", default=2000, type=int)
    parser.add_argument("-lr", "--learning_rate", default=1e-3, type=float)
    parser.add_argument("-bs", "--batch_size", default=4, type=int)
    parser.add_argument("-tts", "--train_test_split", default=1.0, type=float)
    parser.add_argument('--lambda-diff', default=1.0, type=float)
    parser.add_argument('--lambda-img', default=3e-3, type=float)
    parser.add_argument('--lambda-seq', default=3e-3, type=float)

    args = parser.parse_args()
    return args


def write_video(frames, title, path=''):
  frames = np.multiply(np.stack(frames, axis=0).transpose(0, 2, 3, 1), 255).clip(0, 255).astype(np.uint8)[:, :, :, ::-1]  # VideoWrite expects H x W x C in BGR
  _, H, W, _ = frames.shape
  writer = cv2.VideoWriter(os.path.join(path, '%s.mp4' % title), cv2.VideoWriter_fourcc(*'mp4v'), 30., (W, H), True)
  for frame in frames:
    writer.write(frame)
  writer.release()
  print("Video saved at ", title)

# account_id = "AC689668e8b17297ec4bcfc3bbf41fa861"
# token = "bd7db066f3f0a3e18f2f00cf77dc8893"
# client = Client(account_id, token)
# from_whatsapp_number = 'whatsapp:+14155238886'
# to_whatsapp_number = 'whatsapp:+918754497524'

# try:
args = arg_parser()
TOTAL_FRAMES = args.input_frames + args.output_frames
orig_h, orig_w, c = 256, 256, 3
h, w = 64, 64

if torch.cuda.is_available() and not args.disable_cuda:
    device = torch.device('cuda')
    torch.cuda.manual_seed(args.seed)
else:
    device = torch.device('cpu')

print("Device:", device)
data_dir = 'rollout_data'
rollout_nums = len(os.listdir(data_dir))
# client.messages.create(body='Started to run _phyre main.py_ on GPU ' + str(args.gpu), from_=from_whatsapp_number, to=to_whatsapp_number)

train_loader, test_loader = load_rollout_data(args.input_frames, TOTAL_FRAMES, args.train_test_split, args.batch_size, rollout_nums, data_dir, h, w, c, device)
print('loaded data!')
predict_timesteps = np.arange(args.input_frames+1., TOTAL_FRAMES+1., 1.0)
early_stopping = EarlyStopping(patience=20, verbose=True)
# Encoder params
encoder_ode_specs = [
    torch.nn.Conv2d(128, 128, 3, 1, 1),
    torch.nn.Conv2d(128, 128, 3, 1, 1),
    torch.nn.Conv2d(128, 128, 3, 1, 1),
    torch.nn.Conv2d(128, 128, 3, 1, 1)
]
encoder_params = [
    [   # Conv Encoder E
        OrderedDict({'conv1_batchnorm(32)_relu_1': [c, 32, 4, 2, 1]}),
        OrderedDict({'conv2_batchnorm(64)_relu_2': [32, 64, 3, 1, 1]}),
        OrderedDict({'conv3_batchnorm(128)_relu_3': [64, 128, 4, 2, 1]}),
    ],
    [   # ConvGRU cells
        ConvGRU(shape=(int(h/4), int(w/4)), input_channels=128, filter_size=3, num_features=128, ode_specs=encoder_ode_specs, feed='encoder', device=device)
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
        OrderedDict({'deconv2_upsample_relu_2': [64, c, 3, 1, 1]}),
    ], []
]

encoder = Encoder(encoder_params[0], encoder_params[1]).to(device)
decoder = Decoder(decoder_params[0], decoder_params[1], decoder_ode_specs, predict_timesteps, decoder_ode_specs, device=device).to(device)
net = ED(encoder, decoder)

# If we have multiple GPUs
# if torch.cuda.device_count() > 1:   net = nn.DataParallel(net)
net.to(device)
# net = torch.load('entire_model.pt', map_location=device)
# net.load_state_dict(torch.load("entire_model.pt", map_location=device))
net.eval()

cur_epoch = 0
lossfunction = nn.MSELoss()
optimizer = optim.Adamax(net.parameters(), lr=args.learning_rate)
pla_lr_scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.99, patience=4, verbose=True)
# to track the training loss, validation loss, and their averages as the model trains
train_losses = []
valid_losses = []
avg_train_losses = []
avg_valid_losses = []
train_data_length = int(args.train_test_split * rollout_nums)
test_data_length = rollout_nums - train_data_length
pred, inputs, labels, ground_truth = None, None, None, None
net.train()
print('training starts..')

for epoch in tqdm(range(cur_epoch, cur_epoch + args.epochs)):
    losses = []
    for i, (inputs, labels, _) in tqdm(enumerate(train_loader), total=train_data_length):
        # inputs and labels -> B, S, H, W, C
        # inp_image = torchvision.utils.make_grid(inputs[0], nrow=3)
        # out_image = torchvision.utils.make_grid(labels[0], nrow=14)
        optimizer.zero_grad()
        pred = net(inputs).transpose(0, 1)   # B, S, C, H, W
        loss = lossfunction(pred, labels)
        losses.append(loss.item())
        loss_aver = losses[-1]
        train_losses.append(loss_aver)
        loss.backward()
        torch.nn.utils.clip_grad_value_(net.parameters(), clip_value=10.0)
        optimizer.step()

    video_frames = []
    for frame_num in range(args.output_frames):
        true_output_frame = labels.permute(1,0,2,3,4)[frame_num].cpu()
        pred_output_frame = pred.permute(1,0,2,3,4)[frame_num].cpu()
        video_frames.append(make_grid(torch.cat([true_output_frame, pred_output_frame], dim=3), nrow=5).detach().numpy())  # Decentre
    write_video(video_frames, 'test_episode_' + str(epoch), 'phyre_results') 
    print("Epoch", epoch, "/", args.epochs, "Loss: ", losses[-1])
    torch.save(net, 'entire_model.pt')

    # if epoch % 10 == 0:
    #     # client.messages.create(body='_phyre main.py_ Epoch ' + str(epoch) + '/'+ str(cur_epoch+EPOCHS) +'\nLoss:' + str(loss.item()), from_=from_whatsapp_number, to=to_whatsapp_number)
    #     for j, (inputs, labels, _) in enumerate(test_loader):
    #         pred = net(inputs).transpose(0, 1)
    #         inp_image = torchvision.utils.make_grid(inputs[0], nrow=3)
    #         out_image = torchvision.utils.make_grid(labels[0], nrow=14)
    #         pred_image = torchvision.utils.make_grid(pred[0], nrow=14)
    #         filename = str(j) + '_epoch_' + str(epoch) + '.jpg'
    #         torchvision.utils.save_image(inp_image, 'input_ground_truth/inp_sample'+filename)
    #         torchvision.utils.save_image(out_image, 'ground_truth/out_sample'+filename)
    #         torchvision.utils.save_image(pred_image, 'predicted_image/pred_sample'+filename)
    #         break

print('training ends')
        
            
# client.messages.create(body='Training ended _phyre main.py_ on GPU ' + str(args.gpu), from_=from_whatsapp_number, to=to_whatsapp_number)

# except:
#     print("Error")
#     print(sys.exc_info())
    # client.messages.create(body='Error in _phyre main.py_ \n'+ str(sys.exc_info()), from_=from_whatsapp_number, to=to_whatsapp_number)
