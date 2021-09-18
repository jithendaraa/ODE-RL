import sys
sys.path.append('../')

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from modules.DiffEqSolver import DiffEqSolver, ODEFunc
from modules.ConvGRUCell import ConvGRUCell
from modules.ODEConvGRUCell import ODEConvGRUCell
from modules.RIM_GRU import RIM_GRU
from modules.RIM_CGRU import RIM_CGRU
from helpers.utils import *

class Encoder(nn.Module):
    def __init__(self, in_ch, encoder_type='default'):
        super(Encoder, self).__init__()

        if encoder_type == 'default':
            self.resize = 64
            self.layers = nn.Sequential(
                nn.Conv2d(in_ch, 64, 4, 2, 1), nn.BatchNorm2d(64), nn.LeakyReLU(0.2),
                nn.Conv2d(64, 128, 4, 2, 1), nn.BatchNorm2d(128), nn.LeakyReLU(0.2),
                nn.Conv2d(128, 256, 4, 2, 1), nn.BatchNorm2d(256), nn.LeakyReLU(0.2),
                nn.Conv2d(256, 512, 4, 2, 1), nn.BatchNorm2d(512), nn.LeakyReLU(0.2),
                nn.Conv2d(512, 128, 4, 1, 0), nn.BatchNorm2d(128), nn.Tanh())

        elif encoder_type in ['odecgru', 'cgru']:
            self.resize = 16
            self.layers = nn.Sequential(
                nn.Conv2d(in_ch, 16, 4, 2, 1), nn.BatchNorm2d(16), nn.LeakyReLU(0.2),
                nn.Conv2d(16, 32, 4, 2, 1), nn.BatchNorm2d(32), nn.LeakyReLU(0.2),
                nn.Conv2d(32, 64, 4, 2, 1), nn.BatchNorm2d(64), nn.LeakyReLU(0.2),
                nn.Conv2d(64, 128, 4, 2, 1), nn.BatchNorm2d(128), nn.Tanh())
        
        elif encoder_type in ['cgru_sa']:
            self.resize = 8
            self.layers = nn.Sequential(
                nn.Conv2d(in_ch, 16, 3, 2, 1), nn.BatchNorm2d(16), nn.LeakyReLU(0.2),
                nn.Conv2d(16, 32, 3, 2, 1), nn.BatchNorm2d(32), nn.LeakyReLU(0.2),
                nn.Conv2d(32, 64, 3, 1, 1), nn.BatchNorm2d(64), nn.LeakyReLU(0.2),
                nn.Conv2d(64, 128, 3, 2, 1), nn.BatchNorm2d(128), nn.Tanh())


    def forward(self, inputs):
        return self.layers(inputs)

class GRUEncoder(nn.Module):
    def __init__(self, input_size=128, hidden_size=256, z_size=256, ode=False, device=None, method='dopri5', type='static', batch_first=True, opt=None):
        super().__init__()
        self.ode = ode
        self.device = device
        self.type = type
        self.batch_first = batch_first
        self.hidden_size = hidden_size
        self.rim = opt.rim
        self.opt = opt
        self.num_rims = self.opt.n_hid[0] // self.opt.unit_per_rim

        if ode is True:
            z0_outs = hidden_size // 2
            self.z0_net = nn.GRU(input_size, z0_outs, num_layers=1, batch_first=True)
            self.ode_func_net = nn.Sequential(
                nn.Linear(2*z0_outs, hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, 2*z0_outs),
            )
            self.ode_func = ODEFunc(net=self.ode_func_net, device=device)
            self.ode_solver = DiffEqSolver(self.ode_func, method, device=device).to(device)
        
        else:
            self.gru_net = nn.GRU(input_size, hidden_size, num_layers=1, batch_first=batch_first)
            if self.type == 'dynamic':
                
                if self.rim is True:
                    if opt.encoder in ['default']:
                        self.dynamic_net = RIM_GRU(opt.emsize, [hidden_size], opt)
                    
                    else:
                        NotImplementedError('Not implemented RIMs for encoder type ' + opt.encoder)
                else:
                    self.dynamic_net = nn.GRU(hidden_size, hidden_size, num_layers=1, batch_first=batch_first)
                
                total_params = sum(p.numel() for p in self.dynamic_net.parameters() if p.requires_grad)
                print("Model Built with Total Number of Trainable Parameters: " + str(total_params))
        
        if self.rim is True:
            self.mean_net = nn.Linear(hidden_size // self.num_rims, z_size)
            self.std_net = nn.Linear(hidden_size // self.num_rims, z_size)
        else:
            self.mean_net = nn.Linear(hidden_size, z_size)
            self.std_net = nn.Linear(hidden_size, z_size)
    
    def forward(self, inputs, seq_len=10):
        timesteps_to_predict = torch.from_numpy(np.arange(seq_len, dtype=np.int64)) / seq_len

        if self.ode is True:
            outs, hidden = self.z0_net(inputs)
            hidden = hidden.squeeze(0)
            outs = self.ode_solver(hidden, timesteps_to_predict)

        elif self.ode is False:
            outs, hidden = self.gru_net(inputs)
        
        if self.type == 'static':
            hidden = hidden.squeeze(0)
            mean = self.mean_net(hidden)
            std = F.softplus(self.std_net(hidden))
        
        elif self.type == 'dynamic':
            inp_zeros = torch.zeros_like(hidden) # 1, b, f
            if self.batch_first is True: 
                inp_zeros = inp_zeros.permute(1, 0, 2)  # Make batch as first dim
            
            if self.rim is True:
                hidden = hidden.squeeze(0)
                inp = inp_zeros.repeat(1, seq_len, 1).permute(1, 0, 2) # t, b, f
                dynamic_hiddens, _ = self.dynamic_net(inp, hidden, seq_len) # t, b, opt.n_hid[0]
                t, b, f = dynamic_hiddens.size()
                dynamic_hiddens = dynamic_hiddens.view(b, t, -1, self.num_rims).permute(0, 1, 3, 2) # b, t, num_rims, unit_per_rim

                mean = self.mean_net(dynamic_hiddens)
                std = F.softplus(self.std_net(dynamic_hiddens))
                mean = mean.permute(0, 1, 3, 2).reshape(b, t, -1)
                std = std.permute(0, 1, 3, 2).reshape(b, t, -1)

            else:
                dynamic_hiddens = []
                for t in range(seq_len):
                    outs, hidden = self.dynamic_net(inp_zeros, hidden)
                    dynamic_hiddens.append(hidden.squeeze(0))

                dynamic_hiddens = torch.stack(dynamic_hiddens).to(self.device)
                t, b, f = dynamic_hiddens.size()
                mean = self.mean_net(dynamic_hiddens.view(-1, f))
                std = F.softplus(self.std_net(dynamic_hiddens.view(-1, f)))
                mean = mean.view(t, b, -1).permute(1, 0, 2)
                std = std.view(t, b, -1).permute(1, 0, 2)
            
        else: # for 'prior' self.type GRUEncoder
            mean = self.mean_net(outs)
            std = F.softplus(self.std_net(outs))

        return mean, std 

class LSTMEncoder(nn.Module):
    def __init__(self, input_size=128, hidden_size=256, z_size=256, static=False, ode=False, device=None, method='dopri5'):
        super().__init__()
        self.ode = ode
        self.device = device
        if ode is True:
            z0_outs = hidden_size // 2
            self.z0_net = nn.LSTM(input_size, z0_outs, num_layers=1, batch_first=True)
            self.ode_func_net = nn.Sequential(
                nn.Linear(2*z0_outs, hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, 2*z0_outs),
            )
            self.ode_func = ODEFunc(net=self.ode_func_net, device=device)
            self.ode_solver = DiffEqSolver(self.ode_func, method, device=device).to(device)
        else:
            self.lstm_net = nn.LSTM(input_size, hidden_size, num_layers=1, batch_first=True)
        
        self.mean_net = nn.Linear(hidden_size * (1 + int(static)), z_size)
        self.std_net = nn.Linear(hidden_size * (1 + int(static)), z_size)
        self.static = static
    
    def forward(self, inputs, seq_len):
        timesteps_to_predict = torch.from_numpy(np.arange(seq_len, dtype=np.int64)) / seq_len

        if self.ode is True:
            outs, hidden = self.z0_net(inputs)
            hidden = torch.cat(hidden, 2).squeeze(0)
            outs = self.ode_solver(hidden, timesteps_to_predict)

        elif self.ode is False:
            outs, hidden = self.lstm_net(inputs)
        
        if self.static:
            hidden = torch.cat(hidden, 2).squeeze(0)
            mean = self.mean_net(hidden)
            std = F.softplus(self.std_net(hidden))
        else:
            mean = self.mean_net(outs)
            std = F.softplus(self.std_net(outs))

        return mean, std 

class ConvGRUEncoder(nn.Module):
    def __init__(self, in_ch, out_ch, opt, device, resize, type='static'):
        super(ConvGRUEncoder, self).__init__()
        self.opt = opt
        self.device = device
        self.in_ch = in_ch
        self.out_ch = out_ch
        self.resolution_after_encoder = (opt.resolution // resize, opt.resolution // resize)
        self.resize = 1
        self.type = type

        if opt.encoder in ['cgru', 'cgru_sa']:
            self.convgru_cell = ConvGRUCell(self.resolution_after_encoder, in_ch, out_ch, 5).to(device)
            if self.type == 'dynamic':
                if opt.rim is True:
                    self.dynamic_net = RIM_CGRU(opt.emsize, opt)
                else:
                    self.dynamic_convgru_cell = ConvGRUCell(self.resolution_after_encoder, out_ch, out_ch, 5).to(device)

        elif opt.encoder == 'odecgru':
            self.build_odecgru_nets()
                
        self.mean_net = nn.Sequential(
            nn.Conv2d(out_ch, out_ch, 3, 1, 1), nn.ReLU(),
            nn.Conv2d(out_ch, 128, 3, 1, 1), nn.ReLU(),
            nn.Conv2d(128, out_ch, 3, 1, 1))

        self.std_net = nn.Sequential(
            nn.Conv2d(out_ch, out_ch, 3, 1, 1), nn.ReLU(),
            nn.Conv2d(out_ch, 128, 3, 1, 1), nn.ReLU(),
            nn.Conv2d(128, out_ch, 3, 1, 1))

    def build_odecgru_nets(self):
        self.ode_encoder_func = ODEFunc(n_inputs=self.in_ch, n_outputs=self.in_ch, n_layers=3, n_units=self.opt.neural_ode_n_units, downsize=False, nonlinear='relu', device=self.device)
        # Encoding using ODEConvGRU: Feed self.ode_func to ODEConvGRU cell to solve diff equations and to find z0
        self.ode_convgru_cell = ODEConvGRUCell(self.ode_encoder_func, self.opt, self.resolution_after_encoder, self.in_ch, out_ch=self.out_ch, device=self.device)
        self.ode_decoder_func = ODEFunc(n_inputs=self.out_ch, n_outputs=self.out_ch, n_layers=3, n_units=self.opt.neural_ode_n_units, downsize=False, nonlinear='relu', device=self.device)
        # Neural ODE decoding: uses `self.ode_decoder_func` to solve IVP differential equation in latent space
        self.diffeq_solver = DiffEqSolver(self.ode_decoder_func, self.opt.decode_diff_method, device=self.device, memory=False)

    def forward(self, inputs, seq_len=10):

        hiddens, hidden = self.convgru_cell(inputs, None, inputs.size()[0])
        
        if self.type == 'static':
            mean = self.mean_net(hidden)
            std = F.softplus(self.std_net(hidden))

        elif self.type == 'dynamic':
            hiddens, _ = self.dynamic_convgru_cell(None, hidden, seq_len)

        if self.type in ['dynamic', 'prior']:
            t, b, c, h, w = hiddens.size()
            mean = self.mean_net(hiddens.view(-1, c, h, w))
            std = F.softplus(self.std_net(hiddens.view(-1, c, h, w)))
            _, c, h, w = mean.size()
            mean = mean.view(t, b, c, h, w).permute(1, 0, 2, 3, 4)
            std = std.view(t, b, c, h, w).permute(1, 0, 2, 3, 4)
        
        return mean, std

class Decoder(nn.Module):
    def __init__(self, in_ch, final_dim, opt):
        super(Decoder, self).__init__()
        if opt.encoder == 'default':
            self.layers = nn.Sequential(
                nn.ConvTranspose2d(in_ch, 512, 4, 1, 0), nn.BatchNorm2d(512), nn.ReLU(),
                nn.Upsample(scale_factor=2),
                nn.Conv2d(512, 256, 3, 1, 1), nn.BatchNorm2d(256), nn.ReLU(),
                nn.Upsample(scale_factor=2),
                nn.Conv2d(256, 128, 3, 1, 1), nn.BatchNorm2d(128), nn.ReLU(),
                nn.Upsample(scale_factor=2),
                nn.Conv2d(128, 128, 3, 1, 1), nn.BatchNorm2d(128), nn.ReLU(),
                nn.Upsample(scale_factor=2),
                nn.Conv2d(128, 64, 3, 1, 1), nn.BatchNorm2d(64), nn.ReLU(),
                nn.Conv2d(64, final_dim, 1, 1, 0))
        
        elif opt.encoder in ['odecgru', 'cgru']:
            self.layers = nn.Sequential(
                nn.ConvTranspose2d(in_ch, 256, 3, 1, 1), nn.BatchNorm2d(256), nn.ReLU(),
                nn.Upsample(scale_factor=2),
                nn.Conv2d(256, 256, 3, 1, 1), nn.BatchNorm2d(256), nn.ReLU(),
                nn.Upsample(scale_factor=2),
                nn.Conv2d(256, 128, 3, 1, 1), nn.BatchNorm2d(128), nn.ReLU(),
                nn.Upsample(scale_factor=2),
                nn.Conv2d(128, 128, 3, 1, 1), nn.BatchNorm2d(128), nn.ReLU(),
                nn.Upsample(scale_factor=2),
                nn.Conv2d(128, 64, 3, 1, 1), nn.BatchNorm2d(64), nn.ReLU(),
                nn.Conv2d(64, final_dim, 1, 1, 0))
        
        elif opt.encoder in ['cgru_sa']:
            self.layers = nn.Sequential(
                nn.ConvTranspose2d(in_ch, 256, 3, 1, 1), nn.BatchNorm2d(256), nn.ReLU(),
                nn.Upsample(scale_factor=2),
                nn.Conv2d(256, 256, 3, 1, 1), nn.BatchNorm2d(256), nn.ReLU(),
                nn.Upsample(scale_factor=2),
                nn.Conv2d(256, 128, 3, 1, 1), nn.BatchNorm2d(128), nn.ReLU(),
                nn.Upsample(scale_factor=2),
                nn.Conv2d(128, 64, 3, 1, 1), nn.BatchNorm2d(64), nn.ReLU(),
                nn.Conv2d(64, final_dim, 1, 1, 0))
    
    def forward(self, inputs):
        return self.layers(inputs)

class DFP(nn.Module):
    def __init__(self, z_size=128):
        super().__init__()
        self.main_net = nn.Sequential(
            nn.Linear(in_features=z_size, out_features=z_size),
            nn.Linear(in_features=z_size, out_features=z_size)
        )
        self.mean = nn.Linear(in_features=z_size, out_features=3)
        self.std = nn.Linear(in_features=z_size, out_features=3)


    def forward(self, batch):
        shape, feature_shape = batch.shape[:-1], batch.shape[-1]
        print(shape, feature_shape)
        batch = batch.reshape(-1, feature_shape)
        features = self.main_net(batch)
        feature_shape = features.shape[1:]
        features = features.reshape(*shape, *feature_shape)
        # features = self.main_net(z_t)
        mean = F.softmax(self.mean(features))
        std = F.softplus(self.std(features))
        return dist.Normal(loc=mean, scale=std)