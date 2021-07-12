import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from modules.DiffEqSolver import DiffEqSolver, ODEFunc
from modules.ConvGRUCell import ConvGRUCell
from modules.ODEConvGRUCell import ODEConvGRUCell

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

    def forward(self, inputs):
        return self.layers(inputs)

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
    
    def forward(self, inputs):
        b, t, _ = inputs.size()
        timesteps_to_predict = torch.from_numpy(np.arange(t, dtype=np.int64)) / t

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
    def __init__(self, in_ch, out_ch, opt, device, resize, prior=False, static=True):
        super(ConvGRUEncoder, self).__init__()
        self.opt = opt
        self.device = device
        self.static = static
        self.prior = prior
        self.in_ch = in_ch
        self.out_ch = out_ch
        self.resolution_after_encoder = (opt.resolution // resize, opt.resolution // resize)

        if self.static or opt.encoder == 'cgru':
            self.convgru_cell = ConvGRUCell(self.resolution_after_encoder, in_ch, out_ch, 5).to(device)
        
        else:
            if opt.encoder == 'odecgru':
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

    def forward(self, inputs, seq_len):

        if self.static or self.opt.encoder == 'cgru':
            hiddens, hidden = self.convgru_cell(inputs, None, seq_len)

            if self.static and self.prior is False:
                mean = self.mean_net(hidden)
                std = F.softplus(self.std_net(hidden))

            elif self.prior or self.opt.encoder == 'cgru':
                t, b, c, h, w = hiddens.size()
                mean = self.mean_net(hiddens.view(-1, c, h, w))
                std = F.softplus(self.std_net(hiddens.view(-1, c, h, w)))
                mean = mean.view(t, b, c, h, w).permute(1, 0, 2, 3, 4)
                std = std.view(t, b, c, h, w).permute(1, 0, 2, 3, 4)
        
        else:
            if self.opt.encoder == 'odecgru':
                t, b, c, h, w = inputs.size()
                timesteps_to_predict = torch.from_numpy(np.arange(t, dtype=np.int64)) / t
                first_point_mu, first_point_std = self.ode_convgru_cell(inputs, timesteps_to_predict)
                sol_z = self.diffeq_solver(first_point_mu, timesteps_to_predict)    # Get z1...zt
                t, b, c, h, w = sol_z.size()
                sol_z = sol_z.view(-1, c, h, w) 
                mean = self.mean_net(sol_z)
                std = F.softplus(self.std_net(sol_z))
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
        batch = batch.reshape(-1, feature_shape)
        features = self.main_net(batch)
        feature_shape = features.shape[1:]
        features = features.reshape(*shape, *feature_shape)
        # features = self.main_net(z_t)
        mean = self.mean(features)
        std = F.softplus(self.std(features))
        return dist.Normal(loc=mean, scale=std)