import sys
sys.path.append('../')

import torch
import torch.nn as nn
import torch.nn.functional as F

from modules.ODEConvGRUCell import ODEConvGRUCell
from modules.DiffEqSolver import ODEFunc, DiffEqSolver
import helpers.utils as utils

class ODEConvGRU(nn.Module):
    def __init__(self, opt, device):
        super(ODEConvGRU, self).__init__()
        self.opt = opt
        self.device = device
        self.resize = 1
        h, w = opt.resolution, opt.resolution
        self.resize = 2 ** opt.n_downs
        resolution_after_encoder = (h // self.resize, w // self.resize)
        conv_encoder_out_ch = opt.conv_encoder_out_ch
        
        # 1. `conv_encoder` encodes input frames
        # 2. `ODEConvGRUCell` uses encoded inputs and `ode_encoder_func` to find z0 in latent space
        # 3. `diffeq_solver` solves IVP: Given z0 and [t_i,....t_(i+n)], it uses Neural ODE Decoder (`ode_decoder_func`) to predict [z_i,...z_(i+n)] in latent space
        # 4. `conv_decoder` to decode [z_i,...z_(i+n)] back to pixel space

        self.conv_encoder = Encoder(opt.in_channels, 
                                    conv_encoder_out_ch, 
                                    opt.n_downs, nonlinear='leaky_relu').to(device)
        
        # Init encoder ODE with a convnet
        self.ode_encoder_func = ODEFunc(n_inputs=conv_encoder_out_ch, 
                            n_outputs=conv_encoder_out_ch, 
                            n_layers=opt.n_ode_layers, 
                            n_units=opt.neural_ode_n_units,
                            downsize=False,
                            nonlinear='relu',
                            device=device, final_act=False)
        
        # Encoding using ODEConvGRU: Feed self.ode_func to ODEConvGRU cell to solve diff equations and to find z0
        self.ode_convgru_cell = ODEConvGRUCell(self.ode_encoder_func, opt, resolution_after_encoder, conv_encoder_out_ch, device=device)

        # Init decoder neural ODE with a convnet
        self.ode_decoder_func = ODEFunc(n_inputs=conv_encoder_out_ch, 
                            n_outputs=opt.neural_ode_decoder_out_ch, 
                            n_layers=opt.n_ode_layers, 
                            n_units=opt.neural_ode_n_units,
                            downsize=False,
                            nonlinear='relu',
                            device=device, final_act=False)

        # Neural ODE decoding: uses `self.ode_decoder_func` to solve IVP differential equation in latent space
        self.diffeq_solver = DiffEqSolver(self.ode_decoder_func, opt.decode_diff_method, device=device, memory=self.opt.mem)
        self.conv_decoder = Decoder(opt.neural_ode_decoder_out_ch, opt.in_channels, opt.n_downs, nonlinear='leaky_relu').to(device)

    def forward(self, inputs, batch_dict):
        b, t, c, h, w = inputs.size()
        observed_tp = batch_dict['observed_tp']
        time_steps_to_predict = batch_dict['tp_to_predict']

        # 1. ConvEncode the input frames
        encoded_inputs = self.conv_encoder(inputs.view(b*t, c, h, w))
        _, c_, h_, w_ = encoded_inputs.size()
        encoded_inputs = encoded_inputs.view(b, -1, c_, h_, w_)
        
        # 2. ODEConvGRUCell to predict (first_point_mu, first_point_std) for z_0
        encoded_inputs = encoded_inputs.permute(1, 0, 2, 3, 4) # Make time dim first
        first_point_mu, first_point_std = self.ode_convgru_cell(encoded_inputs, observed_tp)

        # Sampling latent features
        if self.opt.z_sample is True:
            # first_point_enc = Gaussian(first_point_mu, first_point_std) might introduce stochasticity in the ODEConvGRU model
            # TODO: reparametrization trick or this might not work
            # sampled_z = torch.normal(mean=first_point_mu, std=first_point_std)
            # first_point_enc = sampled_z.unsqueeze(0).repeat(1, 1, 1, 1, 1).squeeze(0)
            pass
        else:
            first_point_enc = first_point_mu.unsqueeze(0).repeat(1, 1, 1, 1, 1).squeeze(0)
        
        # ODE decoding: Given first_point_enc (z_0) and time_steps_to_predict([t_i,....t_(i+n)]), we predict sol_y([z_i,...z_(i+n)])
        sol_y = self.diffeq_solver(first_point_enc, time_steps_to_predict) # returns time dim first

        t, b, c, h, w = sol_y.size()
        pred_x = F.sigmoid(self.conv_decoder(sol_y.view(b*t, c, h, w)))
        _, c, h, w = pred_x.size()
        pred_x = pred_x.view(t, b, c, h, w).permute(1, 0, 2, 3, 4)
        return pred_x
        
    def get_prediction(self, inputs, batch_dict=None):
        pred_x = self(inputs, batch_dict) # Range in [-1, 1] due to last tanh activation
        return pred_x 
        
    def get_loss(self, pred_frames, truth, loss='MSE'):
        b, t, c, h, w = truth.size()
        if loss == 'MSE':   loss_function = nn.MSELoss().cuda()

        loss = loss_function(pred_frames.reshape(b*t, c, h, w), truth.reshape(b*t, c, h, w))
        return loss

class Encoder(nn.Module):
    def __init__(self, n_inputs, out_ch, n_downs, nonlinear='relu'):
        super(Encoder, self).__init__()

        chan = 16
        if nonlinear == 'relu':         nonlinear = nn.ReLU()
        elif nonlinear == 'leaky_relu': nonlinear = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        else:                           raise NotImplementedError('Wrong activation function')

        layers = [nn.Conv2d(n_inputs, chan, 3, 2, 1), nonlinear]
        for _ in range(n_downs - 2):
            layers = [nn.Conv2d(chan, chan*2, 3, 2, 1), nonlinear]
            chan *= 2

        layers += [nn.Conv2d(chan, out_ch, 3, 2, 1), nonlinear]
        self.encoder = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.encoder(x)


class Decoder(nn.Module):
    def __init__(self, n_inputs, out_ch, n_ups, nonlinear='relu'):
        super(Decoder, self).__init__()

        if nonlinear == 'relu':         nonlinear = nn.ReLU()
        if nonlinear == 'leaky_relu':   nonlinear = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        else:                           raise NotImplementedError('Wrong activation function')

        chan = 32
        layers = [nn.ConvTranspose2d(n_inputs, chan, 4, 2, 1), nonlinear]
        for _ in range(n_ups - 2):
            layers = [nn.ConvTranspose2d(chan, chan // 2, 4, 2, 1), nonlinear]
            chan = chan // 2

        layers += [nn.ConvTranspose2d(chan, out_ch, 4, 2, 1)]
        self.decoder = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.decoder(x)



