import sys
sys.path.append('../')

import torch
import torch.nn as nn

from modules.ODEConvGRUCell import ODEConvGRUCell
from modules.DiffEqSolver import ODEFunc, DiffEqSolver
import helpers.utils as utils

class ODEConvGRU(nn.Module):
    def __init__(self, opt, device, downsize=True):
        super(ODEConvGRU, self).__init__()

        self.device = device
        self.resize = 1
        h, w = opt.resolution, opt.resolution
        upsize = downsize
        if downsize is True:    self.resize = 2 ** opt.n_layers
        resolution_after_encoder = (h // self.resize, w // self.resize)
        
        # 1. `conv_encoder` encodes input frames
        # 2. `ODEConvGRUCell` uses encoded inputs and `ode_encoder_func` to find z0 in latent space
        # 3. `diffeq_solver` solves IVP: Given z0 and [t_i,....t_(i+n)], it uses Neural ODE Decoder (`ode_decoder_func`) to predict [z_i,...z_(i+n)] in latent space
        # 4. `conv_decoder` to decode [z_i,...z_(i+n)] back to pixel space
        self.conv_encoder = utils.create_convnet(opt.in_channels, 
                                                opt.conv_encoder_out_ch, 
                                                n_layers=opt.n_layers, 
                                                n_units=opt.conv_n_units, 
                                                downsize=downsize, 
                                                nonlinear='relu').to(device)
        
        # Init encoder ODE with a convnet
        self.ode_encoder_func = ODEFunc(n_inputs=opt.conv_encoder_out_ch, 
                            n_outputs=opt.neural_ode_encoder_out_ch, 
                            n_layers=opt.n_ode_layers, 
                            n_units=opt.neural_ode_n_units,
                            downsize=False,
                            nonlinear='relu',
                            device=device)
        
        # Encoding using ODEConvGRU: Feed self.ode_func to ODEConvGRU cell to solve diff equations and to find z0
        self.ode_convgru_cell = ODEConvGRUCell(self.ode_encoder_func, opt, resolution_after_encoder, device=device)

        # Init decoder neural ODE with a convnet
        self.ode_decoder_func = ODEFunc(n_inputs=opt.neural_ode_encoder_out_ch, 
                            n_outputs=opt.neural_ode_decoder_out_ch, 
                            n_layers=opt.n_ode_layers, 
                            n_units=opt.neural_ode_n_units,
                            downsize=False,
                            nonlinear='relu',
                            device=device)
        
        # Neural ODE decoding: uses `self.ode_decoder_func` to solve IVP differential equation in latent space
        self.diffeq_solver = DiffEqSolver(self.ode_decoder_func, opt.decode_diff_method, device=device)

        self.conv_decoder = utils.create_transpose_convnet(opt.neural_ode_decoder_out_ch, 
                                                opt.in_channels, 
                                                n_layers=opt.n_layers, 
                                                n_units=opt.conv_n_units, 
                                                upsize=upsize, 
                                                nonlinear='relu').to(device)

    def forward(self, inputs, batch_dict):
        b, t, c, h, w = inputs.size()
        observed_tp = batch_dict['observed_tp']
        time_steps_to_predict = batch_dict['tp_to_predict']
        pred_t_len = len(time_steps_to_predict)

        # ConvEncode the input frames
        encoded_inputs = self.conv_encoder(inputs.view(b*t, c, h, w))
        encoded_inputs = encoded_inputs.view(b, -1, encoded_inputs.size()[-3], encoded_inputs.size()[-2], encoded_inputs.size()[-1])
        
        # ODEConvGRUCell to infer (first_point_mu, first_point_std) for z_0
        first_point_mu, first_point_std = self.ode_convgru_cell(encoded_inputs, observed_tp)
        
        # Sampling latent features
        # TODO: Instead of using `first_point_mu` as `first_point_enc`, try using first_point_enc = Gaussian(first_point_mu, first_point_std); might introduce stochasticity in the ODEConvGRU model
        first_point_enc = first_point_mu.unsqueeze(0).repeat(1, 1, 1, 1, 1).squeeze(0)

        # ODE decoding: Given first_point_enc (z_0) and time_steps_to_predict([t_i,....t_(i+n)]), we predict sol_y([z_i,...z_(i+n)])
        sol_y = self.diffeq_solver(first_point_enc, time_steps_to_predict)
        sol_y = sol_y.contiguous().view(b, pred_t_len, -1, h // self.resize, w // self.resize)  

        # ConvDecode [z_i,...z_(i+n)] to get predictions pred_x [x_hat_i,...x_hat_(i+n)] 
        b, t, c, h, w = sol_y.size()
        pred_x = self.conv_decoder(sol_y.view(b*t, c, h, w))
        pred_x = pred_x.view(b, pred_t_len, -1, h * self.resize, w * self.resize)
        return pred_x

    def get_prediction(self, inputs, batch_dict=None):
        pred_x = self(inputs, batch_dict)
        return pred_x
        
    def get_loss(self, pred_frames, truth):
        """ Returns the reconstruction loss calculated as MSE Error """
        b, t, c, h, w = truth.size()
        loss_function = nn.MSELoss().cuda()
        loss = loss_function(pred_frames, truth) / ( b * t )
        return loss