import sys
sys.path.append('..')

import torch
import torch.nn as nn

from modules.DiffEqSolver import ODEFunc, DiffEqSolver
from modules.ODEConvGRUCell import ODEConvGRUCell

import helpers.utils as utils

class Encoder(nn.Module):
    
    def __init__(self, input_dim=3, ch=64, n_downs=2, device=None):
        super(Encoder, self).__init__()
        cnn_encoder = [nn.Conv2d(input_dim, ch, 3, 1, 1), nn.BatchNorm2d(ch), nn.ReLU()]
        
        for _ in range(n_downs):
            cnn_encoder += [nn.Conv2d(ch, ch * 2, 4, 2, 1), nn.BatchNorm2d(ch * 2), nn.ReLU()]
            ch *= 2
        
        self.cnn_encoder = nn.Sequential(*cnn_encoder).to(device) # CNN Embedding

    def forward(self, x):
        out = self.cnn_encoder(x)
        return out

class Decoder(nn.Module):
    def __init__(self, input_dim=256, output_dim=3, n_ups=2, opt=None, device=None):
        super(Decoder, self).__init__()
        model = []
        ch = input_dim

        for _ in range(n_ups):
            model += [nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)]
            model += [nn.Conv2d(ch, ch // 2, 3, 1, 1), nn.BatchNorm2d(ch // 2), nn.ReLU()]
            ch = ch // 2
        
        model += [nn.Conv2d(ch, output_dim, 3, 1, 1)]
        self.cnn_decoder = nn.Sequential(*model).to(device)
    
    def forward(self, x):
        out = self.cnn_decoder(x)
        return out

class VidODE(nn.Module):
    def __init__(self, opt, device):
        super(VidODE, self).__init__()

        ch = 32
        h, w = opt.resolution, opt.resolution
        resize = 2 ** opt.n_downs
        in_dim, out_dim = ch * resize, ch * resize
        resolution_after_encoder = (h // resize, w // resize)

        self.encoder = Encoder(input_dim=opt.in_channels, ch=ch, n_downs=opt.n_downs, device=device).to(device)

        self.ode_encoder_func = ODEFunc(n_inputs=in_dim, 
                                            n_outputs=out_dim, 
                                            n_layers=opt.n_layers, 
                                            n_units=in_dim // 2,
                                            downsize=False,
                                            nonlinear='relu',
                                            final_act=False,
                                            device=device)

        self.ode_convgru_cell = ODEConvGRUCell(self.ode_encoder_func, 
                                                    opt, 
                                                    resolution_after_encoder, 
                                                    out_dim, 
                                                    device=device)

        # ode_func_netD = utils.create_convnet(n_inputs=in_dim, n_outputs=out_dim, n_layers=opt.n_layers, n_units=in_dim // 2, final_act=False)

    
    def forward(self, inputs, batch_dict):
        pass

    def get_prediction(self):
        pass

    def get_loss(self):
        pass