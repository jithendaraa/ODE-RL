import sys
sys.path.append('../')

import torch
import torch.nn as nn
import copy

from modules.ConvGRUCell import ConvGRUCell
from modules.DiffEqSolver import DiffEqSolver

class ConvGRU(nn.Module):
    
    def __init__(self, opt, device, activation='leaky_relu', decODE=False, decoder_out_channels=None):
        super(ConvGRU, self).__init__()
        
        self.opt = opt
        self.decODE = decODE
        self.device = device
        h, w = opt.resolution, opt.resolution
        resize = 2 ** opt.depth
        encoder_resolution = h // resize, w // resize
        dtype = torch.cuda.FloatTensor if self.device == 'cuda' else torch.FloatTensor
        
        encoder_out_channels = opt.conv_encoder_out_ch

        if decoder_out_channels is None:
            decoder_out_channels = opt.in_channels
            self.decoder_out_channels = decoder_out_channels
        else:
            self.decoder_out_channels = decoder_out_channels
        
        if opt.phase == 'train':
            self.n_input_frames = self.opt.train_in_seq
            self.n_output_frames = self.opt.train_out_seq
        else:
            self.n_input_frames = self.opt.test_in_seq
            self.n_output_frames = self.opt.test_out_seq

        self.encoder = Encoder(in_channels=opt.in_channels, out_channels=encoder_out_channels, act=activation, dtype=dtype, opt=opt, device=device).to(device)
        self.hidden_state_channels = self.encoder.get_hidden_state_channels()
        
        if decODE:
            self.decODEr = DecODEr(in_channels=encoder_out_channels, 
                                    out_channels=self.decoder_out_channels, 
                                    act=activation, 
                                    dtype=dtype, 
                                    opt=opt, 
                                    device=device, 
                                    resolution=encoder_resolution, n_frames=self.n_output_frames).to(device)
        else:
            self.decoder = Decoder(in_channels=encoder_out_channels, 
                                    out_channels=self.decoder_out_channels, 
                                    hidden_state_channels=self.hidden_state_channels,
                                    act=activation, dtype=dtype, opt=opt, device=device, resolution=encoder_resolution, n_frames=self.n_output_frames).to(device)
    
    def forward(self, inputs, batch_dict=None):
        encoded_inputs = self.encoder(inputs)
        hidden_states = encoded_inputs[::-1] # reverse list which has length self.opt.depth
        
        if self.decODE is True:
            time_steps_to_predict = batch_dict['tp_to_predict']
            pred_x = self.decODEr(hidden_states, time_steps_to_predict)
        else:
            pred_x = self.decoder(hidden_states)

        return pred_x / 2.0

    def get_prediction(self, inputs, batch_dict=None):
        pred_x = self(inputs, batch_dict) # Range in [-1, 1] due to last tanh activation
        return pred_x
    
    def get_loss(self, pred_frames, truth, loss='MSE'):
        """ Returns the reconstruction loss calculated as MSE Error """
        if loss == 'MSE':
            b, t, c, h, w = truth.size()
            loss_function = nn.MSELoss().cuda()
            loss = loss_function(pred_frames.view(b*t, c, h, w), truth.view(b*t, c, h, w)) 
        
        elif loss == 'ELBO':            # ELBO loss if prediction were done as in variational inference: sum log prob(x_i|z_i) + log prob(z0) + 
            loss = None
            pass

        return loss


class Encoder(nn.Module):
    def __init__(self, in_channels=1, out_channels=96, act='leaky_relu', dtype=None, opt=None, device=None):
        super(Encoder, self).__init__()
        
        self.opt = opt
        b = self.opt.batch_size
        self.depth = opt.depth
        h, w = opt.resolution, opt.resolution
        self.out_channels = out_channels
        self.device = device

        if act == 'relu':
            nonlinear = nn.ReLU()
        elif act == 'tanh':
            nonlinear = nn.Tanh()
        elif act == 'leaky_relu':
            nonlinear = nn.LeakyReLU(negative_slope=0.2, inplace=True)

        chan = 16
        self.conv_encoders = nn.ModuleList()
        self.conv_gru_cells = nn.ModuleList()
        self.hiddens = {}
        resize = 2
        self.hidden_state_channels = []

        # Make pairs of conv_encoders and convGRU cells
        if self.depth == 1:
            conv_encoders = []
            conv_encoders += [nn.Conv2d(in_channels, chan, 3, 2, 1)]
            conv_encoders += [nonlinear]
            conv_encoders += [nn.Conv2d(chan, chan*2, 3, 1, 1)]
            conv_encoders += [nonlinear]
            conv_encoders += [nn.Conv2d(chan*2, opt.conv_encoder_out_ch, 3, 1, 1)]
            conv_encoders += [nonlinear]
            conv_encoders = nn.Sequential(*conv_encoders).to(self.device)
            self.conv_encoders.append(conv_encoders)

            self.hiddens[0] = [torch.zeros((b, opt.convgru_out_ch, h // resize, w // resize)).to(self.device)]
            conv_gru_cell = ConvGRUCell((h // resize, w // resize), opt.conv_encoder_out_ch, opt.convgru_out_ch, 5, bias=True, dtype=dtype, padding=2).to(self.device)
            self.conv_gru_cells.append(conv_gru_cell)
            self.hidden_state_channels.append(opt.convgru_out_ch)

        else:
            conv_encoders = []
            conv_encoders += [nn.Conv2d(in_channels, chan, 3, 2, 1)]
            conv_encoders += [nonlinear]
            conv_encoders = nn.Sequential(*conv_encoders).to(self.device)
            self.conv_encoders.append(conv_encoders)

            self.hiddens[0] = [torch.zeros((b, chan*2, h // resize, w // resize)).to(self.device)]
            conv_gru_cell = ConvGRUCell((h // resize, w // resize), chan, chan*2, 5, bias=True, dtype=dtype, padding=2).to(self.device)
            self.conv_gru_cells.append(conv_gru_cell)
            self.hidden_state_channels.append(chan*2)
            chan *= 2
            
            for i in range(1, self.depth):

                conv_encoders, conv_gru_cell = [], []
                
                if i == (self.depth - 1):          # Final layer
                    conv_encoders += [nn.Conv2d(chan, opt.conv_encoder_out_ch, 3, 2, 1)]  # half the resolution every time
                    conv_encoders += [nonlinear]
                    conv_encoders = nn.Sequential(*conv_encoders).to(self.device)
                    self.conv_encoders.append(conv_encoders)
                    
                    resize *= 2
                    conv_gru_cell = ConvGRUCell((h // resize, w // resize), opt.conv_encoder_out_ch, opt.convgru_out_ch, kernel_size=5, bias=True, dtype=dtype, padding=2).to(self.device)
                    self.conv_gru_cells.append(conv_gru_cell)
                    self.hiddens[i] = [torch.zeros((b, opt.convgru_out_ch, h // resize, w // resize)).to(self.device)]
                    hidden_channels = opt.convgru_out_ch

                else:
                    conv_encoders += [nn.Conv2d(chan, chan, 3, 2, 1)]  # half the resolution every time
                    conv_encoders += [nonlinear]
                    conv_encoders = nn.Sequential(*conv_encoders).to(self.device)
                    self.conv_encoders.append(conv_encoders)

                    resize *= 2
                    self.hiddens[i] = [torch.zeros((b, chan*2, h // resize, w // resize)).to(self.device)]

                    conv_gru_cell = ConvGRUCell((h // resize, w // resize), chan, chan*2, kernel_size=5, bias=True, dtype=dtype, padding=2).to(self.device)       # twice the channel every time
                    self.conv_gru_cells.append(conv_gru_cell)
                    hidden_channels = chan*2
                    chan *= 2
                
                self.hidden_state_channels.append(hidden_channels)
        
        self.init_hiddens = copy.deepcopy(self.hiddens)
        self.init_hidden_state_channels = self.hidden_state_channels.copy()

    def get_hidden_state_channels(self):
        return self.hidden_state_channels[::-1]

    def forward(self, inputs):
        
        self.hiddens = copy.deepcopy(self.init_hiddens)
        self.hidden_state_channels = self.init_hidden_state_channels.copy()

        b, in_frames, c, h, w = inputs.size()
        ins = [inputs]
        device = self.device

        for i in range(self.depth):
            in_ = ins[-1]

            if len(in_.size()) == 5:
                in_ = in_.view(-1, in_.size()[-3], in_.size()[-2], in_.size()[-1])

            encoded_inputs_i = self.conv_encoders[i](in_)

            _, ei_ch, ei_h, ei_w = encoded_inputs_i.size()
            encoded_inputs_i = encoded_inputs_i.view(b, in_frames, ei_ch, ei_h, ei_w).permute(1, 0, 2, 3, 4) # t, b, c, h, w

            for ei_input in encoded_inputs_i:
                h_prev = self.hiddens[i][-1]
                h_next = self.conv_gru_cells[i](ei_input, h_prev).to(device)
                self.hiddens[i].append(h_next)
        
            _, h_next_ch, h_next_h, h_next_w = h_next.size()
            hiddens = torch.stack(self.hiddens[i][1:]).to(device).view(-1, h_next_ch, h_next_h, h_next_w)
            ins.append(hiddens)
            
        hidden_states = []
        for i in range(self.depth):
            if self.opt.decODE is False:
                hidden_states.append(self.hiddens[i][-1])
            else:
                hiddens = torch.stack(self.hiddens[i][1:][::-1]).to(device) # Reversed for running backwards in time from h_(n-1) to h0 to infer z0
                hidden_states.append(hiddens)                               # has self.depth elements with each element being t, b, c, h , w
        
        return hidden_states


class Decoder(nn.Module):
    
    def __init__(self, in_channels, out_channels, hidden_state_channels, n_frames, act='leaky_relu', dtype=None, opt=None, device=None, resolution=(16, 16)):
        super(Decoder, self).__init__()
        
        assert in_channels == hidden_state_channels[0]
        self.in_channels = in_channels
        self.out_channels = out_channels
        hidden_state_channels.append(out_channels)
        self.hidden_state_channels = hidden_state_channels
        self.opt = opt
        self.device = device
        self.final_nonlinear = nn.Tanh()

        self.depth = opt.depth
        self.encoder_resolution = resolution
        self.n_frames = n_frames

        h, w = opt.resolution, opt.resolution
        e_h, e_w = resolution # h,w after Encoder
        resize = 1

        if act == 'relu':
            nonlinear = nn.ReLU()
        elif act == 'tanh':
            nonlinear = nn.Tanh()
        elif act == 'leaky_relu':
            nonlinear = nn.LeakyReLU(negative_slope=0.2, inplace=True)

        self.conv_decoders = nn.ModuleList()
        self.conv_gru_cells = nn.ModuleList()
        chan = 16           # value of chan from Encoder

        if self.depth == 1:
            conv_gru_ch = self.hidden_state_channels[0]
            last_ch = self.hidden_state_channels[1]

            conv_gru_cell = ConvGRUCell((e_h * resize, e_w * resize), conv_gru_ch, conv_gru_ch, 5, bias=True, dtype=dtype, padding=2).to(self.device)
            self.conv_gru_cells.append(conv_gru_cell)

            conv_decoders = []
            conv_decoders += [nn.ConvTranspose2d(conv_gru_ch, chan*2, 4, 2, 1)]
            conv_decoders += [nonlinear]
            conv_decoders += [nn.ConvTranspose2d(chan*2, chan, 3, 1, 1)]
            conv_decoders += [nonlinear]
            conv_decoders += [nn.ConvTranspose2d(chan, last_ch, 3, 1, 1)]
            conv_decoders += [self.final_nonlinear]
            conv_decoders = nn.Sequential(*conv_decoders).to(self.device)
            self.conv_decoders.append(conv_decoders)

        else:
            conv_gru_ch = self.hidden_state_channels[0]
            next_ch = self.hidden_state_channels[1]

            conv_gru_cell = ConvGRUCell((e_h * resize, e_w * resize), conv_gru_ch, conv_gru_ch, 5, bias=True, dtype=dtype, padding=2).to(self.device)
            self.conv_gru_cells.append(conv_gru_cell)

            conv_decoders = []
            conv_decoders += [nn.ConvTranspose2d(conv_gru_ch, next_ch, 4, 2, 1)]
            conv_decoders += [nonlinear]
            conv_decoders = nn.Sequential(*conv_decoders).to(self.device)
            self.conv_decoders.append(conv_decoders)

            for i in range(1, self.depth):
                
                conv_gru_ch = self.hidden_state_channels[i]
                next_ch = self.hidden_state_channels[i+1]
                
                if i == (self.depth - 1):           # Final layer
                    conv_gru_cell = ConvGRUCell((e_h * resize, e_w * resize), conv_gru_ch, conv_gru_ch, 5, bias=True, dtype=dtype, padding=2).to(self.device)
                    self.conv_gru_cells.append(conv_gru_cell)

                    conv_decoders = []
                    conv_decoders += [nn.ConvTranspose2d(conv_gru_ch, next_ch, 4, 2, 1)]
                    conv_decoders += [self.final_nonlinear]
                    conv_decoders = nn.Sequential(*conv_decoders).to(self.device)
                    self.conv_decoders.append(conv_decoders)
                
                else:
                    conv_gru_cell = ConvGRUCell((e_h * resize, e_w * resize), conv_gru_ch, conv_gru_ch, 5, bias=True, dtype=dtype, padding=2).to(self.device)
                    self.conv_gru_cells.append(conv_gru_cell) 

                    conv_decoders = []
                    conv_decoders += [nn.ConvTranspose2d(conv_gru_ch, next_ch, 4, 2, 1)]
                    conv_decoders += [nonlinear]
                    conv_decoders = nn.Sequential(*conv_decoders).to(self.device)
                    self.conv_decoders.append(conv_decoders)
                    resize *= 2

    def forward(self, hidden_states):

        assert len(hidden_states) == self.depth
        e_h, e_w = self.encoder_resolution
        b = self.opt.batch_size
        chan = self.hidden_state_channels[0]
        
        ins = [torch.zeros((self.n_frames, b, chan, e_h, e_w)).to(self.device)]

        for i in range(self.depth):
            in_ = ins[-1]
            h_prev = hidden_states[i]
            new_inputs = []
            
            for input_ in in_:              # iterating over timesteps to decode
                next_input = self.conv_gru_cells[i](input_, h_prev).to(self.device)
                new_inputs.append(next_input)
                h_prev = next_input
            
            new_inputs = torch.stack(new_inputs).to(self.device)

            inputs = new_inputs.view(b*self.n_frames, -1, e_h, e_w)
            inputs = self.conv_decoders[i](inputs).view(b, self.n_frames, -1, e_h*2, e_w*2).permute(1, 0, 2, 3, 4)  # t, b, c, h , w
            e_h, e_w = e_h * 2, e_w * 2
            ins.append(inputs)
        
        pred_x = ins[-1].permute(1, 0, 2, 3, 4)  # Make batch dim first
        predictions_size = [b, self.n_frames, self.opt.in_channels, self.opt.resolution, self.opt.resolution]
        assert list(pred_x.size()) == predictions_size

        return pred_x


# TODO: Incomplete
class DecODEr(nn.Module):
    def __init__(self, in_channels, out_channels, n_frames, act='leaky_relu', dtype=None, opt=None, device=None, resolution=(16, 16)):
        super(DecODEr, self).__init__()
        
        self.n_frames = n_frames
        self.depth = opt.depth
        self.device = device

        # ConvGRU to run backwards in time
        self.convgru_cell = ConvGRUCell(resolution, opt.conv_encoder_out_ch, opt.latent_dim, 5, bias=True, dtype=dtype, padding=2).to(self.device)

        # last conv layer for generating mu, sigma
        z0_dim = opt.latent_dim
        self.z0_dim = z0_dim
        self.transform_z0 = nn.Sequential(
            nn.Conv2d(z0_dim, z0_dim, 1, 1, 0),
            nn.ReLU(),
            nn.Conv2d(z0_dim, z0_dim * 2, 1, 1, 0), )

        # ODE Solver to obtain z_(n+1)...z(n+m) from z0
        # self.diffeq_solver = DiffEqSolver(self.ode_func)
        
        self.hiddens = [torch.zeros(opt.batch_size, opt.latent_dim, resolution[0], resolution[1]).to(self.device)]
        self.init_hiddens = self.hiddens.copy()
    
    def forward(self, inputs, time_steps_to_predict):
        
        # Run ConvGRU backwards in time to predict y0
        latent_ys, y0_s = self.run_convgru_backwards(inputs)        # returns 2 lists of len self.depth
        y0 = y0_s[-1]
        
        # Use y0 to get mu and std for z0's normal distribution
        trans_last_yi = self.transform_z0(y0)
        mean_z0, std_z0 = torch.split(trans_last_yi, self.z0_dim, dim=1)
        std_z0 = std_z0.abs()

        if self.opt.z_sample is True:
            # Dont sample from Normal, use z0 = mu instead; and use MSE Loss
            z0 = first_point_mu.unsqueeze(0).repeat(1, 1, 1, 1, 1).squeeze(0)
        
        elif self.opt.z_sample is False:
            # z0 ~ N(mu, std) and use ELBO loss (sum prediction log prob + KL(q(z0 | input frames) and p(z0)) ); q is decoder and p(z0) ~ N(0, 1)
            pass
        
        return None, None
    
    def run_convgru_backwards(self, inputs):
        latent_ys, y0_s = [], []        # will have self.depth latent_ys and y_0 at the end of this method

        for i, input_ in enumerate(inputs):             # iterating over the `self.depth` hidden states that we receive from Encoder: list of len `self.depth`
            # Input timesteps: n; Output timesteps: m
            # Each input_ has (n, b, c, h, w) tensor
            print(input_.size())
            self.hiddens = self.init_hiddens.copy()
            # Run backwards in time from h_(n-1)....h_0 (Encoder returns the hidden states in reversed time) to use ConvGRU to estimate H_(n-1)...H_0 and use H_0 to estimate z0
            for in_ in input_:      
                # H_(n-1) -> ConvGRU(h_{n-1}, H_{N}); H_x is ConvGRU estimate of h_x
                next_H = self.hiddens[-1]
                curr_h = in_
                prev_H = self.convgru_cell(curr_h, next_H).to(self.device)
                self.hiddens.append(prev_H)
            
            assert len(self.hiddens[1:]) == self.n_frames
            y0 = prev_H                                                # Same as H0
            H_s = torch.stack(self.hiddens[1:]).to(self.device)
            y0_s.append(y)
            latent_ys.append(H_s)                                       # saved as n, b, c, h, w
        
        return latent_ys, y0_s