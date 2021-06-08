import sys
sys.path.append('../')

import torch
import torch.nn as nn

from modules.ConvGRUCell import ConvGRUCell

class ConvGRU(nn.Module):
    
    def __init__(self, opt, device, activation='relu', encoder_out_channels=96, decoder_out_channels=None):
        super(ConvGRU, self).__init__()
        self.opt = opt

        h, w = opt.resolution, opt.resolution
        encoder_resolution = h // 4, w // 4

        if decoder_out_channels is None:
            self.decoder_out_channels = opt.in_channels
        else:
            self.decoder_out_channels = decoder_out_channels
        
        if opt.phase == 'train':
            self.n_input_frames = self.opt.train_in_seq
            self.n_output_frames = self.opt.train_out_seq
        
        else:
            self.n_input_frames = self.opt.test_in_seq
            self.n_output_frames = self.opt.test_out_seq

        self.device = device
        
        kernel_size = (3, 3)
        dtype = torch.cuda.FloatTensor if self.device == 'cuda' else torch.FloatTensor

        self.encoder = Encoder(in_channels=opt.in_channels, out_channels=encoder_out_channels, act='leaky_relu', dtype=dtype, opt=opt, device=device).to(device)
        self.decoder = Decoder(in_channels=encoder_out_channels, out_channels=decoder_out_channels, act='leaky_relu', dtype=dtype, opt=opt, device=device, resolution=encoder_resolution, n_frames=self.n_output_frames).to(device)
    
    def forward(self, inputs):
        encoded_inputs = self.encoder(inputs)
        encoded_inputs = encoded_inputs[::-1] # reverse list
        pred_x = self.decoder(encoded_inputs)
        return pred_x

    def get_prediction(self, inputs):
        pred_x = self(inputs)
        return pred_x
    
    def get_loss(self, pred_frames, truth):
        """ Returns the reconstruction loss calculated as MSE Error """
        b, t, c, h, w = truth.size()
        loss_function = nn.MSELoss().cuda()
        loss = loss_function(pred_frames, truth) / ( b * t )
        return loss


class Encoder(nn.Module):
    def __init__(self, in_channels=1, out_channels=96, act='leaky_relu', dtype=None, opt=None, device=None):
        super(Encoder, self).__init__()
        
        self.opt = opt
        b = self.opt.batch_size
        h, w = opt.resolution, opt.resolution
        self.out_channels = out_channels
        self.device = device

        if act == 'relu':
            nonlinear = nn.ReLU()
        elif act == 'tanh':
            nonlinear = nn.Tanh()
        elif act == 'leaky_relu':
            nonlinear = nn.LeakyReLU(negative_slope=0.2, inplace=True)

        self.e1 = nn.Sequential(
                        nn.Conv2d(in_channels, 16, 3, 1, 1),
                        nonlinear)
        e1_res = (h, w)

        self.cgru1 = ConvGRUCell(e1_res, 16, 64, 5, bias=True, dtype=dtype, padding=2)
        

        self.e2 = nn.Sequential(
                        nn.Conv2d(64, 64, 3, 2, 1),
                        nonlinear)
        e2_res = (h // 2, w // 2)

        self.cgru2 = ConvGRUCell(e2_res, 64, 96, 5, bias=True, dtype=dtype, padding=2)
        

        self.e3 = nn.Sequential(
                        nn.Conv2d(96, 96, 3, 2, 1),
                        nonlinear)
        e3_res = (h // 4, w // 4)

        self.cgru3 = ConvGRUCell(e3_res, 96, out_channels, 5, bias=True, dtype=dtype, padding=2)
        

    def forward(self, inputs):
        b, in_frames, c, h, w = inputs.size()

        self.h_1_t = [torch.zeros((b, 64, h, w)).to(self.device)]
        self.h_2_t = [torch.zeros((b, 96, h // 2, w // 2)).to(self.device)]
        self.h_3_t = [torch.zeros((b, self.out_channels, h // 4, w // 4)).to(self.device)]

        device = inputs.device

        e1 = self.e1(inputs.view(-1, c, h, w))
        _, e1_ch, e1_h, e1_w = e1.size()
        e1 = e1.view(b, in_frames, e1_ch, e1_h, e1_w).permute(1, 0, 2, 3, 4) # t, b, c, h, w

        for e1_input in e1:
            h_prev = self.h_1_t[-1]
            h_next = self.cgru1(e1_input, h_prev).to(device)
            self.h_1_t.append(h_next)
        
        _, h_next_ch, h_next_h, h_next_w = h_next.size()
        h_1_t = torch.stack(self.h_1_t[1:]).to(device).view(-1, h_next_ch, h_next_h, h_next_w)

        e2 = self.e2(h_1_t)
        _, e2_ch, e2_h, e2_w = e2.size()
        e2 = e2.view(b, in_frames, e2_ch, e2_h, e2_w).permute(1, 0, 2, 3, 4) # t, b, c, h, w

        for e2_input in e2:
            h_prev = self.h_2_t[-1]
            h_next = self.cgru2(e2_input, h_prev).to(device)
            self.h_2_t.append(h_next)

        _, h_next_ch, h_next_h, h_next_w = h_next.size()
        h_2_t = torch.stack(self.h_2_t[1:]).to(device).view(-1, h_next_ch, h_next_h, h_next_w)
        
        e3 = self.e3(h_2_t)
        _, e3_ch, e3_h, e3_w = e3.size()
        e3 = e3.view(b, in_frames, e3_ch, e3_h, e3_w).permute(1, 0, 2, 3, 4) # t, b, c, h, w

        for e3_input in e3:
            h_prev = self.h_3_t[-1]
            h_next = self.cgru3(e3_input, h_prev).to(device)
            self.h_3_t.append(h_next)
        
        # final hidden states
        return [self.h_1_t[-1], self.h_2_t[-1], self.h_3_t[-1]]


class Decoder(nn.Module):
    
    def __init__(self, in_channels=96, out_channels=1, resolution=(16, 16), act='leaky_relu', dtype=None, opt=None, device=None, n_frames=None):
        super(Decoder, self).__init__()
        self.opt = opt
        self.device = device
        self.encoder_resolution = resolution
        self.in_channels = in_channels
        e_h, e_w = resolution # h,w after Encoder
        h, w = opt.resolution, opt.resolution
        self.n_frames = n_frames

        if act == 'relu':
            nonlinear = nn.ReLU()
        elif act == 'tanh':
            nonlinear = nn.Tanh()
        elif act == 'leaky_relu':
            nonlinear = nn.LeakyReLU(negative_slope=0.2, inplace=True)

        self.cgru3 = ConvGRUCell(resolution, 96, 96, 5, bias=True, dtype=dtype, padding=2)
        
        self.d3 = nn.Sequential(
                        nn.ConvTranspose2d(in_channels, 96, 4, 2, 1),
                        nonlinear)
        d3_res = (e_h * 2, e_w * 2)

        self.cgru2 = ConvGRUCell(d3_res, 96, 96, 5, bias=True, dtype=dtype, padding=2)
        self.d2 = nn.Sequential(
                        nn.ConvTranspose2d(96, 96, 4, 2, 1),
                        nonlinear)
        d2_res = (e_h * 4, e_w * 4)
        
        self.cgru1 = ConvGRUCell(d2_res, 96, 64, 5, bias=True, dtype=dtype, padding=2)
        self.d1 = nn.Sequential(
                        nn.Conv2d(64, 16, 3, 1, 1),
                        nonlinear,
                        nn.Conv2d(16, opt.in_channels, 1, 1, 0),
                        nonlinear)

    def forward(self, hidden_states):
        
        e_h, e_w = self.encoder_resolution
        h_3_t, h_2_t, h_1_t = hidden_states[0], hidden_states[1], hidden_states[2]
        b = self.opt.batch_size

        inputs = torch.zeros((self.n_frames, b, self.in_channels, e_h, e_w)).to(self.device)

        h_prev = h_3_t
        new_inputs = []
        for input_ in inputs:   # iterating over timesteps to decode
            next_input = self.cgru3(input_, h_prev).to(self.device)
            new_inputs.append(next_input)
            h_prev = next_input
        
        new_inputs = torch.stack(new_inputs).to(self.device)
        inputs = new_inputs.view(b*self.n_frames, -1, e_h, e_w)
        inputs = self.d3(inputs).view(b, self.n_frames, -1, e_h*2, e_w*2).permute(1, 0, 2, 3, 4)

        h_prev = h_2_t
        new_inputs = []
        for input_ in inputs:   # iterating over timesteps to decode
            next_input = self.cgru2(input_, h_prev).to(self.device)
            new_inputs.append(next_input)
            h_prev = next_input

        new_inputs = torch.stack(new_inputs).to(self.device)
        inputs = new_inputs.view(b*self.n_frames, -1, e_h*2, e_w*2)
        inputs = self.d2(inputs).view(b, self.n_frames, -1, e_h*4, e_w*4).permute(1, 0, 2, 3, 4)

        h_prev = h_1_t
        new_inputs = []
        for input_ in inputs:   # iterating over timesteps to decode
            next_input = self.cgru1(input_, h_prev).to(self.device)
            new_inputs.append(next_input)
            h_prev = next_input

        new_inputs = torch.stack(new_inputs).to(self.device)
        inputs = new_inputs.view(b*self.n_frames, -1, e_h*4, e_w*4)
        decoded_state = self.d1(inputs).view(b, self.n_frames, -1, e_h*4, e_w*4)
        return decoded_state
