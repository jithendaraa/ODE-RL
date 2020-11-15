import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchdiffeq import odeint_adjoint as odeint
import numpy as np
from helper import make_layers

class Decoder_ODEModel(nn.Module):
    def __init__(self, ode_specs, lr=1e-3):
        super().__init__()
        print("Decoder ODE fϕ initialised")
        self.conv1 = ode_specs[0]
        self.conv2 = ode_specs[1]
        self.conv3 = ode_specs[2]
        self.conv4 = ode_specs[3]
        self.optim = optim.Adamax(self.parameters(), lr=lr)

    def forward(self, t, htprev):
        htprev = F.tanh(self.conv1(htprev))
        htprev = F.tanh(self.conv2(htprev))
        htprev = F.tanh(self.conv3(htprev))
        h_t_ = self.conv4(htprev)
        return h_t_

class Decoder(nn.Module):
    def __init__(self, subnets, convgrus, ode_specs, predict_timesteps, lr=1e-3):
        super().__init__()
        self.subnet_blocks = len(subnets)  # A subnet can be nn.conv2d, nn.BatchNorm, or nn.ReLU
        self.convgru_blocks = len(convgrus)
        self.output_frames =  len(predict_timesteps)
        self.ode_model = Decoder_ODEModel(ode_specs)
        self.predict_timesteps = predict_timesteps

        for index, params in enumerate(subnets, 1): # index sign from 1
            setattr(self, 'stage' + str(index), make_layers(params))
        
        for index, convgru in enumerate(convgrus, 1):   # index sign from 1
            setattr(self, 'convgru' + str(index), convgru)

    def forward_by_stage(self, state, subnet):
        # state = [hs1, hs2....]
        seq_number, batch_size, input_channel, height, width = state.size()
        state = torch.reshape(state, (-1, input_channel, height, width))
        outputs = subnet(state)
        outputs = torch.reshape(outputs, (seq_number, batch_size, outputs.size(1),
                                        outputs.size(2), outputs.size(3)))
        return outputs

        
    def forward(self, h_s0):
        # h_s1, h_s2, ..... h_sK = ODESolve(fϕ, h_s0, [s1, s2, .... sK])
        outputs = odeint(self.ode_model, h_s0, torch.tensor(self.predict_timesteps))
        
        for i in list(range(1, self.subnet_blocks+1)):
            outputs = self.forward_by_stage(outputs, getattr(self, 'stage' + str(i)))

        return outputs
