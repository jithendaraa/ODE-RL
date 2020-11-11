import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from helper import make_layers
import logging

class Encoder(nn.Module):
    def __init__(self, subnets, convgrus):
        super().__init__()
        self.subnet_blocks = len(subnets)  # A subnet can be nn.conv2d, nn.BatchNorm, or nn.ReLU
        self.convgru_blocks = len(convgrus)
        
        for index, params in enumerate(subnets, 1): # index sign from 1
            setattr(self, 'stage' + str(index), make_layers(params))
        
        for index, convgru in enumerate(convgrus, 1):   # index sign from 1
            setattr(self, 'convgru' + str(index), convgru)

    def forward_by_stage(self, inputs, subnet):
        seq_number, batch_size, input_channel, height, width = inputs.size()
        inputs = torch.reshape(inputs, (-1, input_channel, height, width))
        embedded_inputs = subnet(inputs) # Pass thru conv
        embedded_inputs = torch.reshape(embedded_inputs, (seq_number, batch_size, embedded_inputs.size(1),
                                        embedded_inputs.size(2), embedded_inputs.size(3)))
        return embedded_inputs

    def forward(self, inputs):
        inputs = inputs.transpose(0, 1)  # to input_frames,Batch,1,64,64 
        hidden_states = []
        logging.debug(inputs.size())

        for i in range(1, self.subnet_blocks + 1):
            inputs = self.forward_by_stage(inputs, getattr(self, 'stage' + str(i)))
        
        for i in range(1, self.convgru_blocks + 1):
            convgru = getattr(self, 'convgru' + str(i))
            outputs_stage, state_stage = convgru(inputs, None)
            hidden_states.append(state_stage)
            
        return hidden_states

class Encoder_ODEModel(nn.Module):
    def __init__(self, ode_specs, lr=1e-3):
        super(Encoder_ODEModel, self).__init__()
        print("Encoder ODE fθ initialised")
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
