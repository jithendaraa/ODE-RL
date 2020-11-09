import torch
import torch.nn as nn
from helper import make_layers
import logging

class Encoder(nn.Module):
    def __init__(self, subnets, rnns):
        super().__init__()
        # A subnet can be nn.conv2d, nn.BatchNorm, or nn.ReLU
        self.blocks = len(subnets)
        self.rnn_blocks = len(rnns)
        for index, params in enumerate(subnets, 1):
            # index sign from 1
            setattr(self, 'stage' + str(index), make_layers(params))
        
        for index, rnn in enumerate(rnns, 1):
            # index sign from 1
            setattr(self, 'rnn' + str(index), rnn)

    def forward_by_stage(self, inputs, subnet):
        seq_number, batch_size, input_channel, height, width = inputs.size()
        print(inputs.size(), subnet)
        inputs = torch.reshape(inputs, (-1, input_channel, height, width))
        embedded_inputs = subnet(inputs) # Pass thru conv
        embedded_inputs = torch.reshape(embedded_inputs, (seq_number, batch_size, embedded_inputs.size(1),
                                        embedded_inputs.size(2), embedded_inputs.size(3)))
        return embedded_inputs

    def forward(self, inputs):
        inputs = inputs.transpose(0, 1)  # to input_frames,Batch,1,64,64 
        logging.debug(inputs.size())
        for i in range(1, self.blocks + 1):
            print("\nBLOCK", i)
            inputs = self.forward_by_stage(
                inputs, 
                getattr(self, 'stage' + str(i))
            )
            print(inputs.size())
            print("__________________________________________________________________________")
        
        for i in range(1, self.rnn_blocks + 1):
            print("\nRNN_BLOCK", i)
            rnn = getattr(self, 'rnn' + str(i))
            outputs_stage, state_stage = rnn(inputs, None)
            print("__________________________________________________________________________")
        return tuple(inputs)