import torch
import torch.nn as nn
from helper import make_layers
import logging

class Encoder(nn.Module):
    def __init__(self, subnets, rnns):
        super().__init__()
        print(len(subnets))
        assert len(subnets) == len(rnns)
        self.blocks = len(subnets)

        for index, (params, rnn) in enumerate(zip(subnets, rnns), 1):
            # index sign from 1
            setattr(self, 'stage' + str(index), make_layers(params))
            setattr(self, 'rnn' + str(index), rnn)

    def forward_by_stage(self, inputs, subnet, rnn):
        seq_number, batch_size, input_channel, height, width = inputs.size()
        inputs = torch.reshape(inputs, (-1, input_channel, height, width))
        # print("X Before Conv", inputs.size())
        inputs = subnet(inputs) # Pass thru conv
        # print("X After Conv", inputs.size())
        inputs = torch.reshape(inputs, (seq_number, batch_size, inputs.size(1),
                                        inputs.size(2), inputs.size(3)))
        outputs_stage, state_stage = rnn(inputs, None)
        return outputs_stage, state_stage

    def forward(self, inputs):
        inputs = inputs.transpose(0, 1)  # to input_frames,Batch,1,64,64 
        hidden_states = []
        logging.debug(inputs.size())
        for i in range(1, self.blocks + 1):
            # print("BLOCK", i)
            inputs, state_stage = self.forward_by_stage(
                inputs, 
                getattr(self, 'stage' + str(i)),
                getattr(self, 'rnn' + str(i)))
            hidden_states.append(state_stage)
        return tuple(hidden_states)


    
class Decoder(nn.Module):
    def __init__(self, subnets, rnns):
        super().__init__()
        assert len(subnets) == len(rnns)
        self.blocks = len(subnets)

        for index, (params, rnn) in enumerate(zip(subnets, rnns)):
            setattr(self, 'rnn' + str(self.blocks - index), rnn)
            setattr(self, 'stage' + str(self.blocks - index),
                    make_layers(params))

    def forward_by_stage(self, inputs, state, subnet, rnn):
        # print("state before ConvGRU", state.size())
        inputs, state_stage = rnn(inputs, state, seq_len=10)
        # print("input after ConvGRU", inputs.size())
        # print("state after ConvGRU", state_stage.size())
        seq_number, batch_size, input_channel, height, width = inputs.size()
        inputs = torch.reshape(inputs, (-1, input_channel, height, width))
        # print("Inputs before Conv", inputs.size())
        inputs = subnet(inputs)
        # print("Inputs after Conv", inputs.size())
        inputs = torch.reshape(inputs, (seq_number, batch_size, inputs.size(1),
                                        inputs.size(2), inputs.size(3)))
        return inputs

    def forward(self, hidden_states):
        # print("BLOCK 2")
        
        inputs = self.forward_by_stage(None, hidden_states[-1],
                                       getattr(self, 'stage3'),
                                       getattr(self, 'rnn3'))
    
        # print("INPUT SIZE: ", inputs.size())
        for i in list(range(1, self.blocks))[::-1]:
            # print("BLOCK", i-1, " hidden state: ", hidden_states[i - 1].size())
            inputs = self.forward_by_stage(inputs, hidden_states[i - 1],
                                           getattr(self, 'stage' + str(i)),
                                           getattr(self, 'rnn' + str(i)))
        inputs = inputs.transpose(0, 1)  # to B,S,1,64,64
        return inputs

class ED(nn.Module):
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, input):
        state = self.encoder(input)
        # print("ENCODING DONE......")
        output = self.decoder(state)
        # print("DECODING DONE......")
        return output