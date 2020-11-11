import torch
import torch.nn as nn

class ED(nn.Module):
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, input):
        hidden_states = self.encoder(input) # list with len = number of convgru blocks (self.encoder.convgru_blocks)
        h_T = hidden_states[-1] # aka h_s0 in figure 2 of the paper; size: batch_size * 64, h/4, w/4
        output = self.decoder(h_T)
        return output