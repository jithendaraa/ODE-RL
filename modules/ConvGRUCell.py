import sys
sys.path.append('..')

import torch
import torch.nn as nn
import numpy as np

from helpers.utils import get_norm_layer

class ConvGRUCell(nn.Module):
    def __init__(self, input_size, input_dim, hidden_dim, kernel_size, bias, dtype=None, padding=None):
        """
        :param input_size: (int, int) / Height and width of input tensor as (height, width).
        :param input_dim: int / Number of channels of input tensor.
        :param hidden_dim: int / Number of channels of hidden state.
        :param kernel_size: (int, int) / Size of the convolutional kernel.
        :param bias: bool / Whether or not to add the bias.
        :param dtype: torch.cuda.FloatTensor or torch.FloatTensor / Whether or not to use cuda.
        """

        super(ConvGRUCell, self).__init__()
        self.height, self.width = input_size
        self.hidden_dim = hidden_dim
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if padding is None:
            self.padding = kernel_size[0] // 2, kernel_size[1] // 2
        else:
            self.padding = padding
        
        if dtype is None:
            dtype = torch.cuda.FloatTensor if self.device == 'cuda' else torch.FloatTensor
        
        self.bias = bias
        self.dtype = dtype
        
        # Outputs reset and update gate together; 1st hidden_dim channels are read 2nd hidden_dim dims are 
        self.conv_gates = nn.Conv2d(in_channels=input_dim + hidden_dim,
                                    out_channels=2 * self.hidden_dim,  # for update_gate,reset_gate respectively
                                    kernel_size=kernel_size,
                                    padding=self.padding,
                                    bias=self.bias)
        
        self.conv_can = nn.Conv2d(in_channels=input_dim + hidden_dim,
                                  out_channels=self.hidden_dim,  # for candidate neural memory
                                  kernel_size=kernel_size,
                                  padding=self.padding,
                                  bias=self.bias)
    
    def init_hidden(self, batch_size):
        return (torch.zeros(batch_size, self.hidden_dim, self.height, self.width)).type(self.dtype)
    
    def forward(self, input_tensor, h_cur, mask=None):
        """
        :param self:
        :param input_tensor: (b, input_dim, h, w) / input is actually the target_model
        :param h_cur: (b, hidden_dim, h, w) / current hidden and cell states respectively
        :return: h_next, next hidden state
        """
        combined = torch.cat([input_tensor, h_cur], dim=1) # (b, input_dim + hidden_dim, h, w)
        combined_conv = self.conv_gates(combined)   # (b, 2 * hidden_dim, h, w)

        reset_gate, update_gate = torch.split(combined_conv, self.hidden_dim, dim=1) # split into # (b, hidden_dim, h, w) and # (b, hidden_dim, h, w)
        reset_gate = torch.sigmoid(reset_gate)      # (b, hidden_dim, h, w)
        update_gate = torch.sigmoid(update_gate)    # (b, hidden_dim, h, w)

        combined = torch.cat([input_tensor, reset_gate * h_cur], dim=1) # (b, input_dim + hidden_dim, h, w)
        cand = torch.tanh(self.conv_can(combined)) # (b, hidden_dim, h, w)
        
        h_next = (1 - update_gate) * h_cur + update_gate * cand    # (b, hidden_dim, h, w)

        if mask is None:
            return h_next
        
        mask = mask.view(-1, 1, 1, 1).expand_as(h_cur)
        h_next = mask * h_next + (1 - mask) * h_cur # (b, hidden_dim, h, w)
        
        return h_next   # (b, hidden_dim, h, w)

class ConvEncoder(nn.Module):
    
    def __init__(self, in_channels=3, output_channels=128, n_units=128, n_layers=1, activation='relu', kernel=3, stride=1, padding=1, device=None):
        
        super(ConvEncoder, self).__init__()
        if activation == 'tanh':
            activation = nn.Tanh()
        elif activation == 'relu':
            activation = nn.ReLU()
        else:
            raise NotImplementedError('There is no named')

        layers = []
        layers += [nn.Conv2d(in_channels, n_units, kernel, stride, padding, dilation=1)]
        layers += [get_norm_layer(n_units)] # Batch norm
        layers += [activation]

        for i in range(n_layers):
            # (Halves the resolution with each iter) 
            # (h,w) -> (h/2, w/2)
            layers += [nn.Conv2d(n_units, n_units*2, 4, 2, 1, dilation=1)]
            layers += [get_norm_layer(n_units*2)] # Batch norm
            layers += [activation]
            n_units *= 2
            
        layers.append(activation)
        layers.append(nn.Conv2d(n_units, output_channels, kernel, stride, padding, dilation=1))

        self.layers = nn.Sequential(*layers).to(device)
        self.to(device)

    def forward(self, x):
        return self.layers(x)

class ConvDecoder(nn.Module):
    
    def __init__(self):
        super(ConvDecoder, self).__init__()
    
    def forward(self, x):
        pass