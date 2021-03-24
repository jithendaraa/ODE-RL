import torch
import torch.nn as nn
import torch.optim as optim
from torchdiffeq import odeint_adjoint as odeint
from encoder import Encoder_ODEModel
from decoder import Decoder_ODEModel

class ConvGRU(nn.Module):
    """
    ConvGRU Cell
    """
    def __init__(self, shape, input_channels, filter_size, num_features, ode_specs, feed='encoder', print_details=False, lr=1e-3, device=None):
        super(ConvGRU, self).__init__()
        self.device = device
        self.shape = shape
        self.input_channels = input_channels
        self.filter_size = filter_size
        self.num_features = num_features
        self.padding = (filter_size - 1) // 2
        self.feed = feed
        
        if feed == 'encoder':
            self.ode_model = Encoder_ODEModel(ode_specs, device=self.device).to(self.device)
        elif feed == 'decoder':
            self.ode_model = Decoder_ODEModel(ode_specs, device=self.device).to(self.device)

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels = self.input_channels + self.num_features,
                      out_channels = 2 * self.num_features, 
                      kernel_size = self.filter_size, 
                      stride = 1,
                      padding = self.padding),
            nn.GroupNorm(2 * self.num_features // 32, 
                        2 * self.num_features)
            )
            
        self.conv2 = nn.Sequential(
            nn.Conv2d(self.input_channels + self.num_features,
                      self.num_features, 
                      self.filter_size, 
                      1, 
                      self.padding),
            nn.GroupNorm(self.num_features // 32, self.num_features))
        
    def forward(self, inputs=None, hidden_state=None, seq_len=10):
        # inputs size: S, B, C, H, W
        if hidden_state is None:
            htprev = torch.zeros(inputs.size(1), self.num_features, self.shape[0], self.shape[1]).to(self.device)
        else:
            htprev = hidden_state
        
        output_inner = []
        htnext = None
        for index in range(seq_len):
            if inputs is None:
                x = torch.zeros(htprev.size(0), self.input_channels,
                                self.shape[0], self.shape[1]).to(self.device)
            else:
                x = inputs[index, ...].to(self.device)
            
            # h_t_ = ODESolve(f(theta), htprev, (index, index+1)) --> odeint(Encoder_ODE_model, htprev, (t-1, t))
            h_t_ = odeint(self.ode_model, htprev, torch.tensor([float(index), float(index+1.0)])).to(self.device)[1]
            combined_1 = torch.cat((x, h_t_), 1)  # E(X_t) + H_t_
            gates = self.conv1(combined_1)  # W * (E(X_t) + H_t_)
            zgate, rgate = torch.split(gates, self.num_features, dim=1)
            z = torch.sigmoid(zgate)
            r = torch.sigmoid(rgate)
            combined_2 = torch.cat((x, r * h_t_), 1)  # h' = tanh(W*(E(x)+r*H_t_))
            ht = self.conv2(combined_2)
            ht = torch.tanh(ht)
            htnext = (1 - z) * htprev + z * ht
            output_inner.append(htnext)
            htprev = htnext
        return torch.stack(output_inner), htnext
