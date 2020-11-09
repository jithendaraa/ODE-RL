import torch
import torch.nn as nn
from torchdiffeq import odeint

class ConvGRU(nn.Module):
    """
    ConvGRU Cell
    """
    def __init__(self, shape, input_channels, filter_size, num_features, print_details=False):
        super(ConvGRU, self).__init__()
        
        self.shape = shape
        self.input_channels = input_channels
        # kernel_size of input_to_state equals state_to_state
        self.filter_size = filter_size
        self.num_features = num_features
        self.padding = (filter_size - 1) // 2

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
        # seq_len=10 for moving_mnist
        if hidden_state is None:
            htprev = torch.zeros(inputs.size(1), self.num_features,
                                 self.shape[0], self.shape[1]).cuda()
        else:
            htprev = hidden_state
        
        print("h(t-1): ", htprev.size())
        # ht_ = ODESolve(f(theta), htprev, (tprev, t)) --> odeint(Encoder_ODE_model, htprev, (t-1, t))
        # Change all `htprev` in next lines as ht_ 
        output_inner = []
        for index in range(seq_len):
            if inputs is None:
                x = torch.zeros(htprev.size(0), self.input_channels,
                                self.shape[0], self.shape[1]).cuda()
            else:
                x = inputs[index, ...]

            combined_1 = torch.cat((x, htprev), 1)  # E(X_t) + H_t_
            gates = self.conv1(combined_1)  # W * (E(X_t) + H_t_)

            zgate, rgate = torch.split(gates, self.num_features, dim=1)
            z = torch.sigmoid(zgate)
            r = torch.sigmoid(rgate)

            combined_2 = torch.cat((x, r * htprev), 1)  # h' = tanh(W*(E(x)+r*H_t_))
            
            ht = self.conv2(combined_2)
            ht = torch.tanh(ht)

            htnext = (1 - z) * htprev + z * ht
            output_inner.append(htnext)
            htprev = htnext
        
        # print("____________________________________")
        # print(combined_1.size(), "COMBINEd before conv1 of ConvGRU")
        # print(gates.size(), "COMBINEd after conv1 of ConvGRU")
        # print(combined_2.size(), "COMBINEd before conv2 of ConvGRU")
        # print(ht.size(), "COMBINEd after conv2 of ConvGRU")
        # print("____________________________________")
        return torch.stack(output_inner), htnext