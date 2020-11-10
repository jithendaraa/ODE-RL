import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from helper import make_layers

class Decoder_ODEModel(nn.Module):
    def __init__(self, lr=1e-3):
        super(Decoder_ODEModel, self).__init__()
        print("Decoder ODE fθ initialised")
        self.conv1 = nn.Conv2d(64, 64, 3, 1, 1)
        self.conv2 = nn.Conv2d(64, 64, 3, 1, 1)
        self.conv3 = nn.Conv2d(64, 64, 3, 1, 1)
        self.conv4 = nn.Conv2d(64, 64, 3, 1, 1)

        self.optimizer = optim.Adamax(self.parameters(), lr=lr)
    
    def forward(self, t, htprev):
        htprev = F.tanh(self.conv1(htprev))
        htprev = F.tanh(self.conv2(htprev))
        htprev = F.tanh(self.conv3(htprev))
        h_t_ = self.conv4(htprev)
        return h_t_