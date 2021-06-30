import torch 
from torch import nn

class EncoderDecoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.seq_len = 10
        self.conv1 = nn.Sequential(nn.Conv2d(in_channels=1,out_channels=16,kernel_size=3,stride=1,padding=1),
                                   nn.LeakyReLU(negative_slope=0.2, inplace=True),
                                   nn.Conv2d(in_channels=16,out_channels=32,kernel_size=3,stride=2,padding=1),
                                   nn.LeakyReLU(negative_slope=0.2, inplace=True),
                                   nn.Conv2d(in_channels=32,out_channels=64,kernel_size=3,stride=2,padding=1),
                                   nn.LeakyReLU(negative_slope=0.2, inplace=True))
        
        self.conv2 = nn.Sequential(nn.Conv2d(in_channels=128,out_channels=128,kernel_size=5,stride=1,padding=2),
                                   nn.GroupNorm(128 // 32, 128))
        self.conv3 = nn.Sequential(nn.Conv2d(in_channels=128,out_channels=64,kernel_size=5,stride=1,padding=2),
                                   nn.GroupNorm(64 // 32, 64))

        self.conv4 = nn.Sequential(nn.ConvTranspose2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
                                   nn.LeakyReLU(negative_slope=0.2, inplace=True))                           

        self.conv5 = nn.Sequential(nn.ConvTranspose2d(in_channels=64, out_channels=64, kernel_size=4, stride=2, padding=1),
                                   nn.LeakyReLU(negative_slope=0.2, inplace=True),
                                   nn.ConvTranspose2d(in_channels=64, out_channels=64, kernel_size=4, stride=2, padding=1),
                                   nn.LeakyReLU(negative_slope=0.2, inplace=True),
                                   nn.Conv2d(in_channels=64, out_channels=16,kernel_size=3, stride=1, padding=1),
                                   nn.Sigmoid(),
                                   nn.Conv2d(in_channels=16, out_channels=1,kernel_size=1, stride=1, padding=0),
                                   nn.Sigmoid())  

    def cgru(self, inputs, hidden_states):
        if hidden_states is None:
            htprev = torch.zeros(inputs.size(0), inputs.size(2), inputs.size(3), inputs.size(4)).cuda()
        else:
            htprev = hidden_states
       
        output_inner = []
        for index in range(self.seq_len):
            x = inputs[:, index]

            combined_1 = torch.cat((x, htprev), 1) 
            gates = self.conv2(combined_1)
            zgate, rgate = torch.split(gates, 64, dim=1)
            z = torch.sigmoid(zgate)
            r = torch.sigmoid(rgate)
            combined_2 = torch.cat((x, r*htprev), 1)
            ht = self.conv3(combined_2)
            ht = torch.tanh(ht)
            htnext = (1 -z) * htprev + z * ht  
            output_inner.append(htnext)
            htprev = htnext
        
        return torch.stack(output_inner), htnext

    def forward(self, inputs):
        batch_size, seq_number, input_channel, height, width = inputs.size()
        inputs = torch.reshape(inputs, (-1, input_channel, height, width))
        inputs = self.conv1(inputs)
        inputs = torch.reshape(inputs, (batch_size, seq_number, inputs.size(1), inputs.size(2), inputs.size(3)))
        outputs, hidden_state = self.cgru(inputs,None)
        
        outputs = torch.reshape(outputs, (-1, outputs.size(2), outputs.size(3), outputs.size(4)))
        outputs = self.conv4(outputs)
        outputs = torch.reshape(outputs, (batch_size, seq_number, outputs.size(1), outputs.size(2), outputs.size(3)))

        outputs, hidden_state = self.cgru(outputs, hidden_state)
        seq_number, batch_size, input_channel, height, width = outputs.size()
        outputs = torch.reshape(outputs, (-1, input_channel, height, width))
        outputs = self.conv5(outputs)
        outputs = torch.reshape(outputs, (batch_size, seq_number, outputs.size(1), outputs.size(2), outputs.size(3)))
        return outputs





