import torch 
from torch import nn

class EncoderDecoder(nn.Module):
    def __init__(self, args, device):
        super().__init__()
        self.args=args
        self.device = device
        self.seq_len = args.seq_len
        self.n_layers = args.n_layers
        self.encoder_params = [[1,16,3,1,1],[64,64,3,2,1],[96,96,3,2,1]]
        self.decoder_params = [[96,96,4,2,1],[96,96,4,2,1]]
        self.gru_n_features = [64,96,96]
        self.get_encoder_layers()
        self.get_decoder_layers()
        
    def get_encoder_layers(self):
        self.encoder_layers = []
        for i in range(self.n_layers):
            ((ic, oc, ke, s, p), numfea) = (self.encoder_params[i], self.gru_n_features[i]) 
            self.layer1 = nn.Sequential(nn.Conv2d(in_channels=ic,out_channels=oc,kernel_size=ke,stride=s,padding=p),
                                    nn.LeakyReLU(negative_slope=0.2, inplace=True)).to(self.device)
            self.layer2 = nn.Sequential(nn.Conv2d(in_channels=numfea+oc,out_channels=numfea*2,kernel_size=5,stride=1,padding=2),
                                    nn.GroupNorm((numfea*2)//32, (numfea*2))).to(self.device)
            self.layer3 = nn.Sequential(nn.Conv2d(in_channels=numfea+oc,out_channels=numfea,kernel_size=5,stride=1,padding=2),
                                    nn.GroupNorm(numfea//32, numfea)).to(self.device)
            self.encoder_layers.append([self.layer1,self.layer2,self.layer3])
        
    def get_decoder_layers(self):
        self.decoder_layers = []
        for i in range(self.n_layers-1):
            ((ic, oc, ke, s, p), numfea) = (self.decoder_params[i], self.gru_n_features[-1-i])
            self.layer1 = nn.Sequential(nn.Conv2d(in_channels=ic+numfea,out_channels=numfea*2,kernel_size=5,stride=1,padding=2),
                                    nn.GroupNorm(numfea*2//32, numfea*2)).to(self.device)
            self.layer2 = nn.Sequential(nn.Conv2d(in_channels=ic+numfea,out_channels=numfea,kernel_size=5,stride=1,padding=2),
                                    nn.GroupNorm(numfea//32, numfea)).to(self.device)                           
            self.layer3 = nn.Sequential(nn.ConvTranspose2d(in_channels=numfea,out_channels=oc,kernel_size=ke,stride=s,padding=p),
                                        nn.LeakyReLU(negative_slope=0.2, inplace=True)).to(self.device)
            self.decoder_layers.append([self.layer1,self.layer2,self.layer3])                            

        
        self.dcgru31 = nn.Sequential(nn.Conv2d(in_channels=oc+self.gru_n_features[0],out_channels=self.gru_n_features[0]*2,kernel_size=5,stride=1,padding=2),
                                   nn.GroupNorm(self.gru_n_features[0]*2//32, self.gru_n_features[0]*2)).to(self.device)
        self.dcgru32 = nn.Sequential(nn.Conv2d(in_channels=oc+self.gru_n_features[0],out_channels=self.gru_n_features[0],kernel_size=5,stride=1,padding=2),
                                   nn.GroupNorm(self.gru_n_features[0]//32, self.gru_n_features[0])).to(self.device)                                                                                  
        
        self.conv4 = nn.Sequential(nn.Conv2d(in_channels=64, out_channels=16,kernel_size=3, stride=1, padding=1),
                                   nn.Sigmoid(),
                                   nn.Conv2d(in_channels=16, out_channels=1,kernel_size=1, stride=1, padding=0),
                                   nn.Sigmoid()).to(self.device) 

        self.decoder_layers.append([self.dcgru31,self.dcgru32,self.conv4])                            

    def cgru(self, inputs, hidden_states, gru1, gru2, n_features):
        if hidden_states is None:
            htprev = torch.zeros(inputs.size(0), n_features, inputs.size(3), inputs.size(4)).cuda()
        else:
            htprev = hidden_states
        if inputs is None:
            inputs = torch.zeros(htprev.size(0), self.seq_len, htprev.size(1), htprev.size(2), htprev.size(3)).cuda()    

        output_inner = []
        for index in range(self.seq_len):
            x = inputs[:, index]
            combined_1 = torch.cat((x, htprev), 1) 
            gates = gru1(combined_1)
            ugate, rgate = torch.split(gates, n_features, dim=1)
            update_gate = torch.sigmoid(ugate)
            reset_gate = torch.sigmoid(rgate)
            combined_2 = torch.cat((x, reset_gate*htprev), 1)
            ht = gru2(combined_2)
            ht = torch.tanh(ht)
            htnext = (1 -update_gate) * htprev + update_gate * ht  
            output_inner.append(htnext)
            htprev = htnext
        
        outputs = torch.stack(output_inner)
        outputs = torch.reshape(outputs, (-1, outputs.size(2), outputs.size(3), outputs.size(4))) 
        return outputs, htnext

    def encoder(self, inputs):
        hidden_states = []
        batch_size, seq_number, _, _, _ = inputs.size()
        inputs = torch.reshape(inputs, (-1, inputs.size(2), inputs.size(3), inputs.size(4)))
        layers = self.encoder_layers
        
        for i in range(self.n_layers):
            inputs = layers[i][0](inputs)
            inputs = torch.reshape(inputs, (batch_size, seq_number, inputs.size(1), inputs.size(2), inputs.size(3)))
            inputs, hidden_state = self.cgru(inputs,None, layers[i][1], layers[i][2], self.gru_n_features[i])
            hidden_states.append(hidden_state)
        
        return hidden_states    

    def decoder(self, hidden_states):

        layers = self.decoder_layers
        inputs = None        
        for i in range(self.n_layers):
            outputs, _ = self.cgru(inputs, hidden_states[-1-i],  layers[i][0], layers[i][1], self.gru_n_features[-1-i])
            outputs = layers[i][2](outputs)
            outputs = torch.reshape(outputs, (self.args.batch_size, self.args.seq_len, outputs.size(1), outputs.size(2), outputs.size(3)))
            inputs = outputs
        
        return outputs

    def forward(self, inputs):
        states = self.encoder(inputs)
        outputs = self.decoder(states)        
        return outputs





