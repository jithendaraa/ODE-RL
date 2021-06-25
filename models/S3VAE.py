import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import distributions as dist

import numpy as np

class Encoder(nn.Module):
    def __init__(self, in_ch):
        super(Encoder, self).__init__()

        self.layers = nn.Sequential(
            nn.Conv2d(in_ch, 64, 4, 2, 1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            nn.Conv2d(128, 256, 4, 2, 1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),
            nn.Conv2d(256, 512, 4, 2, 1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2),
            nn.Conv2d(512, 128, 4, 1, 0),
            nn.BatchNorm2d(128),
            nn.Tanh())

    def forward(self, inputs):
        return self.layers(inputs)

class LSTMEncoder(nn.Module):
    def __init__(self, input_size=128, hidden_size=256, z_size=256, static=False):
        super().__init__()

        self.lstm_net = nn.LSTM(input_size, hidden_size, num_layers=1, batch_first=True)
        self.mean_net = nn.Linear(hidden_size * (1 + int(static)), z_size)
        self.std_net = nn.Linear(hidden_size * (1 + int(static)), z_size)
        self.static = static
    
    def forward(self, inputs):
        outs, hidden = self.lstm_net(inputs)
        if self.static:
            hidden = torch.cat(hidden, 2).squeeze(0)
            mean = self.mean_net(hidden)
            std = F.softplus(self.std_net(hidden))
        else:
            mean = self.mean_net(outs)
            std = F.softplus(self.std_net(outs))

        return mean, std

class Decoder(nn.Module):
    def __init__(self, in_ch, final_dim):
        super(Decoder, self).__init__()
        self.layers = nn.Sequential(
            nn.ConvTranspose2d(in_ch, 512, 4, 1, 0),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(512, 256, 3, 1, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(256, 128, 3, 1, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 128, 3, 1, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 64, 3, 1, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, final_dim, 1, 1, 0),
            nn.Sigmoid()
        )
    
    def forward(self, inputs):
        return self.layers(inputs)

class S3VAE(nn.Module):
    def __init__(self, opt, device):
        super(S3VAE, self).__init__()

        self.opt = opt
        self.device = device
        in_ch = opt.in_channels
        T = opt.train_in_seq
        d_zf, d_zt = opt.d_zf, opt.d_zt

        self.conv_encoder = Encoder(in_ch).to(device)
        self.static_rnn = LSTMEncoder(128, 256, d_zf, static=True).to(device)
        self.dynamic_rnn = LSTMEncoder(128, 256, d_zt, static=False).to(device)
        self.prior_rnn = LSTMEncoder(d_zt*2, 256, d_zt, static=False).to(device)
        self.conv_decoder = Decoder(d_zf + d_zt, in_ch).to(device)
        self.log_scale = nn.Parameter(torch.Tensor([0.0])).to(device)
    
    def set_zero_losses(self):
        self.vae_loss = 0
        self.scc_loss = 0
        self.dfp_loss = 0
        self.mi_loss = 0

    def forward(self, inputs):
        self.set_zero_losses()

        # Shuffled inputs to generate zf_neg
        other = inputs[torch.from_numpy(np.random.permutation(len(inputs)))].to(self.device)

        b, t, c, h, w = inputs.size()
        print()
        print("Inputs:", inputs.size())

        encoded_inputs = self.conv_encoder(inputs.view(b*t, c, h, w))
        num_features = encoded_inputs.size()[1]
        # print("encoded_inputs", encoded_inputs.size())

        # Get mu and std of static latent variable zf of dim d_zf
        mu_zf, std_zf = self.static_rnn(encoded_inputs.view(b, t, num_features))
        # print("zf gaussian params: ", mu_zf.size(), std_zf.size())

        # Get posterior mu and std of dynamic latent variables z1....zt each of dim d_zt
        mu_zt, std_zt = self.dynamic_rnn(encoded_inputs.view(b, t, num_features))
        mu_std_zt = torch.cat((mu_zt, std_zt), dim=2)
        
        # Get prior mu and std of dynamic latent variables z1....zt each of dim d_zt
        prior_mu_zt, prior_std_zt = self.prior_rnn(mu_std_zt)
        # print("posterior zt gaussian params: ", mu_zt.size(), std_zt.size())
        # print("prior zt gaussian params: ", prior_mu_zt.size(), prior_std_zt.size())
        
        # p(z_f) prior -> N(0, 1) and q(z_f | x_1:T) posterior
        self.p_zf = dist.Normal(loc=torch.zeros_like(mu_zf).cuda(), scale=torch.ones_like(std_zf).cuda())
        self.q_zf_xT = dist.Normal(loc=mu_zf, scale=std_zf)

        # p(z_t | z<t) prior and q(z_t | x <= T) posterior
        self.p_zt = dist.Normal(loc=prior_mu_zt, scale=prior_std_zt)
        self.q_zt_xt = dist.Normal(loc=mu_zt, scale=std_zt)

        zf_sample = self.q_zf_xT.rsample()
        zt_sample = self.q_zt_xt.rsample()
        # print("Sampled zf and zt", zf_sample.size(), zt_sample.size()) 
        
        zf_zt = torch.cat((zf_sample.view(b, 1, -1).repeat(1, t, 1), zt_sample), dim=2).view(b*t, -1, 1, 1)
        # print("zf_zt", zf_zt.size())

        x_hat = self.conv_decoder(zf_zt).view(b, t, c, h, w)
        
        self.get_vae_loss(x_hat, inputs, zf_sample, zt_sample)
        
        return x_hat

    def get_prediction(self, inputs, batch_dict=None):
        pred_frames = self(inputs)
        return pred_frames
    
    def kl_divergence(self, p, q, z):
        log_qzx = q.log_prob(z)
        log_pz = p.log_prob(z)
        kl = (log_qzx - log_pz)
        kl = kl.sum(-1)
        return kl

    def get_vae_loss(self, x_hat, x, zf, zt):
        
        # 1. Reconstruction loss: p(xt | zf, zt)
        scale = torch.exp(self.log_scale)
        log_pxz = dist.Normal(x_hat, scale).log_prob(x)
        recon_loss = log_pxz.sum(dim=(1,2,3,4))   # Sum over t = 1..T log p(xt| zf, zt)  {dim 1 sums over time, dims 2,3,4 sum over c,h,w}

        # 2. KL for static latent variable zf
        zf_KL_div_loss = self.kl_divergence(self.p_zf, self.q_zf_xT, zf)
        
        # 3. KL for static latent variable zt
        zt_KL_div_loss = self.kl_divergence(self.p_zt, self.q_zt_xt, zt).sum(dim=(1))

        kl_loss = zf_KL_div_loss + zt_KL_div_loss
        
        # ELBO Loss
        self.vae_loss = (kl_loss - recon_loss).mean()
        print("VAE Loss:", self.vae_loss)

    def get_scc_loss(self):
        pass

    def get_dfp_loss(self):
        pass

    def get_MI_loss(self):
        pass

    def get_loss(self, preds, gt, loss=None):
        loss = self.vae_loss
        return loss 

