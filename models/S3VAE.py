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


class DFP(nn.Module):
    def __init__(self, z_size=128):
        super().__init__()
        self.main_net = nn.Sequential(
            nn.Linear(in_features=z_size, out_features=z_size),
            nn.Linear(in_features=z_size, out_features=z_size)
        )
        self.mean = nn.Linear(in_features=z_size, out_features=3)
        self.std = nn.Linear(in_features=z_size, out_features=3)


    def forward(self, batch):
        shape, feature_shape = batch.shape[:-1], batch.shape[-1]
        batch = batch.reshape(-1, feature_shape)
        features = self.main_net(batch)
        feature_shape = features.shape[1:]
        features = features.reshape(*shape, *feature_shape)
        # features = self.main_net(z_t)
        mean = self.mean(features)
        std = F.softplus(self.std(features))
        return dist.Normal(loc=mean, scale=std)

class S3VAE(nn.Module):
    def __init__(self, opt, device):
        super(S3VAE, self).__init__()

        self.opt = opt
        self.device = device
        self.log_scale = nn.Parameter(torch.Tensor([0.0])).to(device)
        in_ch = opt.in_channels
        d_zf, d_zt = opt.d_zf, opt.d_zt

        self.conv_encoder = Encoder(in_ch).to(device)
        self.static_rnn = LSTMEncoder(128, 256, d_zf, static=True).to(device)
        self.dynamic_rnn = LSTMEncoder(128, 256, d_zt, static=False).to(device)
        self.prior_rnn = LSTMEncoder(d_zt*2, 256, d_zt, static=False).to(device)
        self.conv_decoder = Decoder(d_zf + d_zt, in_ch).to(device)
        # For SCC
        self._triplet_loss = nn.TripletMarginLoss(margin=opt.m)
        # Dynamic Factor Prediction
        # self.dfp_net = DFP(z_size=d_zt)

    
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
        # print("Inputs:", inputs.size())

        encoded_inputs = self.conv_encoder(inputs.view(b*t, c, h, w))
        num_features = encoded_inputs.size()[1]

        # Get mu and std of static latent variable zf of dim d_zf
        mu_zf, std_zf = self.static_rnn(encoded_inputs.view(b, t, num_features))

        # Get posterior mu and std of dynamic latent variables z1....zt each of dim d_zt
        mu_zt, std_zt = self.dynamic_rnn(encoded_inputs.view(b, t, num_features))
        mu_std_zt = torch.cat((mu_zt, std_zt), dim=2)
        
        # Get prior mu and std of dynamic latent variables z1....zt each of dim d_zt
        prior_mu_zt, prior_std_zt = self.prior_rnn(mu_std_zt)
        
        # p(z_f) prior -> N(0, 1) and q(z_f | x_1:T) posterior
        self.p_zf = dist.Normal(loc=torch.zeros_like(mu_zf).cuda(), scale=torch.ones_like(std_zf).cuda())
        self.q_zf_xT = dist.Normal(loc=mu_zf, scale=std_zf)

        # p(z_t | z<t) prior and q(z_t | x <= T) posterior
        self.p_zt = dist.Normal(loc=prior_mu_zt, scale=prior_std_zt)
        self.q_zt_xt = dist.Normal(loc=mu_zt, scale=std_zt)

        zf_sample = self.q_zf_xT.rsample()
        zt_sample = self.q_zt_xt.rsample()
        zf_zt = torch.cat((zf_sample.view(b, 1, -1).repeat(1, t, 1), zt_sample), dim=2).view(b*t, -1, 1, 1)

        x_hat = self.conv_decoder(zf_zt).view(b, t, c, h, w)

        # 1. VAE ELBO Loss
        self.get_vae_loss(x_hat, inputs, zf_sample, zt_sample)

        # 2. SCC Loss
        # shuffle batch to get zf_pos
        shuffle_idx = torch.randperm(encoded_inputs.shape[1])
        shuffled_encoded_inputs = encoded_inputs[:, shuffle_idx].contiguous()
        zf_pos_mu, zf_pos_std = self.static_rnn(shuffled_encoded_inputs.view(b, t, num_features))
        zf_pos = dist.Normal(loc=zf_pos_mu, scale=zf_pos_std)
        # Get zf from another sequence for zf_neg
        another_encoded_tensor = self.conv_encoder(other.view(b*t, c, h, w))
        zf_neg_mu, zf_neg_std = self.static_rnn(another_encoded_tensor.view(b, t, num_features))
        zf_neg = dist.Normal(loc=zf_neg_mu, scale=zf_neg_std)
        self.get_scc_loss(zf_pos, zf_neg)

        # 3. TODO: DFP Loss

        # 4. MI Loss
        self.get_mi_loss()
        
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
        self.vae_loss = (- recon_loss + kl_loss).mean()

    def get_scc_loss(self, zf_pos, zf_neg):
        # zf equivalent to self.q_zf_xT -- time-invariant representation from real data
        # zf_pos -- time-invariant representation from shuffled real video
        # zf_neg -- time-invariant representation from another video
        zf_sample = self.q_zf_xT.rsample()
        zf_pos_sample = zf_pos.sample()
        zf_neg_sample = zf_neg.sample()
        # max(D(zf, zf_pos) - D(zf, zf_neg) + margin, 0)
        self.scc_loss = self._triplet_loss(zf_sample, zf_pos_sample, zf_neg_sample)

    def get_dfp_loss(self):
        pass

    def get_mi_loss(self):
        # sum t from 1..T H(zf) + H(zt) - H(zf, zt)
        # zf_dist is self.q_zf_xT and zt_dist = self.q_zt_xt
        # H(.) -> -log q(.)
        def dist_op(dist1, op, t=False):
            if t is True:
                return dist.Normal(loc=op(dist1.loc.permute(1, 0, 2)), scale=op(dist1.scale.permute(1, 0, 2)))

            else:
                return dist.Normal(loc=op(dist1.loc), scale=op(dist1.scale))


        z_t1 = dist_op(self.q_zt_xt, lambda x: x.unsqueeze(1), t=True) # t, 1, b, d_zt
        z_t2 = dist_op(self.q_zt_xt, lambda x: x.unsqueeze(2), t=True) # t, b, 1, d_zt
        log_q_t = z_t1.log_prob(z_t2.rsample()).sum(-1)        # t, b, b   
        H_t = log_q_t.logsumexp(2).mean(1) - np.log(log_q_t.shape[2]) # t
        
        z_f1 = dist_op(self.q_zf_xT, lambda x: x.unsqueeze(0)) # 1, b, d_zf
        z_f2 = dist_op(self.q_zf_xT, lambda x: x.unsqueeze(1)) # b, 1, d_zf
        log_q_f = z_f1.log_prob(z_f2.rsample()).sum(-1)        # b, b 
        H_f = log_q_f.logsumexp(1).mean(0) - np.log(log_q_t.shape[2])  

        H_ft = (log_q_f.unsqueeze(0) + log_q_t).logsumexp(1).mean(1) # t
        self.mi_loss = -(H_f + H_t.mean() - H_ft.mean())

    def get_loss(self):
        
        # TODO: add self.opt.l2 * self.dfp_loss after adding DFP and uncomment DFP Loss in loss_dict below
        loss = self.vae_loss + (self.opt.l1 * self.scc_loss) + (self.opt.l3 * self.mi_loss)

        loss_dict = {
            'Loss': loss.item(),
            'VAE Loss': self.vae_loss.item(),
            'SCC Loss': self.scc_loss.item(),
            # 'DFP Loss': self.dfp_loss.item(),
            'MI Loss': self.mi_loss.item()
        }
        return loss, loss_dict

