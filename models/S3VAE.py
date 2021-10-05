import sys
sys.path.append('..')

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import distributions as dist
import numpy as np
import wandb
import math

from modules.S3VAE_ED import Encoder, GRUEncoder, ConvGRUEncoder, Decoder, DFP

from modules.SlotAttention import SlotAttentionAutoEncoder
from helpers.utils import *

class S3VAE(nn.Module):
    def __init__(self, opt, device):
        super(S3VAE, self).__init__()
        self.opt = opt

        if opt.phase == 'train':
            self.in_seq = opt.train_in_seq
            self.out_seq = opt.train_in_seq
        
        elif opt.phase == 'test':
            self.in_seq = opt.test_in_seq
            self.out_seq = opt.test_in_seq + opt.test_out_seq

        self.h, self.w = opt.resolution, opt.resolution
        self.device = device
        self.log_scale = nn.Parameter(torch.Tensor([0.0])).to(device)
        in_ch = opt.in_channels
        self.in_ch = in_ch
        d_zf, d_zt = opt.d_zf, opt.d_zt
        n_hid = 512
        self.latent_dims = {}
        encoder_out = self.opt.encoder_out_dims

        if self.opt.rim is True: 
            n_hid=opt.n_hid[0]
            self.num_rims = self.opt.n_hid[0] // self.opt.unit_per_rim
        else:
            self.num_rims = 1

        # Only relevant for slot attention
        broadcast = False
        if opt.encoder in ['cgru_sa']:  broadcast = True
        
        # For SCC Loss
        self._triplet_loss = nn.TripletMarginLoss(margin=opt.m)
        self._triplet_loss_sc = nn.TripletMarginLoss(margin=opt.m)
        
        # For DFP Loss
        self.dfp_net = DFP(opt, z_size=d_zt).to(device)

        # Encoder, dynamics and Decoder networks
        self.conv_encoder = Encoder(in_ch, opt.encoder, opt.encoder_out_dims).to(device)
        resize = self.conv_encoder.resize
        self.res_after_encoder = opt.resolution // resize
        
        if opt.encoder == 'default':
            self.static_rnn = GRUEncoder(encoder_out, n_hid, d_zf, ode=False, device=device, type='static', batch_first=True, opt=opt).to(device)
            self.dynamic_rnn = GRUEncoder(encoder_out, n_hid, d_zt, ode=opt.ode, device=device, type='dynamic', batch_first=True, opt=opt).to(device)
            self.prior_rnn = GRUEncoder(d_zt * 2 * self.num_rims, n_hid, d_zt * self.num_rims, ode=False, device=device, type='prior', batch_first=True, opt=opt).to(device)
        
        elif opt.encoder in ['odecgru', 'cgru', 'cgru_sa']:
            self.static_rnn = ConvGRUEncoder(encoder_out, d_zf, opt, device, self.conv_encoder.resize, type='static').to(device)
            self.dynamic_rnn = ConvGRUEncoder(encoder_out, d_zt, opt, device, self.conv_encoder.resize, type='dynamic').to(device)
            self.prior_rnn = ConvGRUEncoder(d_zt*2, d_zt, opt, device, self.conv_encoder.resize,type='prior').to(device)

        if opt.slot_att is True:
            self.mu_slot_att = SlotAttentionAutoEncoder(opt, resolution=(self.h // resize, self.w // (resize)), num_slots=opt.num_slots, num_iterations=opt.num_iterations, device=device, resize=resize, broadcast=broadcast).to(device)
            self.logvar_slot_att = SlotAttentionAutoEncoder(opt, resolution=(self.h // resize, self.w // (resize)), num_slots=opt.num_slots, num_iterations=opt.num_iterations, device=device, resize=resize, broadcast=broadcast).to(device)
            
            if opt.unmasked is False: ch_ = in_ch + 1
            else: ch_ = in_ch
            self.conv_decoder = Decoder((opt.num_slots * self.opt.slot_size) + (d_zt * self.num_rims), ch_, opt).to(device)

        else:
            self.conv_decoder = Decoder(d_zf + d_zt, in_ch, opt).to(device)
            
    def set_zero_losses(self):
        self.vae_loss = 0
        self.scc_loss = 0
        self.dfp_loss = 0
        self.mi_loss = 0

    def get_static_rep(self, encoded_inputs, shuffled_encoded_inputs, another_encoded_tensor):
        b, t = self.opt.batch_size, self.in_seq

        if self.opt.encoder in ['odecgru', 'cgru', 'cgru_sa']:
            _, c_, h_, w_ = encoded_inputs.size()
            encoded_inputs = encoded_inputs.view(b, t, c_, h_, w_).permute(1, 0, 2, 3, 4)
            shuffled_encoded_inputs = shuffled_encoded_inputs.view(b, t, c_, h_, w_).permute(1, 0, 2, 3, 4)
            another_encoded_tensor = another_encoded_tensor.view(b, t, c_, h_, w_).permute(1, 0, 2, 3, 4)
            
            if self.opt.k_stat != -1:   
                t = self.opt.k_stat
                in_anch, in_pos, in_neg = encoded_inputs[:self.opt.k_stat, :], shuffled_encoded_inputs[:self.opt.k_stat, :], another_encoded_tensor[:self.opt.k_stat, :]

            else:
                in_anch, in_pos, in_neg = encoded_inputs, shuffled_encoded_inputs, another_encoded_tensor

            # Get posterior mu and logvar of static latent variable zf of channels dim d_zf
            mu_zf, logvar_zf = self.static_rnn(in_anch, t)
            zf_pos_mu, zf_pos_logvar = self.static_rnn(in_pos, t)
            zf_neg_mu, zf_neg_logvar = self.static_rnn(in_neg, t)

            # Pass the static representations through slots and get object-centric representations
            if self.opt.slot_att is True and self.opt.encoder in ['cgru_sa']:
                mu_zf, logvar_zf = self.mu_slot_att(mu_zf), self.logvar_slot_att(logvar_zf)
                b_s, c, h, w = mu_zf.size()
                zf_pos_mu, zf_pos_logvar = self.mu_slot_att(zf_pos_mu), self.logvar_slot_att(zf_pos_logvar)
                zf_neg_mu, zf_neg_logvar = self.mu_slot_att(zf_neg_mu), self.logvar_slot_att(zf_neg_logvar)
                
                mu_zf, logvar_zf = mu_zf.view(b, self.opt.num_slots, c, h, w).contiguous().view(b, -1, h, w), logvar_zf.view(b, self.opt.num_slots, c, h, w).contiguous().view(b, -1, h, w)
                zf_pos_mu, zf_pos_logvar = zf_pos_mu.view(b, self.opt.num_slots, c, h, w).contiguous().view(b, -1, h, w), zf_pos_logvar.view(b, self.opt.num_slots, c, h, w).contiguous().view(b, -1, h, w)
                zf_neg_mu, zf_neg_logvar = zf_neg_mu.view(b, self.opt.num_slots, c, h, w).contiguous().view(b, -1, h, w), zf_neg_logvar.view(b, self.opt.num_slots, c, h, w).contiguous().view(b, -1, h, w)


        elif self.opt.encoder in ['default']:
            num_features = encoded_inputs.size()[1]
            if self.opt.k_stat == -1:
                in_anch, in_pos, in_neg = encoded_inputs.view(b, t, num_features), shuffled_encoded_inputs.view(b, t, num_features), another_encoded_tensor.view(b, t, num_features)
            else:
                in_anch, in_pos, in_neg = encoded_inputs.view(b, t, num_features)[:, :self.opt.k_stat, :], shuffled_encoded_inputs.view(b, t, num_features)[:, :self.opt.k_stat, :], another_encoded_tensor.view(b, t, num_features)[:, :self.opt.k_stat, :]
            
            mu_zf, logvar_zf = self.static_rnn(in_anch)
            zf_pos_mu, zf_pos_logvar = self.static_rnn(in_pos)
            zf_neg_mu, zf_neg_logvar = self.static_rnn(in_neg)
            
            if self.opt.slot_att is True:
                # Pass the static representations through slots and hopefully get object-centric representations
                mu_zf, logvar_zf = self.mu_slot_att(mu_zf).reshape(b, -1), self.logvar_slot_att(logvar_zf).reshape(b, -1)
                zf_pos_mu, zf_pos_logvar = self.mu_slot_att(zf_pos_mu).reshape(b, -1), self.logvar_slot_att(zf_pos_logvar).reshape(b, -1)
                zf_neg_mu, zf_neg_logvar = self.mu_slot_att(zf_neg_mu).reshape(b, -1), self.logvar_slot_att(zf_neg_logvar).reshape(b, -1)
        
        std_zf, zf_pos_std, zf_neg_std = torch.exp(0.5 * logvar_zf), torch.exp(0.5 * zf_pos_logvar), torch.exp(0.5 * zf_neg_logvar)
        
        return mu_zf, std_zf, zf_pos_mu, zf_pos_std, zf_neg_mu, zf_neg_std

    def get_dynamic_rep(self, encoded_inputs):
        b = self.opt.batch_size

        if self.opt.encoder in ['odecgru', 'cgru', 'cgru_sa']:
            bt, c, h, w = encoded_inputs.size()
            encoded_inputs = encoded_inputs.view(b, self.in_seq, c, h, w).permute(1, 0, 2, 3, 4)
            mu_zt, logvar_zt = self.dynamic_rnn(encoded_inputs, self.out_seq)  # Get posterior mu and std of dynamic latent variables z1....zt each of channel dim d_zt
            std_zt = torch.exp(0.5 * logvar_zt) 
            mu_std_zt = torch.cat((mu_zt, std_zt), dim=2)
            prior_mu_zt, prior_logvar_zt = self.prior_rnn(mu_std_zt.permute(1, 0, 2, 3, 4), self.out_seq)  # Get prior mu and std of dynamic latent variables z1....zt each of dim d_zt

        elif self.opt.encoder in ['default']:
            bt, num_features = encoded_inputs.size()[0], encoded_inputs.size()[1]
            t = bt // b
            mu_zt, logvar_zt = self.dynamic_rnn(encoded_inputs.view(b, t, num_features), self.out_seq)
            std_zt = torch.exp(0.5 * logvar_zt) 
            mu_std_zt = torch.cat((mu_zt, std_zt), dim=2)
            prior_mu_zt, prior_logvar_zt = self.prior_rnn(mu_std_zt)   # Get prior mu and std of dynamic latent variables z1....zt each of dim d_zt

        prior_std_zt = torch.exp(0.5 * prior_logvar_zt)
        return mu_zt, std_zt, prior_mu_zt, prior_std_zt

    def visualize_latent_dims(self, mu, std, sampled_z, type='static'):
        t = self.in_seq
        b = self.opt.batch_size
        mu = ((torch.tanh(mu) + 1.0)/2) * 255.0
        std = ((torch.tanh(std) + 1.0)/2) * 255.0
        sampled_z = ((torch.tanh(sampled_z) + 1.0)/2) * 255.0

        if self.opt.encoder in ['default']:
            if type == 'static':
                mu = mu.unsqueeze(1).repeat(1, t, 1).unsqueeze(3).cpu().detach().numpy()
                std = std.unsqueeze(1).repeat(1, t, 1).unsqueeze(3).cpu().detach().numpy()
                sampled_z = sampled_z.unsqueeze(1).repeat(1, t, 1).unsqueeze(3).cpu().detach().numpy()

                self.latent_dims['static_mu'] = [wandb.Image(m) for m in mu]
                self.latent_dims['static_std'] = [wandb.Image(m) for m in std]
                self.latent_dims['static_sampled'] = [wandb.Image(m) for m in sampled_z]
            else:
                mu = mu.unsqueeze(-1).cpu().detach().numpy()
                std = std.unsqueeze(-1).cpu().detach().numpy()
                sampled_z = sampled_z.unsqueeze(-1).cpu().detach().numpy()

                self.latent_dims['dynamic_mu'] = [wandb.Image(m) for m in mu]
                self.latent_dims['dynamic_std'] = [wandb.Image(m) for m in std]
                self.latent_dims['dynamic_sampled'] = [wandb.Image(m) for m in sampled_z]

        elif self.opt.encoder in ['cgru']:
            if type == 'static':
                mu = mu[:, :3, :, :].cpu().detach().permute(0, 2, 3, 1).numpy()
                std = std[:, :3, :, :].cpu().detach().permute(0, 2, 3, 1).numpy()
                sampled_z = sampled_z[:, :3, :, :].cpu().detach().permute(0, 2, 3, 1).numpy()

                self.latent_dims['static_mu'] = [wandb.Image(m) for m in mu]
                self.latent_dims['static_std'] = [wandb.Image(m) for m in std]
                self.latent_dims['static_sampled'] = [wandb.Image(m) for m in sampled_z]

            else:
                mu = mu[:, :, :3, :, :].cpu().detach().permute(0, 1, 3, 4, 2).numpy()
                std = std[:, :, :3, :, :].cpu().detach().permute(0, 1, 3, 4, 2).numpy()
                sampled_z = sampled_z[:, :, :3, :, :].cpu().detach().permute(0, 1, 3, 4, 2).numpy()

                self.latent_dims['dynamic_mu'] = wandb.Video(mu)
                self.latent_dims['dynamic_std'] = wandb.Video(std)
                self.latent_dims['dynamic_sampled'] = wandb.Video(sampled_z)
                
        return mu, std, sampled_z

    def forward(self, inputs):
        b, t, c, h, w = inputs.size()
        assert t == self.in_seq
        self.set_zero_losses()

        # Conv Encoding
        encoded_inputs = self.conv_encoder(inputs.view(b*t, c, h, w))

        # shuffle batch to get zf_pos and use another sequence for zf_neg
        _h, _w = encoded_inputs.size()[-2], encoded_inputs.size()[-1]
        shuffle_idx = torch.randperm(t)
        _5d_encoded_input = encoded_inputs.view(b, t, -1, _h, _w)
        shuffled_encoded_inputs = _5d_encoded_input[:, shuffle_idx, :, :, :].contiguous().view(b*t, -1, _h, _w)
        
        other = inputs[torch.from_numpy(np.random.permutation(len(inputs)))].to(self.device).view(b*t, c, h, w)
        another_encoded_tensor = self.conv_encoder(other)  

        # Get mu and std of static latent variable zf, zf_pos, zf_neg each of dim d_zf
        mu_zf, std_zf, zf_pos_mu, zf_pos_std, zf_neg_mu, zf_neg_std = self.get_static_rep(encoded_inputs, shuffled_encoded_inputs, another_encoded_tensor)
        # print("mu_zf", mu_zf.size())
        mu_zt, std_zt, prior_mu_zt, prior_std_zt = self.get_dynamic_rep(encoded_inputs)
        # print("mu_zt", mu_zt.size())

        # zf prior p(z_f) ~ N(0, 1) and zf posterior q(z_f | x_1:T)
        # if self.opt.slot_att is True and self.opt.encoder in ['cgru_sa']:
            # reshaped_mu_zf = mu_zf.view(b, self.opt.num_slots, -1, self.res_after_encoder, self.res_after_encoder)
            # reshaped_std_zf = std_zf.view(b, self.opt.num_slots, -1, self.res_after_encoder, self.res_after_encoder)
            # print("reshaped_mu_zf", reshaped_mu_zf.size())
            # self.p_zf = dist.Normal(loc=torch.zeros_like(reshaped_mu_zf), scale=torch.ones_like(reshaped_std_zf))
            # self.q_zf_xT = dist.Normal(loc=reshaped_mu_zf, scale=reshaped_std_zf)
        
        self.p_zf = dist.Normal(loc=torch.zeros_like(mu_zf), scale=torch.ones_like(std_zf))
        self.q_zf_xT = dist.Normal(loc=mu_zf, scale=std_zf)

        # zt prior p(z_t | z<t) and zt posterior q(z_t | x <= T)
        self.p_zt = dist.Normal(loc=prior_mu_zt, scale=prior_std_zt)
        self.q_zt_xt = dist.Normal(loc=mu_zt, scale=std_zt)
        zf_sample = self.q_zf_xT.rsample()
        zt_sample = self.q_zt_xt.rsample()
        # print("zf_sample", zf_sample.size())
        # print("zt_sample", zt_sample.size())

        self.mu_zf, self.std_zf, self.sampled_zf = self.visualize_latent_dims(mu_zf, std_zf, zf_sample, 'static')
        self.mu_zt, self.std_zt, self.sampled_zt = self.visualize_latent_dims(mu_zt, std_zt, zt_sample, 'dynamic')

        if self.opt.encoder == 'default' and self.opt.slot_att is True:
            # Make zf and zt, (b, t, num_slots, f)
            zf_zt = torch.cat((zf_sample.reshape(b, 1, -1).repeat(1, self.out_seq, 1), zt_sample), dim=2).view(b*self.out_seq, -1, 1, 1)
            # print("Combining", zf_sample.unsqueeze(1).repeat(1, self.out_seq, 1, 1).size(), zt_sample.unsqueeze(2).repeat(1, 1, self.opt.num_slots, 1).size())
            # zf_zt = torch.cat((zf_sample.unsqueeze(1).repeat(1, self.out_seq, 1, 1), zt_sample.unsqueeze(2).repeat(1, 1, self.opt.num_slots, 1)), dim=-1).view(b*self.out_seq*self.opt.num_slots, -1, 1, 1)
            # print("zf_zt", zf_zt.size(), zf_sample.size(), zt_sample.size())

        elif self.opt.encoder == 'default' and self.opt.slot_att is False:
            zf_zt = torch.cat((zf_sample.unsqueeze(1).repeat(1, self.out_seq, 1), zt_sample), dim=2).view(b*self.out_seq, -1, 1, 1)

        elif self.opt.encoder in ['odecgru', 'cgru']:
            zf_zt = torch.cat((zf_sample.unsqueeze(1).repeat(1, self.out_seq, 1, 1, 1), zt_sample), dim=2) # Join across channel dim
            _, _, c_, h_, w_ = zf_zt.size()
            zf_zt = zf_zt.view(b*self.out_seq, c_, h_, w_)

        elif self.opt.encoder in ['cgru_sa']:
            h, w = zf_sample.size()[-2], zf_sample.size()[-1]
            zf_zt = torch.cat((zf_sample.reshape(b, -1, h, w).unsqueeze(1).repeat(1, self.out_seq, 1, 1, 1), zt_sample), dim=2) # Join across channel dim
            _, _, c_, h_, w_ = zf_zt.size()
            zf_zt = zf_zt.view(b*self.out_seq, c_, h_, w_)
        
        x_hat = self.conv_decoder(zf_zt).view(b, self.out_seq, -1, self.h, self.w)
        
        if self.opt.slot_att is True and self.opt.unmasked is False:
            # split alpha masks and reconstruct.
            recons, masks = unstack_and_split(x_hat, batch_size=self.opt.batch_size, num_channels=self.in_ch)
            masks = F.softmax(masks, dim=1)
            recons = F.sigmoid(recons)
            x_hat = torch.sum(recons * masks, dim=1)
            self.masks = masks

        else:
            x_hat = F.sigmoid(x_hat)

        if self.opt.phase == 'train':
            # 1. VAE ELBO Loss
            self.get_vae_loss(x_hat, inputs, zf_sample, zt_sample) 

            # 2. SCC Loss
            if self.opt.encoder in ['cgru_sa']:
                zf_pos_mu   = zf_pos_mu.view(b, self.opt.num_slots, -1, self.res_after_encoder, self.res_after_encoder) 
                zf_pos_std  = zf_pos_std.view(b, self.opt.num_slots, -1, self.res_after_encoder, self.res_after_encoder) 
                zf_neg_mu   = zf_neg_mu.view(b, self.opt.num_slots, -1, self.res_after_encoder, self.res_after_encoder) 
                zf_neg_std  = zf_neg_std.view(b, self.opt.num_slots, -1, self.res_after_encoder, self.res_after_encoder) 

            zf_pos = dist.Normal(loc=zf_pos_mu, scale=zf_pos_std)
            zf_neg = dist.Normal(loc=zf_neg_mu, scale=zf_neg_std)
            self.get_scc_loss(zf_pos, zf_neg)
            # print("Got SCC")

            # 3. DFP Loss
            self.get_dfp_loss(zt_sample)  
            # print("Got DFP")

            # 4. MI Loss
            self.get_mi_loss()  

        return x_hat

    def get_prediction(self, inputs, batch_dict=None):
        self.ground_truth = batch_dict['data_to_predict'].to(self.device)
        self.in_flow_labels = batch_dict['in_flow_labels'].to(self.device)
        self.out_flow_labels = batch_dict['out_flow_labels'].to(self.device)
        self.ground_truth = (self.ground_truth + 0.5)
        pred_frames = self(inputs)
        return pred_frames
    
    def kl_divergence(self, p, q, z):
        log_qzx = q.log_prob(z)
        log_pz = p.log_prob(z)
        kl = (log_qzx - log_pz)
        
        if self.opt.encoder == 'default':   
            kl = kl.mean(-1)

        elif self.opt.encoder in ['odecgru', 'cgru', 'cgru_sa']: 
            dims = len(kl.size())
            kl = kl.mean(dim=(dims-3, dims-2, dims-1))

        return kl

    def get_vae_loss(self, x_hat, x, zf, zt):
        if self.opt.extrapolate is True:    x = self.ground_truth
        t = x.size()[1]
        
        # 1. Reconstruction loss: p(xt | zf, zt)
        recon_loss = F.mse_loss(x_hat, x, reduction='sum') / (self.opt.batch_size * t)

        # 2. KL for static latent variable zf
        q_zf_mean, q_zf_std = self.q_zf_xT.loc, self.q_zf_xT.scale
        q_zf_logvar = 2 * torch.log(q_zf_std)
        zf_KL_div_loss = -0.5 * torch.sum(1 + q_zf_logvar - torch.pow(q_zf_mean,2) - torch.exp(q_zf_logvar)) / (self.opt.batch_size * t)

        # 3. KL for dynamic latent variable zt
        z_prior_mean, z_prior_std = self.p_zt.loc, self.p_zt.scale
        z_post_mean, z_post_std = self.q_zt_xt.loc, self.q_zt_xt.scale
        z_prior_logvar, z_post_logvar = 2 * torch.log(z_prior_std), 2 * torch.log(z_post_std)
        z_prior_var, z_post_var = torch.exp(z_prior_logvar), torch.exp(z_post_logvar)
        zt_KL_div_loss = 0.5 * torch.sum(z_prior_logvar - z_post_logvar + ((z_post_var + torch.pow(z_post_mean - z_prior_mean, 2)) / z_prior_var) - 1) / (self.opt.batch_size * t)

        kl_loss = zf_KL_div_loss + zt_KL_div_loss

        # ELBO Loss
        self.vae_loss = (recon_loss + kl_loss).mean()
        self.recon_loss = recon_loss
        self.total_kl_loss = kl_loss.mean()
        self.zf_KL_div_loss = zf_KL_div_loss
        self.zt_KL_div_loss = zt_KL_div_loss
        
    def get_scc_loss(self, zf_pos, zf_neg):
        # zf equivalent to self.q_zf_xT -- time-invariant representation from real data
        # zf_pos -- time-invariant representation from shuffled real video
        # zf_neg -- time-invariant representation from another video
        zf_sample = self.q_zf_xT.rsample()
        
        if self.opt.slot_att is True and self.opt.encoder in ['cgru_sa']:
            b, f_s, h, w = zf_sample.size()
            zf_sample = zf_sample.view(b, self.opt.num_slots, -1, h, w)

        zf_pos_sample = zf_pos.sample()
        zf_neg_sample = zf_neg.sample()
        
        # max(D(zf, zf_pos) - D(zf, zf_neg) + margin, 0)
        self.scc_loss = self._triplet_loss(zf_sample, zf_pos_sample, zf_neg_sample)

    def get_dfp_loss(self, zt):
        if self.opt.extrapolate is True:    
            motion_mag_label = self.out_flow_labels.float()
        elif self.opt.reconstruct is True:
            motion_mag_label = self.in_flow_labels.float()
            
        pred_area = self.dfp_net(zt)
        self.dfp_loss = F.binary_cross_entropy(torch.sigmoid(pred_area), motion_mag_label)
        return

    def get_mi_loss(self):
        M = self.opt.batch_size

        if self.opt.phase == 'train':
            N = self.opt.train_test_split * self.opt.data_points    # dataset size
        else:
            N = (1 - self.opt.train_test_split) * self.opt.data_points
        # sum t from 1..T H(zf) + H(zt) - H(zf, zt)
        # zf_dist is self.q_zf_xT and zt_dist = self.q_zt_xt
        # H(.) -> -log q(.)

        def dist_op(dist1, op, t=False):
            if t is True:
                if self.opt.encoder == 'default':
                    return dist.Normal(loc=op(dist1.loc.permute(1, 0, 2)), scale=op(dist1.scale.permute(1, 0, 2)))
                elif self.opt.encoder in ['odecgru', 'cgru', 'cgru_sa']:
                    return dist.Normal(loc=op(dist1.loc.permute(1, 0, 2, 3, 4)), scale=op(dist1.scale.permute(1, 0, 2, 3, 4)))
            else:
                return dist.Normal(loc=op(dist1.loc), scale=op(dist1.scale))

        z_t1 = dist_op(self.q_zt_xt, lambda x: x.unsqueeze(1), t=True) # t, 1, b, d_zt / t, 1, b, c, h, w
        z_t2 = dist_op(self.q_zt_xt, lambda x: x.unsqueeze(2), t=True) # t, b, 1, d_zt / t, b, 1, c, h, w
        z_t2_sample = z_t2.rsample()
        t = z_t2_sample.size()[0]
        
        log_q_t = z_t1.log_prob(z_t2_sample)            # t, b, b, d_zt, (h, w) if cgru encoder

        z_f1 = dist_op(self.q_zf_xT, lambda x: x.unsqueeze(0)) # 1, b, d_zf / 1, b, c, h, w
        z_f2 = dist_op(self.q_zf_xT, lambda x: x.unsqueeze(1)) # b, 1, d_zf / b, 1, c, h, w
        
        if self.opt.encoder == 'default':
            log_q_f = z_f1.log_prob(z_f2.rsample()).unsqueeze(0).repeat(t, 1, 1, 1)              # t, b, b, d_zf
        
        elif self.opt.encoder in ['cgru_sa']:
            log_q_f = z_f1.log_prob(z_f2.rsample()).unsqueeze(0).repeat(t, 1, 1, 1, 1, 1)   # t, b, b, num_slots*slot_dim, h, w 
        
        else:
            log_q_f = z_f1.log_prob(z_f2.rsample()).unsqueeze(0).repeat(t, 1, 1, 1, 1, 1) 

        # print("log_q_f", log_q_f.size(), log_q_t.size())

        log_q_ft = torch.cat((log_q_t, log_q_f), dim=3)
        # print("log_q_ft", log_q_ft.size())
        
        if self.opt.encoder in ['default']:

            H_t = - (log_q_t.sum(3) - math.log(N * M)).logsumexp(2)  
            H_f = - (log_q_f.sum(3) - math.log(N * M)).logsumexp(2)  
            H_ft = - (log_q_ft.sum(3) - math.log(N * M)).logsumexp(2)
        
        elif self.opt.encoder in ['cgru', 'cgru_sa']:
            log_q_ft = torch.cat((log_q_t, log_q_f), dim=-3)
            
            dims = len(z_t1.loc.size())
            H_t = - (log_q_t.sum(dim=(dims-3, dims-2, dims-1)) - math.log(N * M)).logsumexp(2)  
            H_f = - (log_q_f.sum(dim=(dims-3, dims-2, dims-1)) - math.log(N * M)).logsumexp(2)
            H_ft = - (log_q_ft.sum(dim=(dims-3, dims-2, dims-1)) - math.log(N * M)).logsumexp(2)

        self.mi_loss = F.relu(- H_ft + H_f + H_t).mean()
        
    def get_loss(self):
        loss = (self.opt.l0 * self.vae_loss) + (self.opt.l1 * self.scc_loss) + (self.opt.l2 * self.dfp_loss) + (self.opt.l3 * self.mi_loss)

        loss_dict = {
            'Loss': loss.item(),
            'VAE Loss': self.vae_loss.item(),
            'Reconstruction Loss': self.recon_loss.item(),
            'Total KL Loss': self.total_kl_loss.item(),
            'Static Latent KL Loss': self.zf_KL_div_loss.item(),
            'Dynamic Latent KL Loss': self.zt_KL_div_loss.item(),
            'SCC Loss': self.scc_loss.item(),
            'DFP Loss': self.dfp_loss.item(),
            'MI Loss': self.mi_loss.item()
        }
        print("VAE Loss:", loss_dict['VAE Loss'], "|", 'Reconstruction Loss:', loss_dict['Reconstruction Loss'], "| Static KL", loss_dict['Static Latent KL Loss'], "|Dynamic KL", loss_dict['Dynamic Latent KL Loss'], "| SCC Loss:", loss_dict['SCC Loss'], "| DFP Loss:", loss_dict['DFP Loss'] ,"| MI Loss:", loss_dict['MI Loss'])
        print()
        return loss, loss_dict

def unstack_and_split(x, batch_size, num_channels=3):
    """Unstack batch dimension and split into channels and alpha mask."""
    reqd_shape = [batch_size, -1] + list(x.size())[1:]
    unstacked = x.view(reqd_shape) 
    channels, masks = torch.split(unstacked, [num_channels, 1], dim=-3)
    return channels, masks