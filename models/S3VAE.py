import sys
sys.path.append('..')
# sys.path.append('./flownet2_pytorch/networks')
# sys.path.append('./flownet2_pytorch/networks/channelnorm_package')
# sys.path.append('./flownet2_pytorch/networks/correlation_package')
# sys.path.append('./flownet2_pytorch/networks/resample2d_package')

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import distributions as dist
import numpy as np
import wandb
import PIL

from modules.S3VAE_ED import Encoder, GRUEncoder, ConvGRUEncoder, Decoder, DFP
# from flownet2_pytorch.models import FlowNet2

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
        n_hid = 256
        self.latent_dims = {}

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
        self.conv_encoder = Encoder(in_ch, opt.encoder).to(device)
        resize = self.conv_encoder.resize
        self.res_after_encoder = opt.resolution // resize
        
        if opt.encoder == 'default':
            self.static_rnn = GRUEncoder(128, n_hid, d_zf, ode=False, device=device, type='static', batch_first=True, opt=opt).to(device)
            self.dynamic_rnn = GRUEncoder(128, n_hid, d_zt, ode=opt.ode, device=device, type='dynamic', batch_first=True, opt=opt).to(device)
            self.prior_rnn = GRUEncoder(d_zt * 2 * self.num_rims, n_hid, d_zt * self.num_rims, ode=False, device=device, type='prior', batch_first=True, opt=opt).to(device)
        
        elif opt.encoder in ['odecgru', 'cgru', 'cgru_sa']:
            conv_encoder_out_ch = self.conv_encoder.layers[-3].out_channels
            self.static_rnn = ConvGRUEncoder(conv_encoder_out_ch, d_zf, opt, device, self.conv_encoder.resize, type='static').to(device)
            self.dynamic_rnn = ConvGRUEncoder(conv_encoder_out_ch, d_zt, opt, device, self.conv_encoder.resize, type='dynamic').to(device)
            self.prior_rnn = ConvGRUEncoder(d_zt*2, d_zt, opt, device, self.conv_encoder.resize,type='prior').to(device)

        if opt.slot_att is True:
            self.mu_slot_att = SlotAttentionAutoEncoder(opt, resolution=(self.h // resize, self.w // (resize)), num_slots=opt.num_slots, num_iterations=opt.num_iterations, device=device, resize=resize, broadcast=broadcast).to(device)
            self.std_slot_att = SlotAttentionAutoEncoder(opt, resolution=(self.h // resize, self.w // (resize)), num_slots=opt.num_slots, num_iterations=opt.num_iterations, device=device, resize=resize, std=True, broadcast=broadcast).to(device)
            
            if opt.unmasked is False: ch_ = in_ch + 1
            else: ch_ = in_ch
            self.conv_decoder = Decoder((opt.num_slots * self.opt.slot_size) + (d_zt * self.num_rims), ch_, opt).to(device)

        else:
            self.conv_decoder = Decoder(d_zf + d_zt, in_ch, opt).to(device)
            
        # TODO Dynamic Factor Prediction
        # self.dfp_net = DFP(z_size=d_zt)
        # self.flownet = FlowNet2(opt)
        # self.flownet.load_state_dict(torch.load(opt.flownet_params_path))
        # print("Loaded params for FlowNet model")

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
            
            # Get posterior mu and std of static latent variable zf of channels dim d_zf
            mu_zf, std_zf = self.static_rnn(encoded_inputs, t)
            zf_pos_mu, zf_pos_std = self.static_rnn(shuffled_encoded_inputs, t)
            zf_neg_mu, zf_neg_std = self.static_rnn(another_encoded_tensor, t)

            # Pass the static representations through slots and get object-centric representations
            if self.opt.slot_att is True and self.opt.encoder in ['cgru_sa']:
                mu_zf, std_zf = self.mu_slot_att(mu_zf), self.std_slot_att(std_zf)
                zf_pos_mu, zf_pos_std = self.mu_slot_att(zf_pos_mu), self.std_slot_att(zf_pos_std)
                zf_neg_mu, zf_neg_std = self.mu_slot_att(zf_neg_mu), self.std_slot_att(zf_neg_std)
            
        elif self.opt.encoder in ['default']:
            num_features = encoded_inputs.size()[1]
            mu_zf, std_zf = self.static_rnn(encoded_inputs.view(b, t, num_features))
            zf_pos_mu, zf_pos_std = self.static_rnn(shuffled_encoded_inputs.view(b, t, num_features))
            zf_neg_mu, zf_neg_std = self.static_rnn(another_encoded_tensor.view(b, t, num_features))
            
            if self.opt.slot_att is True:
                # Pass the static representations through slots and hopefully get object-centric representations
                mu_zf, std_zf = self.mu_slot_att(mu_zf).reshape(b, -1), self.std_slot_att(std_zf).reshape(b, -1)
                zf_pos_mu, zf_pos_std = self.mu_slot_att(zf_pos_mu), self.std_slot_att(zf_pos_std)
                zf_neg_mu, zf_neg_std = self.mu_slot_att(zf_neg_mu), self.std_slot_att(zf_neg_std)
            
        return mu_zf, std_zf, zf_pos_mu, zf_pos_std, zf_neg_mu, zf_neg_std

    def get_dynamic_rep(self, encoded_inputs):
        b = self.opt.batch_size

        if self.opt.encoder in ['odecgru', 'cgru', 'cgru_sa']:
            bt, c, h, w = encoded_inputs.size()
            encoded_inputs = encoded_inputs.view(b, self.in_seq, c, h, w).permute(1, 0, 2, 3, 4)
            mu_zt, std_zt = self.dynamic_rnn(encoded_inputs, self.out_seq)  # Get posterior mu and std of dynamic latent variables z1....zt each of channel dim d_zt
            mu_std_zt = torch.cat((mu_zt, std_zt), dim=2)
            prior_mu_zt, prior_std_zt = self.prior_rnn(mu_std_zt.permute(1, 0, 2, 3, 4), self.out_seq)  # Get prior mu and std of dynamic latent variables z1....zt each of dim d_zt

        elif self.opt.encoder in ['default']:
            bt, num_features = encoded_inputs.size()[0], encoded_inputs.size()[1]
            t = bt // b
            mu_zt, std_zt = self.dynamic_rnn(encoded_inputs.view(b, t, num_features), self.out_seq)
            mu_std_zt = torch.cat((mu_zt, std_zt), dim=2)
            prior_mu_zt, prior_std_zt = self.prior_rnn(mu_std_zt)   # Get prior mu and std of dynamic latent variables z1....zt each of dim d_zt

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

                self.latent_dims['dynamic_mu'] = [wandb.Video(m) for m in mu]
                self.latent_dims['dynamic_std'] = [wandb.Video(m) for m in std]
                self.latent_dims['dynamic_sampled'] = [wandb.Video(m) for m in sampled_z]
                


    def forward(self, inputs):
        self.set_zero_losses()
        other = inputs[torch.from_numpy(np.random.permutation(len(inputs)))].to(self.device)
        b, t, c, h, w = inputs.size()
        assert t == self.in_seq

        # Conv Encoding
        encoded_inputs = self.conv_encoder(inputs.view(b*t, c, h, w))

        # shuffle batch to get zf_pos and use another sequence for zf_neg
        _h, _w = encoded_inputs.size()[-2], encoded_inputs.size()[-1]
        _5d_encoded_input = encoded_inputs.view(b, t, -1, _h, _w)
        shuffle_idx = torch.randperm(t)
        shuffled_encoded_inputs = _5d_encoded_input[:, shuffle_idx].contiguous().view(b*t, -1, _h, _w)
        another_encoded_tensor = self.conv_encoder(other.view(b*t, c, h, w))  

        # Get mu and std of static latent variable zf, zf_pos, zf_neg each of dim d_zf
        mu_zf, std_zf, zf_pos_mu, zf_pos_std, zf_neg_mu, zf_neg_std = self.get_static_rep(encoded_inputs, shuffled_encoded_inputs, another_encoded_tensor)
        mu_zt, std_zt, prior_mu_zt, prior_std_zt = self.get_dynamic_rep(encoded_inputs)

        # zf prior p(z_f) ~ N(0, 1) and zf posterior q(z_f | x_1:T)
        if self.opt.slot_att is True and self.opt.encoder in ['cgru_sa']:
            reshaped_mu_zf = mu_zf.view(b, self.opt.num_slots, -1, self.res_after_encoder, self.res_after_encoder)
            reshaped_std_zf = std_zf.view(b, self.opt.num_slots, -1, self.res_after_encoder, self.res_after_encoder)
            self.p_zf = dist.Normal(loc=torch.zeros_like(reshaped_mu_zf), scale=torch.ones_like(reshaped_std_zf))
            self.q_zf_xT = dist.Normal(loc=reshaped_mu_zf, scale=reshaped_std_zf)
        
        else:
            self.p_zf = dist.Normal(loc=torch.zeros_like(mu_zf), scale=torch.ones_like(std_zf))
            self.q_zf_xT = dist.Normal(loc=mu_zf, scale=std_zf)

        # zt prior p(z_t | z<t) prior and zt posterior q(z_t | x <= T) posterior
        self.p_zt = dist.Normal(loc=prior_mu_zt, scale=prior_std_zt)
        self.q_zt_xt = dist.Normal(loc=mu_zt, scale=std_zt)
        zf_sample = self.q_zf_xT.rsample()
        zt_sample = self.q_zt_xt.rsample()

        self.visualize_latent_dims(mu_zf, std_zf, zf_sample, 'static')
        self.visualize_latent_dims(mu_zt, std_zt, zt_sample, 'dynamic')

        if self.opt.encoder == 'default' and self.opt.slot_att is True:
            zf_zt = torch.cat((zf_sample.reshape(b, 1, -1).repeat(1, self.out_seq, 1), zt_sample), dim=2).view(b*self.out_seq, -1, 1, 1)
        
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

            # 3. TODO: DFP Loss
            # self.get_dfp_loss(x_hat)  

            # 4. MI Loss
            self.get_mi_loss()  

        # print("[End of forward: S3VAE]")
        return x_hat

    def get_prediction(self, inputs, batch_dict=None):
        self.ground_truth = batch_dict['data_to_predict'].to(self.device)
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

        if self.opt.extrapolate is True:    
            x = self.ground_truth
        
        # 1. Reconstruction loss: p(xt | zf, zt)
        scale = torch.exp(self.log_scale)
        log_pxz = dist.Normal(x_hat, scale).log_prob(x)
        recon_loss = log_pxz.mean(dim=(1,2,3,4))   # sum over t = 1..T log p(xt| zf, zt)  {dim 1 sum over time, dims 2,3,4 sum over c,h,w}

        # 2. KL for static latent variable zf
        zf_KL_div_loss = self.kl_divergence(self.p_zf, self.q_zf_xT, zf)
        # 3. Sum KL across time dimension for dynamic latent variable zt
        zt_KL_div_loss = self.kl_divergence(self.p_zt, self.q_zt_xt, zt).sum(dim=(1))
        
        if self.opt.encoder in ['cgru_sa']:
            zf_KL_div_loss = zf_KL_div_loss.mean(dim=1)

        kl_loss = zf_KL_div_loss + zt_KL_div_loss

        # ELBO Loss
        self.vae_loss = (- recon_loss + kl_loss).mean()
        self.recon_loss = - recon_loss.mean()
        self.total_kl_loss = kl_loss.mean()
        self.zf_KL_div_loss = zf_KL_div_loss.mean()
        self.zt_KL_div_loss = zt_KL_div_loss.mean()

    def get_scc_loss(self, zf_pos, zf_neg):
        # zf equivalent to self.q_zf_xT -- time-invariant representation from real data
        # zf_pos -- time-invariant representation from shuffled real video
        # zf_neg -- time-invariant representation from another video
        zf_sample = self.q_zf_xT.rsample()
        zf_pos_sample = zf_pos.sample()
        zf_neg_sample = zf_neg.sample()
        
        # For slotted S3VAE
        if len(zf_sample.size()) == 2:
            zf_pos_sample = zf_pos_sample.view(self.opt.batch_size, -1)
            zf_neg_sample = zf_neg_sample.view(self.opt.batch_size, -1)
        
        # max(D(zf, zf_pos) - D(zf, zf_neg) + margin, 0)
        self.scc_loss = self._triplet_loss(zf_sample, zf_pos_sample, zf_neg_sample)

    def get_dfp_loss(self, x_hat):
        pass

    def get_mi_loss(self):
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
        
        if self.opt.encoder == 'default':
            log_q_t = z_t1.log_prob(z_t2.rsample()).sum(-1)                             # t, b, b
        else:
            dims = len(z_t1.loc.size())
            log_q_t = z_t1.log_prob(z_t2.rsample()).sum(dim=(dims-3, dims-2, dims-1))   # t, b, b
        
        H_t = log_q_t.logsumexp(2).mean(1) - np.log(log_q_t.shape[2]) # t
        
        z_f1 = dist_op(self.q_zf_xT, lambda x: x.unsqueeze(0)) # 1, b, d_zf / 1, b, c, h, w
        z_f2 = dist_op(self.q_zf_xT, lambda x: x.unsqueeze(1)) # b, 1, d_zf / b, 1, c, h, w
        
        if self.opt.encoder == 'default':
            log_q_f = z_f1.log_prob(z_f2.rsample()).sum(-1)                             # b, b 
        
        elif self.opt.encoder in ['cgru_sa']:
            dims = len(z_f1.loc.size())
            log_q_f = z_f1.log_prob(z_f2.rsample()).mean(dim=(2,3,4,5))
        
        else:
            dims = len(z_f1.loc.size())
            log_q_f = z_f1.log_prob(z_f2.rsample()).sum(dim=(dims-3, dims-2, dims-1))   # b, b
        
        H_f = log_q_f.logsumexp(1).mean(0) - np.log(log_q_t.shape[2])  
        
        # print("ZF", z_f1.log_prob(z_f2.rsample()).size())
        # print("ZT", z_t1.log_prob(z_t2.rsample()).size())
        # print("Log t and Log q", log_q_t.size(), log_q_f.size())
        # print("hf, ht", H_f.size(), H_t.size())
        # print(log_q_f, H_f)

        if self.opt.encoder in ['cgru_sa']:
            H_ft = (log_q_f + log_q_t).logsumexp(1).mean(1) # t
            self.mi_loss = -(H_f.mean() + H_t.mean() - H_ft.mean())
        else:
            H_ft = (log_q_f.unsqueeze(0) + log_q_t).logsumexp(1).mean(1) # t
            self.mi_loss = -(H_f + H_t.mean() - H_ft.mean())
        
    def get_loss(self):
        # TODO: add self.opt.l2 * self.dfp_loss after adding DFP and uncomment DFP Loss in loss_dict below
        loss = self.vae_loss + (self.opt.l1 * self.scc_loss) + (self.opt.l3 * self.mi_loss)

        loss_dict = {
            'Loss': loss.item(),
            'VAE Loss': self.vae_loss.item(),
            'Reconstruction Loss': self.recon_loss.item(),
            'Total KL Loss': self.total_kl_loss.item(),
            'Static Latent KL Loss': self.zf_KL_div_loss.item(),
            'Dynamic Latent KL Loss': self.zt_KL_div_loss.item(),
            'SCC Loss': self.scc_loss.item(),
            # 'DFP Loss': self.dfp_loss.item(),
            'MI Loss': self.mi_loss.item()
        }
        print("VAE Loss:", loss_dict['VAE Loss'], "|", 'Reconstruction Loss:', loss_dict['Reconstruction Loss'], "| SCC Loss:", loss_dict['SCC Loss'], "| MI Loss:", loss_dict['MI Loss'])
        return loss, loss_dict

def unstack_and_split(x, batch_size, num_channels=3):
    """Unstack batch dimension and split into channels and alpha mask."""
    reqd_shape = [batch_size, -1] + list(x.size())[1:]
    unstacked = x.view(reqd_shape) 
    channels, masks = torch.split(unstacked, [num_channels, 1], dim=-3)
    return channels, masks