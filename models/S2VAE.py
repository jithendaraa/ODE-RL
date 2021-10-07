# S2VAE: Slot-Sequential VAE
import sys
sys.path.append('..')

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import distributions as dist
import numpy as np
import wandb

from modules.DS2VAE_ED import C3DEncoder, CNNDecoder
from modules.ConvGRUCell import ConvGRUCell
from modules.SlotAttention import SlotAttentionAutoEncoder

class S2VAE(nn.Module):
    def __init__(self, opt, device):
        super(S2VAE, self).__init__()
        self.opt = opt
        self.device = device
        self.set_in_out_seq()
        self.resolution_after_c3d = (4, 4)
        resize = opt.resolution // self.resolution_after_c3d[0]

        num_slots = opt.num_slots
        slot_size = opt.slot_size
        channels = 32

        self.init_c3d_encoder(channels)
        self.z_net = C3DEncoder(channels, opt.d_zf, mode=opt.transition).to(device)
        self.slot_z = SlotAttentionAutoEncoder(opt, resolution=self.resolution_after_c3d, num_slots=num_slots, num_iterations=opt.num_iterations, resize=resize, device=device).to(device)
        self.init_transition_net()
        if self.opt.prior == 'infer':   self.init_prior_transition_nets()
        self.cnn_decoder = CNNDecoder(num_slots * slot_size, self.opt.in_channels, self.opt.unmasked).to(device)
    
    def init_c3d_encoder(self, channels):
        k, s, p = (3, 3, 3), (1, 2, 2), (1, 1, 1)
        
        self.C3D_encoder = nn.Sequential(
            nn.Conv3d(self.opt.in_channels, channels, k, s, p), nn.LeakyReLU(0.2)
        ).to(self.device)

    def init_transition_net(self):
        if self.opt.transition in ['gru']:
            self.transition_net = nn.ModuleList([nn.GRU(self.opt.slot_size, self.opt.slot_size, self.opt.gru_layers) for i in range(self.opt.num_slots)]).to(self.device)
            self.slotwise_post_mu_nets = nn.ModuleList([nn.Linear(self.opt.slot_size, self.opt.slot_size) for i in range(self.opt.num_slots)]).to(self.device)
            self.slotwise_post_logvar_nets = nn.ModuleList([nn.Linear(self.opt.slot_size, self.opt.slot_size) for i in range(self.opt.num_slots)]).to(self.device)

        elif self.opt.transition in ['cgru']:
            self.transition_net = nn.ModuleList([ConvGRUCell(self.resolution_after_c3d, self.opt.slot_size, self.opt.slot_size, 5) for i in range(self.opt.num_slots)]).to(self.device)
            self.upsample_net = nn.Sequential(nn.ConvTranspose2d(self.opt.slot_size, self.opt.slot_size, 4, 1, 0)).to(self.device)
            self.slotwise_post_mu_nets = nn.ModuleList([nn.Conv2d(self.opt.slot_size, self.opt.slot_size, 3, 1, 1) for i in range(self.opt.num_slots)]).to(self.device)
            self.slotwise_post_logvar_nets = nn.ModuleList([nn.Conv2d(self.opt.slot_size, self.opt.slot_size, 3, 1, 1) for i in range(self.opt.num_slots)]).to(self.device)

    def init_prior_transition_nets(self):
        if self.opt.transition in ['gru']:
            self.transition_net = nn.ModuleList([nn.GRU(self.opt.slot_size, self.opt.slot_size, self.opt.gru_layers) for i in range(self.opt.num_slots)]).to(self.device)
            self.slotwise_post_mu_nets = nn.ModuleList([nn.Linear(self.opt.slot_size, self.opt.slot_size) for i in range(self.opt.num_slots)]).to(self.device)
            self.slotwise_post_logvar_nets = nn.ModuleList([nn.Linear(self.opt.slot_size, self.opt.slot_size) for i in range(self.opt.num_slots)]).to(self.device)

        elif self.opt.transition in ['cgru']:
            self.transition_net = nn.ModuleList([ConvGRUCell(self.resolution_after_c3d, self.opt.slot_size, self.opt.slot_size, 5) for i in range(self.opt.num_slots)]).to(self.device)
            self.upsample_net = nn.Sequential(nn.ConvTranspose2d(self.opt.slot_size, self.opt.slot_size, 4, 1, 0)).to(self.device)
            self.slotwise_post_mu_nets = nn.ModuleList([nn.Conv2d(self.opt.slot_size, self.opt.slot_size, 3, 1, 1) for i in range(self.opt.num_slots)]).to(self.device)
            self.slotwise_post_logvar_nets = nn.ModuleList([nn.Conv2d(self.opt.slot_size, self.opt.slot_size, 3, 1, 1) for i in range(self.opt.num_slots)]).to(self.device)

    def set_zero_losses(self):
        self.vae_loss = 0

    def set_in_out_seq(self):
        if self.opt.phase == 'train':
            self.in_seq, self.out_seq = self.opt.train_in_seq, self.opt.train_out_seq
        elif self.opt.phase == 'test':
            self.in_seq, self.out_seq = self.opt.test_in_seq, self.opt.test_out_seq
        else:
            NotImplementedError(f"Opt parameter 'phase={self.opt.phase}' is not supported! Please check configs.yaml")

    def set_ground_truth(self, batch_dict):
        input_frames, output_frames = (batch_dict['observed_data'] + 0.5), (batch_dict['data_to_predict'] + 0.5)
        if self.opt.extrapolate is True:    
            self.ground_truth = output_frames.to(self.device)
        elif self.opt.reconstruct is True:
            NotImplementedError("opt.reconstruct == True is not supported yet for model DS2VAE")

    def get_slots(self, z_enc):
        if self.opt.transition in ['cgru']:
            _, t_, c_, h_, w_ = z_enc.size()
            time_flattened_z_enc = z_enc.contiguous().view(-1, c_, h_, w_)
            slot_z0 = self.slot_z(time_flattened_z_enc)
        
        elif self.opt.transition in ['gru']:
            slot_z0 = self.slot_z(z_enc)
        
        return slot_z0

    def gru_rollout(self, inp, gru, mu_net, logvar_net):
        b, f = inp.size()
        x = inp.unsqueeze(0).repeat(gru.num_layers, 1, 1)
        z_rollouts, z_mus, z_logvars = [], [], []

        # Feed 0's as inputs and inp as hidden to extrapolate till seq_len
        for i in range(self.out_seq):
            zeros = torch.zeros((1, b, f)).to(self.device)
            _, x = gru(zeros, x)
            z_rollouts.append(x[-1, :, :].squeeze(0))
            z_i_mu = mu_net(x[-1, :, :].squeeze(0))
            z_i_logvar = logvar_net(x[-1, :, :].squeeze(0))
            z_mus.append(z_i_mu)
            z_logvars.append(z_i_logvar)

        z_rollouts = torch.stack(z_rollouts).permute(1, 0, 2) # Make b, t, f
        z_mus = torch.stack(z_mus).permute(1, 0, 2) # Make b, t, f
        z_logvars = torch.stack(z_logvars).permute(1, 0, 2) # Make b, t, f
        return z_rollouts, z_mus, z_logvars

    def cgru_rollout(self, x, cgru, mu_net, logvar_net):
        x = self.upsample_net(x.unsqueeze(-1).unsqueeze(-1))
        z_mus, z_logvars = [], []
        z_rollouts, hidden = cgru(None, x, seq_len=self.out_seq, dim=1) # b, t, c, h, w
        
        for i in range(self.out_seq):
            z = z_rollouts[:, i, :, :, :]
            z_mu, z_logvar = mu_net(z), logvar_net(z)
            z_mus.append(z_mu)
            z_logvars.append(z_logvar)
        
        z_mus = torch.stack(z_mus, dim=1)
        z_logvars = torch.stack(z_logvars, dim=1)
        return z_rollouts, z_mus, z_logvars

    def transition(self, z_i, transition_net, mu_net, logvar_net):
        if self.opt.transition in ['gru']:
            slot_z_i, slot_z_i_mu, slot_z_i_logvar = self.gru_rollout(z_i, transition_net, mu_net, logvar_net)
        elif self.opt.transition in ['cgru']:
            slot_z_i, slot_z_i_mu, slot_z_i_logvar = self.cgru_rollout(z_i, transition_net, mu_net, logvar_net)

        return slot_z_i, slot_z_i_mu, slot_z_i_logvar

    def forward(self, inputs):
        b, t, c, h, w = inputs.size()
        assert t == self.in_seq
        self.set_zero_losses()

        # 1. C3D Encoding - inputs to C3D must have dims (b, c, t, h, w) and gives out (B, T, C, H ,W)
        encoded_inputs = self.C3D_encoder(inputs.permute(0, 2, 1, 3, 4))

        # 2. Get z encoding (b, t, c)
        z_enc = self.z_net(encoded_inputs).permute(0, 2, 1, 3, 4)
        if self.opt.transition == 'gru':    z_enc = z_enc.squeeze(-1).squeeze(-1)

        # 3. From z_enc, get mu and logvar for [z0_1...z0_s] where s is the number of slots
        slot_z0 = self.get_slots(z_enc)
        print("slot_z0", slot_z0.size())
        # 4. From [z0_1..z1_s] use GRU to unroll in time, till t.
        #    GRU(z0_1....z0_s) -> mean and std of [[z1_1,...z1_s],
        #                                           [z2_1....z2_s],
        #                                           ..............
        #                                           [zt_1....zt_s]]
        
        slot_zs, slot_post_mus, slot_post_logvars = [], [], []
        permuted_slot_z0 = slot_z0.permute(1, 0, 2) # Make t, b, f 
        
        for z_i, transition_net, mu_net, logvar_net in zip(permuted_slot_z0, self.transition_net, self.slotwise_post_mu_nets, self.slotwise_post_logvar_nets):
            slot_z_i, slot_z_i_mu, slot_z_i_logvar = self.transition(z_i, transition_net, mu_net, logvar_net)
            slot_zs.append(slot_z_i)
            slot_post_mus.append(slot_z_i_mu)
            slot_post_logvars.append(slot_z_i_logvar)

        slot_zs = torch.stack(slot_zs, dim=1)               # Make (b, num_slots, t, f)
        slot_post_mus = torch.stack(slot_post_mus, dim=1)             # Make (b, num_slots, t, f)
        slot_post_logvars = torch.stack(slot_post_logvars, dim=1)     # Make (b, num_slots, t, f)
        print(slot_zs.size(), slot_post_mus.size(), slot_post_logvars.size())

        # 5. Set z slot priors to N(0, 1) or infer using self.slotwise_prior_gru depending on opt.prior is 'standard' or 'infer'
        if self.opt.prior == 'standard':
            self.slot_z_prior = dist.Normal(loc=torch.zeros_like(slot_post_mus), scale=torch.ones_like(slot_post_logvars))
        elif self.opt.prior == 'infer':
            # TODO Use a GRU to compute (t x s) priors for each variable calculated in 6b
            NotImplementedError("Inferring prior with a GRU not supported yet! Set prior to 'standard'")

        # 6. Set z slot posteriors and sample from slot_z_posterior
        slot_post_stds = 0.5 * torch.exp(slot_post_logvars)
        self.slot_z_post = dist.Normal(loc=slot_post_mus, scale=slot_post_stds)
        slot_z_posterior_sample = self.slot_z_post.rsample()
        if self.opt.model in ['S2VAE']: slot_z_posterior_sample.unsqueeze(-1).unsqueeze(-1) # b, slot, t, c, h, w
        print("slot_z_posterior_sample", slot_z_posterior_sample.size())

        b, num_slots, t, slot_size, h_, w_ = slot_z_posterior_sample.size()
        # 7. Combine all z slots and decode z slots. One decoder across slots
        if self.opt.unmasked is True:
            x_hat = self.cnn_decoder(slot_z_posterior_sample.view(-1, num_slots * slot_size, h_, w_)).view(b, t, c, h, w)
        else:
            NotImplementedError("Not implemented slot-masked decoding yet!")

        if self.opt.phase == 'train':
            self.get_vae_loss(x_hat)

        return x_hat

    def get_prediction(self, inputs, batch_dict):
        self.set_ground_truth(batch_dict)
        pred_frames = self(inputs)
        return pred_frames

    def get_vae_loss(self, x_hat):
        x = self.ground_truth
        t = x.size()[1]
        
        # 1. Reconstruction loss: p(xt | zf, zt)
        recon_loss = F.mse_loss(x_hat, x, reduction='sum') / (self.opt.batch_size * t)

        # 2. KL for z
        z_prior_mean, z_prior_std = self.slot_z_prior.loc, self.slot_z_prior.scale  # b, num_slots, t, slot_size
        z_post_mean, z_post_std = self.slot_z_post.loc, self.slot_z_post.scale
        z_prior_logvar, z_post_logvar = 2 * torch.log(z_prior_std), 2 * torch.log(z_post_std)
        z_prior_var, z_post_var = torch.exp(z_prior_logvar), torch.exp(z_post_logvar)
        z_KL_div_loss = 0.5 * torch.sum(z_prior_logvar - z_post_logvar + ((z_post_var + torch.pow(z_post_mean - z_prior_mean, 2)) / z_prior_var) - 1) / (self.opt.batch_size * t)

        self.vae_loss = (recon_loss + z_KL_div_loss).mean()
        self.recon_loss = recon_loss
        self.z_KL_div_loss = z_KL_div_loss

    def get_loss(self):
        loss = self.vae_loss
        loss_dict = {
            'Loss': loss.item(),
            'VAE Loss': self.vae_loss.item(),
            'Reconstruction Loss': self.recon_loss.item(),
            'Total KL Loss': self.z_KL_div_loss.item(),
        }
        print("VAE Loss:", loss_dict['VAE Loss'], "|", 'Reconstruction Loss:', loss_dict['Reconstruction Loss'], "| KL Loss:", loss_dict['Total KL Loss'])
        print()
        return loss, loss_dict