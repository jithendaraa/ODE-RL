# S2VAE: Slot-Sequential VAE
import sys
sys.path.append('..')

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import distributions as dist
import numpy as np
import wandb

from modules.DS2VAE_ED import C3DEncoder
from modules.SlotAttention import SlotAttentionAutoEncoder

class S2VAE(nn.Module):
    def __init__(self, opt, device):
        super(S2VAE, self).__init__()
        self.opt = opt
        self.device = device
        self.set_in_out_seq()

        channels = 32

        self.C3D_encoder = nn.Sequential(
            nn.Conv3d(self.opt.in_channels, channels, (3, 3, 3), (1, 2, 2), (1, 1, 1)), nn.LeakyReLU(0.2)
        ).to(device)
        self.z_net = C3DEncoder(channels, opt.d_zf, mode='static').to(device)

        self.slot_z = SlotAttentionAutoEncoder(opt, num_slots=opt.num_slots, num_iterations=opt.num_iterations, device=device).to(device)

        self.slotwise_post_gru = nn.ModuleList([nn.GRU(self.opt.slot_size, self.opt.slot_size) for i in range(self.opt.num_slots)]).to(device)
        self.slotwise_post_mu_nets = nn.ModuleList([nn.Linear(self.opt.slot_size, self.opt.slot_size)]).to(device)
        self.slotwise_post_logvar_nets = nn.ModuleList([nn.Linear(self.opt.slot_size, self.opt.slot_size)]).to(device)

        self.slotwise_prior_gru = nn.ModuleList([nn.GRU(self.opt.slot_size, self.opt.slot_size) for i in range(self.opt.num_slots)]).to(device)

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

    def get_prediction(self, inputs, batch_dict):
        self.set_ground_truth(batch_dict)
        pred_frames = self(inputs)
        return pred_frames

    def gru_rollout(self, inp, seq_len, gru, mu_net, logvar_net):
        b, f = inp.size()
        hx = torch.zeros((seq_len, b, f)).to(self.device)
        x = inp.unsqueeze(0)
        z_rollouts, z_mus, z_logvars = [], [], []

        # Feed 0's as inputs and inp as hidden to extrapolate till seq_len
        for i in range(seq_len):
            _, x = gru(hx[i:i+1, :, :], x)
            z_rollouts.append(x.squeeze(0))
            z_i_mu = mu_net(x.squeeze(0))
            z_i_logvar = logvar_net(x.squeeze(0))
            z_mus.append(z_i_mu)
            z_logvars.append(z_i_logvar)

        z_rollouts = torch.stack(z_rollouts)
        z_mus = torch.stack(z_mus)
        z_logvars = torch.stack(z_logvars)

        return z_rollouts, z_mus, z_logvars

    def forward(self, inputs):
        b, t, c, h, w = inputs.size()
        assert t == self.in_seq
        print("Inputs:", inputs.size())

        # 1. C3D Encoding - inputs to C3D must have dims (b, c, t, h, w) and gives out (B, T, C, H ,W)
        encoded_inputs = self.C3D_encoder(inputs.permute(0, 2, 1, 3, 4))
        print("After C3D Encode:", encoded_inputs.size())

        # 2. Get z encoding (b, t, c)
        z_enc = self.z_net(encoded_inputs).squeeze(-1).squeeze(-1).permute(0, 2, 1)
        print("z_enc", z_enc.size())

        # 3. From z_enc, get mu and logvar for [z0_1...z0_s] where s is the number of slots
        slot_z0 = self.slot_z(z_enc)
        print("slot_z", slot_z0.size())

        # 4. From [z0_1..z1_s] use GRU to unroll in time, till t.
        #    GRU(z0_1....z0_s) -> [[z1_1,...z1_s],
        #                          [z2_1....z2_s],
        #                          ..............
        #                          [zt_1....zt_s]]
        
        slot_zs, slot_post_mus, slot_post_logvars = [], [], []
        permuted_slot_z0 = slot_z0.permute(1, 0, 2) # Make t, b, f
        for z_i, post_gru, mu_net, logvar_net in zip(permuted_slot_z0, self.slotwise_post_gru, self.slotwise_post_mu_nets, self.slotwise_post_logvar_nets):
            slot_z_i, slot_z_i_mu, slot_z_i_logvar = self.gru_rollout(z_i, self.out_seq, post_gru, mu_net, logvar_net).permute(1, 0, 2) # Make b, t, f
            
            slot_zs.append(slot_z_i)
            slot_post_mus.append(slot_z_i_mu)
            slot_post_logvars.append(slot_z_i_logvar)
            print(f"Got z for slot: {slot_z_i.size()}")

        slot_zs = torch.stack(slot_zs, dim=1)               # Make (b, num_slots, t, f)
        slot_post_mus = torch.stack(slot_post_mus, dim=1)             # Make (b, num_slots, t, f)
        slot_post_logvars = torch.stack(slot_post_logvars, dim=1)     # Make (b, num_slots, t, f)

        print("slot_zs", slot_zs.size(), slot_post_mus.size(), slot_post_logvars.size())

        # 5. Set z slot priors to N(0, 1) or infer using self.slotwise_prior_gru depending on opt.prior is 'standard' or 'infer'
        # if self.opt.prior == 'standard':
        #     self.slot_z_prior = dist.Normal(loc=torch.zeros_like(slot_post_z_mu), scale=torch.ones_like(slot_post_z_std))
        # elif self.opt.prior == 'infer':
        #     NotImplementedError("Inferring prior with a GRU not supported yet! Set prior to 'standard'")

        # Set z slot posteriors

        # 7. TODO Use a GRU to compute (t x s) priors for each variable calculated in 6b

        # 8. TODO Combine across all i {zf_i with [zi_1,...zi_s]} and deconv to get pred_X for timestep self.in_seq:self.in_seq+self.out_seq
        
        # 9. TODO Call get_loss() to compute MSE(ground_truth, pred_X) + mean(slot wise static KLD) + mean(slot wise dynamic KLD)
        

    def get_loss(self):
        # TODO compute MSE(ground_truth, pred_X)
        pass