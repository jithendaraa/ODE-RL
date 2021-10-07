# Disentagled, Slot-Sequential Variational AutoEncoder
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
from modules.RIM_GRU import RIM_GRU


class DS2VAE(nn.Module):
    def __init__(self, opt, device):
        super(DS2VAE, self).__init__()
        self.opt = opt
        self.device = device
        self.set_in_out_seq()

        channels = 32

        self.C3D_encoder = nn.Sequential(
            nn.Conv3d(self.opt.in_channels, channels, (3, 3, 3), (1, 2, 2), (1, 1, 1)), nn.LeakyReLU(0.2)
        ).to(device)
        self.zf_net = C3DEncoder(channels, opt.d_zf, mode='static').to(device)
        self.zt_net = C3DEncoder(channels, opt.n_hid[0], mode='dynamic').to(device)

        self.slot_zf = SlotAttentionAutoEncoder(opt, num_slots=opt.num_slots, num_iterations=opt.num_iterations, device=device).to(device)
        self.slot_zf_mu_net = nn.Linear(opt.slot_size, opt.slot_size).to(device)
        self.slot_zf_logvar_net = nn.Linear(opt.slot_size, opt.slot_size).to(device)

        self.dynamic_net = RIM_GRU(opt.emsize, opt.n_hid, opt, device).to(device)

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

    def forward(self, inputs):
        b, t, c, h, w = inputs.size()
        assert t == self.in_seq
        print("Inputs:", inputs.size())

        # 1. C3D Encoding - inputs to C3D must have dims (b, c, t, h, w) and gives out (B, T, C, H ,W)
        encoded_inputs = self.C3D_encoder(inputs.permute(0, 2, 1, 3, 4))
        print("After C3D Encode:", encoded_inputs.size())

        # 2. Get zf encoding (b, t, c)
        zf_enc = self.zf_net(encoded_inputs).squeeze(-1).squeeze(-1).permute(0, 2, 1)
        print("mu_zf", zf_enc.size())

        # 3. From zf_enc, get mu and logvar for [zf_1...zf_s] where s is the number of slots
        slot_zf_enc = self.slot_zf(zf_enc)
        slot_post_mu_zf, slot_post_logvar_zf = self.slot_zf_mu_net(slot_zf_enc), self.slot_zf_logvar_net(slot_zf_enc)
        slot_post_std_zf = 0.5 * torch.exp(slot_post_logvar_zf)
        print("slot_zf (mu and std)", slot_post_mu_zf.size(), slot_post_std_zf.size())
        
        # 4. Set zf slot priors to N(0, 1)
        self.slot_zf_prior = dist.Normal(loc=torch.zeros_like(slot_post_mu_zf), scale=torch.ones_like(slot_post_std_zf))
        self.slot_zf_post = dist.Normal(loc=slot_post_mu_zf, scale=slot_post_std_zf)

        # 5. From encoded_inputs infer some z0_enc
        print()
        print("getting z0_encoding")
        print("encoded_inputs", encoded_inputs.size())
        z0_enc = self.zt_net(encoded_inputs).squeeze(-1).squeeze(-1).permute(0, 2, 1) # permute from (b, c, t) to (b, t, c)
        print("z0_enc", z0_enc.size())

        # 6. TODO From z0_enc, get [z0_1..z1_s] and unroll in time, till t, with a GRU using a RIM based approach. 
        # a. I.e., z0_1....z0_s = slots(z0_enc);
        # b. RIM(z0_1....z0_s) -> [[z1_1,...z1_s],
        #                          [z2_1....z2_s],
        #                          ..............
        #                          [zt_1....zt_s]]
        hidden = self.dynamic_net.init_hidden(b)
        print(len(hidden))
        # RIM_GRU takes in input of the form (t, b, features)
        output, hidden = self.dynamic_net(z0_enc.permute(1, 0, 2), hidden, seq_len=self.out_seq)
        print("output, hidden", output.size(), len(hidden))

        # 7. TODO Use a GRU to compute (t x s) priors for each variable calculated in 6b

        # 8. TODO Combine across all i {zf_i with [zi_1,...zi_s]} and deconv to get pred_X for timestep self.in_seq:self.in_seq+self.out_seq
        
        # 9. TODO Call get_loss() to compute MSE(ground_truth, pred_X) + mean(slot wise static KLD) + mean(slot wise dynamic KLD)
        

    def get_loss(self):
        # TODO compute MSE(ground_truth, pred_X)
        pass