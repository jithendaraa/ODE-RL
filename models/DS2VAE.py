# Disentagled, Slot-Sequential Variational AutoEncoder
import sys
sys.path.append('..')

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import distributions as dist
import numpy as np
import wandb

class DS2VAE(nn.Module):
    def __init__(self, opt, device):
        super(DS2VAE, self).__init__()
        self.opt = opt
        self.device = device

        self.set_in_out_seq()


    def set_in_out_seq(self):
        if self.opt.phase == 'train':
            self.in_seq, self.out_seq = self.opt.train_in_seq, self.opt.train_out_seq
        elif self.opt.phase == 'test':
            self.in_seq, self.out_seq = self.opt.test_in_seq, self.opt.test_out_seq
        else:
            NotImplementedError(f"Opt parameter 'phase={self.opt.phase}' is not supported! Please check configs.yaml")
    
    def get_prediction(self, inputs, batch_dict):
        if self.opt.extrapolate is True:    
            self.ground_truth = batch_dict['data_to_predict']
        elif self.opt.reconstruct is True:
            NotImplementedError("opt.reconstruct == True is not supported yet for model DS2VAE")
        pass

    def forward(self):
        pass

    def get_loss(self, pred_frames):
        pass