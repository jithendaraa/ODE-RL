import sys
sys.path.append('..')

import torch
import torch.nn as nn
import torch.nn.functional as F
import time

from modules.Attention import MultiHeadAttention, blocked_grad
from modules.BlockGRU import BlockCGRU

class RIM_CGRU(nn.Module):
    def __init__(self, ninp, opt, dropout=0.5, nlayers=1, 
                discrete_input=False, use_inactive=False, blocked_grad=False,
                num_modules_read_input=2):

        super(RIM_CGRU, self).__init__()
        self.ninp = ninp
        self.opt = opt
        self.nlayers = nlayers

        # Blocks Core and Dropout networks
        for i in range(nlayers):
            if i==0:
                self.bc_list.append(BlocksCore(ninp, n_hid[i], num_blocks_in[i], opt.num_blocks[i], opt.topk[i], True, opt, num_modules_read_input=num_modules_read_input))
            else:
                self.bc_list.append(BlocksCore(n_hid[i-1], n_hid[i], num_blocks_in[i], opt.num_blocks[i], opt.topk[i], True, opt, num_modules_read_input=num_modules_read_input))
        for i in range(nlayers - 1):
            self.dropout_list.append(nn.Dropout(dropout))

        print("RIM CGRU initialised!")

    def forward(self, input, hidden, seq_len):
        print("[Inside forward of RIM_CGRU]")

        inp = input
        new_hidden = [[] for _ in range(self.nlayers)]

        for idx_layer in range(self.nlayers):
            output = []
            t0 = time.time()
            self.bc_list[idx_layer].blockify_params()
            hx = hidden

        print("[End forward of RIM_CGRU]")


class ConvBlocksCore(nn.Module):
    def __init__(self, ninp, n_hid, num_blocks_in, num_blocks_out, topkval, step_att, opt, num_modules_read_input=2, device=None):
        super(ConvBlocksCore, self).__init__()

        self.block_size_in = n_hid // num_blocks_in
        self.block_size_out = n_hid // num_blocks_out
        self.att_out = self.block_size_out*4

        self.block_cgru = BlockCGRU()
    
    def blockify_params(self):
        self.block_cgru.blockify_params()
    
    def forward(self):
        pass