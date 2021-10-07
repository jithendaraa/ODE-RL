import sys
sys.path.append('..')

import torch
import torch.nn as nn
import torch.nn.functional as F
import time

from modules.Attention import blocked_grad
from modules.BlocksCore import BlocksCore

'''
Core blocks module.  Takes:
    input: (ts, minibatch, h)
    hx: (ts, mb, h)
    cx: (ts, mb, h)

    output:
    output, hx, cx
'''

class RIM_GRU(nn.Module):
    def __init__(self, ninp, n_hid, opt, device=None, dropout=0.5, nlayers=1, 
                discrete_input=False, use_inactive=False, blocked_grad=False,
                num_modules_read_input=2):

        super(RIM_GRU, self).__init__()
        self.n_hid = n_hid
        self.nlayers = nlayers
        self.use_inactive = use_inactive
        self.blocked_grad = blocked_grad
        self.layer_dilation = [1] * nlayers
        self.block_dilation = [1] * nlayers
        self.bc_list, self.dropout_list = [], []
        self.num_blocks = opt.num_blocks
        self.device = device

        print()
        print('Top k Blocks: ', opt.topk)
        print('Number of Inputs, ninp: ', ninp)
        print('Dimensions of Hidden Layers: ', n_hid)
        print('Number of Blocks: ', opt.num_blocks)
        print("Dropout rate", dropout)
        print('Is the model using inactive blocks for higher representations? ', use_inactive)

        self.drop = nn.Dropout(dropout)
        self.sigmoid = nn.Sigmoid()
        self.sm = nn.Softmax(dim=1)
        num_blocks_in = [1 for _ in opt.topk]

        # Blocks Core and Dropout networks
        for i in range(nlayers):
            if i==0:
                self.bc_list.append(BlocksCore(ninp, n_hid[i], num_blocks_in[i], opt.num_blocks[i], opt.topk[i], True, opt, num_modules_read_input=num_modules_read_input))
            else:
                self.bc_list.append(BlocksCore(n_hid[i-1], n_hid[i], num_blocks_in[i], opt.num_blocks[i], opt.topk[i], True, opt, num_modules_read_input=num_modules_read_input))
        for i in range(nlayers - 1):
            self.dropout_list.append(nn.Dropout(dropout))
        
        self.bc_list = nn.ModuleList(self.bc_list)
        self.dropout_list = nn.ModuleList(self.dropout_list)
        print("RIM GRU initialised!")

    def init_hidden(self, b):
        weight = next(self.bc_list[0].block_gru.parameters())
        hidden = []
        for i in range(self.nlayers):
            hidden.append(( weight.new_zeros(b, self.n_hid[i]).to(self.device) ))
        return hidden

    def forward(self, input, hidden, seq_len=10):
        inp = input
        new_hidden = [[] for _ in range(self.nlayers)]
        print("input", inp.size(), hidden[0].size())

        for idx_layer in range(self.nlayers):
            output = []
            t0 = time.time()
            self.bc_list[idx_layer].blockify_params()
            hx = hidden
            
            for t_step in range(seq_len):
                if t_step % self.layer_dilation[idx_layer] == 0:
                    
                    if t_step % self.block_dilation[idx_layer] == 0:
                        hx, mask = self.bc_list[idx_layer](inp[t_step], hx, t_step, do_block=True)
                    else:
                        hx, mask = self.bc_list[idx_layer](inp[t_step], hx, t_step, do_block=False)
                    
                print("Got hx", idx_layer, self.nlayers - 1)

                if idx_layer < self.nlayers - 1:
                    print("HERE")
                    if self.use_inactive and self.opt.inactive_rims is True:
                        if self.blocked_grad:
                            bg = blocked_grad()
                            output.append(bg(hx,mask))
                        else:
                            output.append(hx)
                    else:
                        if self.blocked_grad:
                            bg = blocked_grad()
                            output.append((mask)*bg(hx,mask))
                        else:
                            output.append((mask)*hx)
                    print("OUT")
                else:
                    output.append(hx)

            output = torch.stack(output)
            if idx_layer < self.nlayers - 1:
                layer_input = self.dropout_list[idx_layer](output)
            else:
                layer_input = output
            
            new_hidden[idx_layer] = hx
        
        hidden = new_hidden
        output = self.drop(output)
        return output, hidden


