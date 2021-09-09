import torch
import torch.nn as nn
import torch.nn.functional as F

from Attention import MultiHeadAttention, blocked_grad
from BlockGRU import BlockGRU

'''
Core blocks module.  Takes:
    input: (ts, mb, h)
    hx: (ts, mb, h)
    cx: (ts, mb, h)

    output:
    output, hx, cx
'''

class BlocksCore(nn.Module):

    def __init__(self, ninp, nhid, num_blocks_in, num_blocks_out, topkval, step_att, do_gru, num_modules_read_input=2, device=None):
        super(BlocksCore, self).__init__()
        self.nhid = nhid
        self.num_blocks_in = num_blocks_in
        self.num_blocks_out = num_blocks_out
        self.block_size_in = nhid // num_blocks_in
        self.block_size_out = nhid // num_blocks_out
        self.ninp = ninp
        self.topkval = topkval
        self.step_att = step_att
        self.do_gru = do_gru
        self.num_modules_read_input = num_modules_read_input

        # print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~Blocks Core Initialize~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
        # print("nhid: ", nhid)
        # print("num_blocks_in: ", num_blocks_in)
        # print("num_blocks_out: ", num_blocks_out)
        # print("block_size_in: ", self.block_size_in)
        # print("block_size_out: ", self.block_size_out)
        # print("topkval: ", topkval)
        # input()

        self.mha = MultiHeadAttention(n_head=4, d_model_read=self.block_size_out, d_model_write=self.block_size_out, d_model_out=self.block_size_out, d_k=16, d_v=16, num_blocks_read=self.num_blocks_out, num_blocks_write=self.num_blocks_out, topk=self.num_blocks_out, grad_sparse=False)

        self.att_out = self.block_size_out*4

        self.inp_att = MultiHeadAttention(n_head=1, d_model_read=self.block_size_out, d_model_write=ninp, d_model_out=self.att_out, d_k=64, d_v=self.att_out, num_blocks_read=num_blocks_out, num_blocks_write=num_modules_read_input,residual=False, topk=self.num_blocks_in+1, grad_sparse=False, skip_write=True)

        if do_gru:
            self.block_lstm = BlockGRU(self.att_out*self.num_blocks_out, self.nhid, k=self.num_blocks_out)
        else:
            self.block_lstm = BlockLSTM(self.att_out*self.num_blocks_out, self.nhid, k=self.num_blocks_out)

        self.device = device

    def blockify_params(self):
        self.block_lstm.blockify_params()

    def forward(self, inp, hx, cx, step,do_print=False, do_block=True):

        # print()
        # print("~~~~~~~~~~~Forward of Blocks Core~~~~~~~~~~~~~~~~")
        # print("inp, hx, cx: ", inp.size(), hx.size(), cx.size())
        
        hxl = []
        cxl = []

        inp_use = inp #layer_input[idx_step]
        batch_size = inp.shape[0]
        # print("inp size", inp.size())

        #use attention here.
        inp_use = inp_use.reshape((inp_use.shape[0], self.num_blocks_in, self.ninp))
        # print("inp_use size", inp_use.size())
        inp_use = inp_use.repeat(1,self.num_modules_read_input-1,1)
        # print("inp_use size", inp_use.size())
        inp_use = torch.cat([torch.zeros_like(inp_use[:,0:1,:]), inp_use], dim=1)
        # print("inp_use size", inp_use.size1())

        inp_use, iatt, _ = self.inp_att(hx.reshape((hx.shape[0], self.num_blocks_out, self.block_size_out)), inp_use, inp_use)
        # print('inp_use', inp_use.size())
        inp_use = inp_use.reshape((inp_use.shape[0], self.att_out*self.num_blocks_out))
        # print('inp_use', inp_use.size())
        # print('iatt', iatt.size())

        null_score = iatt.mean((0,1))[1]

        new_mask = torch.ones_like(iatt[:,:,0])

        bottomk_indices = torch.topk(iatt[:,:,0], dim=1,
                                sorted=True, largest=True,
                                k = self.num_blocks_out - self.topkval)[1]

        print(null_score.size(), new_mask.size(), bottomk_indices.size())
        new_mask.index_put_((torch.arange(bottomk_indices.size(0)).unsqueeze(1), bottomk_indices),
                                                torch.zeros_like(bottomk_indices[0], dtype=new_mask.dtype))

        mask = new_mask
        assert(torch.mean(torch.sum(mask, dim=1)).item() == self.topkval)

        mask = mask.reshape((inp_use.shape[0],self.num_blocks_out,1)).repeat((1,1,self.block_size_out)).reshape((inp_use.shape[0], self.num_blocks_out*self.block_size_out))
        mask = mask.detach()
        # print("mask", mask.size())

        hx_old = hx*1.0
        cx_old = cx*1.0


        if self.do_gru:
            hx_new = self.block_lstm(inp_use, hx)
            cx_new = hx_new
        else:
            hx_new, cx_new = self.block_lstm(inp_use, hx, cx)

        #Communication b/w different Blocks
        if do_block:
            if self.step_att:
                hx_new = hx_new.reshape((hx_new.shape[0], self.num_blocks_out, self.block_size_out))
                # print("hx_new", hx_new.size())
                hx_new_grad_mask = blocked_grad.apply(hx_new, mask.reshape((mask.shape[0], self.num_blocks_out, self.block_size_out)))
                hx_new_att,attn_out,extra_loss_att = self.mha(hx_new_grad_mask,hx_new_grad_mask,hx_new_grad_mask)
                hx_new = hx_new + hx_new_att
                # print("HAAAA", hx_new_grad_mask.size(), hx_new_att.size(), attn_out.size())
                hx_new = hx_new.reshape((hx_new.shape[0], self.nhid))
                # print("hx_new reshaped", hx_new.size())
                extra_loss = extra_loss_att

        hx = (mask)*hx_new + (1-mask)*hx_old
        cx = (mask)*cx_new + (1-mask)*cx_old

        return hx, cx, mask