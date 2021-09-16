import sys
sys.path.append('..')

import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from modules.GroupLinearLayer import GroupLinearLayer

class Sparse_attention(nn.Module):
    def __init__(self, top_k = 5):
        super(Sparse_attention,self).__init__()
        top_k += 1
        self.top_k = top_k

    def forward(self, attn_s):
        eps = 10e-8
        time_step = attn_s.size()[1]
        if time_step <= self.top_k:
            return attn_s   # just make everything greater than 0, and return it
        else:
            delta = torch.topk(attn_s, self.top_k, dim= 1)[0][:,-1] + eps   # get top k and return it
            delta = delta.reshape((delta.shape[0],1))  

        # normalize
        attn_w = attn_s - delta.repeat(1, time_step)
        attn_w = torch.clamp(attn_w, min = 0)
        attn_w_sum = torch.sum(attn_w, dim = 1, keepdim=True)
        attn_w_sum = attn_w_sum + eps 
        attn_w_normalize = attn_w / attn_w_sum.repeat(1, time_step)
        return attn_w_normalize

class blocked_grad(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, mask):
        ctx.save_for_backward(x, mask)
        return x

    @staticmethod
    def backward(ctx, grad_output):
        x, mask = ctx.saved_tensors
        return grad_output * mask, mask * 0.0

class Sparse_grad_attention(torch.autograd.Function):
    @staticmethod
    def forward(ctx, inp, sa):
        sparsified = sa(inp)
        ctx.save_for_backward(inp, sparsified)
        return inp

    @staticmethod
    def backward(ctx, grad_output):
        inp, sparsified = ctx.saved_tensors
        return (grad_output) * (sparsified > 0.0).float()

class ScaledDotProductAttention(nn.Module):
    ''' Scaled Dot-Product Attention '''
    def __init__(self, temperature, topk, grad_sparse, attn_dropout=0.1):
        super().__init__()
        self.temperature = temperature
        self.softmax = nn.Softmax(dim=2)
        self.grad_sparse = grad_sparse
        self.topk = topk
        self.sa = Sparse_attention(top_k=topk) #k=2

    def forward(self, q, k, v, mask=None):
        attn = torch.bmm(q, k.transpose(1, 2))
        attn = attn / self.temperature

        if mask is not None:
            attn = attn.masked_fill(mask, -np.inf)

        attn = self.softmax(attn)
        extra_loss = 0.0
        use_sparse = True

        if use_sparse:
            mb, ins, outs = attn.shape[0], attn.shape[1], attn.shape[2]
            sparse_attn = attn.reshape((mb*ins, outs))
            if self.grad_sparse:
                sga = Sparse_grad_attention(self.topk)
                sparse_attn = sga(sparse_attn)
            else:
                sparse_attn = self.sa(sparse_attn)
            sparse_attn = sparse_attn.reshape((mb,ins,outs))
            attn = sparse_attn*1.0

        output = torch.bmm(attn, v)

        return output, attn, extra_loss
    
class MultiHeadAttention(nn.Module):
    ''' Multi-Head Attention module '''
    def __init__(self, n_head, d_model_read, d_model_write, d_model_out, d_k, d_v, num_blocks_read, num_blocks_write, topk, grad_sparse, residual=True, dropout=0.1, skip_write=False):
        super().__init__()

        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v

        self.GLN_qs = GroupLinearLayer(d_model_read, n_head * d_k, num_blocks_read)
        self.GLN_ks = GroupLinearLayer(d_model_write, n_head * d_k, num_blocks_write)
        self.GLN_vs = GroupLinearLayer(d_model_write, n_head * d_v, num_blocks_write)

        self.residual = residual
        self.attention = ScaledDotProductAttention(temperature=np.power(d_k, 0.5), topk=topk, grad_sparse=grad_sparse)
        self.gate_fc = nn.Linear(n_head * d_v, d_model_out)

        if not skip_write:
            self.fc = nn.Linear(n_head * d_v, d_model_out)
        else:
            self.fc = lambda a: a

        self.dropout = nn.Dropout(dropout)
        self.ln = nn.LayerNorm(d_model_out)

    def forward(self, q, k, v, mask=None):
        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head
        sz_b, len_q, _ = q.size()
        sz_b, len_k, _ = k.size()
        sz_b, len_v, _ = v.size()

        q = self.GLN_qs(q).view(sz_b, len_q, n_head, d_k)
        k = self.GLN_ks(k).view(sz_b, len_k, n_head, d_k)
        v = self.GLN_vs(v).view(sz_b, len_v, n_head, d_v)

        q = q.permute(2, 0, 1, 3).contiguous().view(-1, len_q, d_k) # (n*b) x lq x dk
        k = k.permute(2, 0, 1, 3).contiguous().view(-1, len_k, d_k) # (n*b) x lk x dk
        v = v.permute(2, 0, 1, 3).contiguous().view(-1, len_v, d_v) # (n*b) x lv x dv

        output, attn, extra_loss = self.attention(q, k, v, mask=None)
        output = output.view(n_head, sz_b, len_q, d_v)
        output = output.permute(1, 2, 0, 3).contiguous().view(sz_b, len_q, -1) # b x lq x (n*dv)

        output_init = output*1.0
        output = self.dropout(self.fc(output_init))
        gate = torch.sigmoid(self.gate_fc(output_init))

        if self.residual:
            output = gate * torch.tanh(output)
        else:
            pass

        return output, attn, extra_loss