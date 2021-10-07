import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

def build_grid(resolution):
    ranges = [np.linspace(0., 1., num=res) for res in resolution]
    grid = np.meshgrid(*ranges, sparse=False, indexing="ij")
    grid = np.stack(grid, axis=0)
    grid = np.reshape(grid, [-1, resolution[0], resolution[1]])
    grid = np.expand_dims(grid, axis=0).astype(np.float32)
    res = np.concatenate([grid, 1.0 - grid], axis=-3)
    res = torch.from_numpy(res)
    return res

def spatial_flatten(x):
    if len(x.size()) < 4: return x
    bt, c, h, w = x.size()
    return x.view(bt, c, h*w)

def spatial_broadcast(slots, resolution):
  """Broadcast slot features to a 2D grid and collapse slot dimension."""
  # `slots` has shape: [batch_size, num_slots, slot_size].
  b, num_slots, slot_size = slots.size()
  slots = slots.reshape(-1, slot_size)[:, None, None, :]
  grid = slots.repeat(1, resolution[0], resolution[1], 1)   # `grid` has shape: [batch_size*num_slots, width, height, slot_size].
  return grid

class SoftPositionEmbed(nn.Module):
    """Adds soft positional embedding with learnable projection."""
    def __init__(self, hidden_size, resolution, device):
        """Builds the soft position embedding layer.

        Args:
          hidden_size: Size of input feature dimension.
          resolution: Tuple of integers specifying width and height of grid.
        """
        super(SoftPositionEmbed, self).__init__()
        self.resolution = resolution
        self.dense = nn.Sequential(
            nn.Linear(4, hidden_size, bias=True).to(device)
        ).to(device)
        self.grid = build_grid(resolution).to(device)
        print("Made grid for ", resolution)

    def forward(self, inputs):
        self.grid = self.grid.permute(0, 2, 3, 1) # channel dim last
        dense_grid = self.dense(self.grid).permute(0, 3, 1, 2) # make channel dim = 1
        print("[inputs]", inputs.size(), dense_grid.size())
        return inputs + dense_grid

class SlotAttention(nn.Module):
    """Slot Attention module."""

    def __init__(self, opt, features, num_iterations, num_slots, slot_size, mlp_hidden_size=128,
               epsilon=1e-8, device=None, resize=4):
        """Builds the Slot Attention module.

        Args:
            features: of the input variable.
            num_iterations: Number of iterations.
            num_slots: Number of slots.
            slot_size: Dimensionality of slot feature vectors.
            mlp_hidden_size: Hidden layer size of MLP.
            epsilon: Offset for attention coefficients before normalization.
            device: device to which this class is assigned to.
    """
        super(SlotAttention, self).__init__()   
        self.opt = opt
        self.num_iterations = num_iterations
        self.num_slots = num_slots
        self.slot_size = slot_size
        self.mlp_hidden_size = mlp_hidden_size
        self.epsilon = epsilon
        self.device = device

        if self.opt.encoder in ['cgru_sa'] or self.opt.transition in ['cgru']:
            dim = (opt.resolution // resize) ** 2
            self.norm_inputs = nn.LayerNorm([dim, features]).to(device)
        else:   self.norm_inputs = nn.LayerNorm(features).to(device)
        self.norm_slots = nn.LayerNorm([self.num_slots, self.slot_size]).to(device)
        self.norm_mlp = nn.LayerNorm([self.num_slots, self.slot_size]).to(device)

        # Parameters for Gaussian init (shared by all slots).
        self.slots_mu = nn.init.xavier_uniform_(torch.empty(1, 1, self.slot_size), gain=1.0).to(device)
        self.slots_log_sigma  = nn.init.xavier_uniform_(torch.empty(1, 1, self.slot_size), gain=1.0).to(device)

        # Linear maps for the attention module.
        self.project_q = nn.Linear(self.slot_size, self.slot_size, bias=False).to(device)
        self.project_k = nn.Linear(features, self.slot_size, bias=False).to(device)
        self.project_v = nn.Linear(features, self.slot_size, bias=False).to(device)

        # Slot update functions.
        self.gru = nn.GRUCell(self.slot_size, self.slot_size).to(device)
        self.mlp = nn.Sequential(
            nn.Linear(self.slot_size, self.mlp_hidden_size),
            nn.ReLU(),
            nn.Linear(self.mlp_hidden_size, self.slot_size)
        ).to(device)


    def forward(self, x):
        # `x` has shape [batch_size, num_inputs, inputs_size].
        if self.opt.encoder in ['cgru_sa'] or self.opt.transition in ['cgru']: 
            # Make num_inputs last dim
            x = x.permute(0, 2, 1)

        # Layer norm 
        x = self.norm_inputs(x)     # Shape: [batch_size, input_size, num_inputs].
        k = self.project_k(x)       # Shape: [batch_size, num_inputs, slot_size].
        v = self.project_v(x)       # Shape: [batch_size, num_inputs, slot_size].

        if len(k.size()) == 2: k = k.view(k.size()[0], 1, k.size()[1])
        if len(v.size()) == 2: v = v.view(v.size()[0], 1, v.size()[1])

        # Initialize the slots. Shape: [batch_size, num_slots, slot_size].
        slots = self.slots_mu + torch.exp(self.slots_log_sigma) * torch.randn(x.size()[0], self.num_slots, self.slot_size).to(self.device)

        # Multiple rounds of attention.
        for _ in range(self.num_iterations):
            slots_prev = slots
            slots = self.norm_slots(slots)

            # Attention.
            q = self.project_q(slots)  # Shape: [batch_size, num_slots, slot_size].
            q = q * (self.slot_size ** -0.5)  # Normalization.
            attn_logits = torch.bmm(k, torch.transpose(q, 1, 2)) # k.qT (b, num_inputs, slot_size) x (b, slot_size, num_slots)
            attn = F.softmax(attn_logits, dim=-1)   # `attn` has shape: [batch_size, num_inputs, num_slots].

            # Weigted mean.
            attn = attn + self.epsilon
            attn = attn / torch.sum(attn, dim=-2, keepdim=True)
            updates = torch.bmm(torch.transpose(attn, 1, 2), v)   # `updates` has shape: [batch_size, num_slots, slot_size].

            # Slot update.
            updated_slots = []
            for i in range(self.num_slots):
                update = updates[:, i, :]
                prev_slot = slots_prev[:, i, :]
                updated_slot = self.gru(update, prev_slot)
                updated_slots.append(updated_slot)
            
            slots = torch.stack(updated_slots).permute(1, 0, 2)
            slots = slots + self.mlp(self.norm_mlp(slots))

        return slots

class SlotAttentionAutoEncoder(nn.Module):
    def __init__(self, opt, resolution=(64,64), num_slots=3, num_iterations=3, device=None, resize=4, broadcast=False):
        super(SlotAttentionAutoEncoder, self).__init__()
        self.opt = opt
        self.device = device
        self.resolution = resolution
        self.encoder_pos = None
        self.cgru = False
        self.decoder_initial_size = (8, 8)
        self.broadcast = broadcast
        
        # if opt.encoder in ['odecgru', 'cgru', 'cgru_sa']:
            # Position encoder and decoder
            # self.encoder_pos = SoftPositionEmbed(opt.d_zf, self.resolution, device)
            # self.decoder_pos = SoftPositionEmbed(self.opt.in_channels, self.decoder_initial_size, device)
            # print("built position encoding", self.resolution)
        if opt.encoder in ['cgru_sa'] or opt.transition in ['cgru']:
            self.layer_norm = nn.LayerNorm([resolution[0] * resolution[1], opt.d_zf]).to(device)
        
        else:
            self.layer_norm = nn.LayerNorm(opt.d_zf).to(device)            

        self.mlp = nn.Sequential(
                        nn.Linear(opt.d_zf, opt.d_zf), 
                        nn.ReLU(), 
                        nn.Linear(opt.d_zf, opt.d_zf)).to(device)
        self.slot_attention = SlotAttention(opt, opt.d_zf, num_iterations, num_slots, opt.slot_size, device=device, resize=resize).to(device)
            
    def forward(self, x):
        # print('[Slot Att. input]', x.size())
        if self.opt.encoder in ['cgru_sa'] or self.opt.transition in ['cgru']:
            x = self.conv_preprocess(x)
        else:
            x = self.default_preprocess(x)
        # Slot Attention module.
        slots = self.slot_attention(x)  # `slots` has shape: [batch_size, num_slots, slot_size].
        if self.broadcast is True:
            slots = spatial_broadcast(slots, self.decoder_initial_size).permute(0, 3, 1, 2)
        return slots


    def conv_preprocess(self, x):
        x = spatial_flatten(x)
        # print("Spatial flatten done", x.size())
        # Make features last dim, feed to layer norm
        x = self.layer_norm(x.permute(0, 2, 1))
        # print("LayerNorm done", x.size())
        # Pass through MLP and make dim 1 as features again
        x = self.mlp(x).permute(0, 2, 1)  # Feedforward network on set.
        # print("MLP done", x.size())
        return x

    def default_preprocess(self, x):
        x = self.mlp(self.layer_norm(x))
        return x

