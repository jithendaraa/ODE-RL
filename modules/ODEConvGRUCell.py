import sys
sys.path.append('../')
import helpers.utils as utils

import torch
import torch.nn as nn
from modules.ConvGRUCell import ConvGRUCell

class ODEConvGRUCell(nn.Module):
    def __init__(self, ode_func, opt, resolution, ch, out_ch=None, device=None, kernel_size=5):
        super(ODEConvGRUCell, self).__init__()

        self.ode_func = ode_func
        self.device = device
        self.z0_diffeq_solver = None
        if out_ch is None: out_ch = ch

        self.cgru_cell = ConvGRUCell(input_size=resolution, 
                                        input_dim=ch, 
                                        hidden_dim=ch,     # also the num_out_channels of the ConvGRU cell
                                        kernel_size=kernel_size,
                                        bias=True).to(device)

        # last conv layer for generating mu, sigma
        self.z0_dim = out_ch
        z = ch
        self.transform_z0 = nn.Sequential(
            nn.Conv2d(z, z, 1, 1, 0),
            nn.ReLU(),
            nn.Conv2d(z, out_ch * 2, 1, 1, 0)).to(device)
        
    def forward(self, inputs, timesteps, mask=None):    # inputs has time dim first
        last_yi, _ = self.run_ode_conv_gru(inputs, timesteps, mask=mask)
        trans_last_yi = self.transform_z0(last_yi)  # (b, self.z0_dim*2, h, w)
        mean_z0, std_z0 = torch.split(trans_last_yi, self.z0_dim, dim=1)
        std_z0 = std_z0.abs()
        return mean_z0, std_z0

    def run_ode_conv_gru(self, inputs, timesteps, run_backwards=True, mask=None):
        t, b, c, h, w = inputs.size()
        assert (t == len(timesteps)), "Sequence length should be same as time_steps"

        # Set initial inputs
        prev_input = torch.zeros((b, c, h, w)).to(self.device)

        # Run ODE backwards and combine the y(t) estimates using gating
        prev_t, t_i = timesteps[-1] + 0.01, timesteps[-1]
        latent_ys = []
        time_points_iter = range(0, timesteps.size(-1))
        if run_backwards:   
            time_points_iter = reversed(time_points_iter)

        for idx, i in enumerate(time_points_iter):
            inc = self.ode_func(prev_t, prev_input) * (t_i - prev_t)    # Integ(prev_input') from prev_t to t_i

            assert (not torch.isnan(inc).any())
            ode_sol = prev_input + inc  # next_input at t_i = prev_input(at prev_t) + Integ(prev_input') from prev_t to t_i
            ode_sol = torch.stack((prev_input, ode_sol), dim=1)  # [1, b, 2, c, h, w] => [b, 2, c, h, w]
            assert (not torch.isnan(ode_sol).any())
            
            if torch.mean(ode_sol[:, 0, :] - prev_input) >= 0.001:
                print("Error: first point of the ODE is not equal to initial value")
                print(torch.mean(ode_sol[:, :, 0, :] - prev_input))
                exit()

            yi_ode = ode_sol[:, -1, :]      # ODE estimate input at t_i esimated from prev_input + Integ(prev_input) from prev_t to t_i
            xi = inputs[i:i+1, :, :]            # Actual encoded input at t_i
            if mask is not None: m = mask[:, i]
            else: m = None
            _, yi = self.cgru_cell(input_tensor=xi, h_cur=yi_ode, seq_len=1, mask=m)
            # return to iteration
            prev_input = yi     # ODEConvGRU estimate of input at t_i: (b, c, h , w)
            prev_t, t_i = timesteps[i], timesteps[i - 1]
            latent_ys.append(yi)

        latent_ys = torch.stack(latent_ys, 0).permute(1, 0, 2, 3, 4)   # (b, t, c, h, w)
        # yi would be ODE-ConvGRU's prediction for xi at timestep 0
        return yi, latent_ys
        
