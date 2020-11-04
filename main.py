# Imports
import math
import numpy as np

import torch
from torch.nn import functional as f
from torch import nn

# ODE Solvers
# 1. Euler's ODE Solver h_(t+1) = h_t + n_steps*h

def euler_ode_solve(z0, t0, t1, f):
    """
    Simplest ODE Solver: Euler's method
    """
    h_max = 0.05
    n_steps = math.ceil((abs(t1 - t0)/h_max).max().item())

    h = (t1 - t0)/n_steps
    t = t0
    z = z0

    for i_step in range(n_steps):
        z = z + h * f(z, t)
        t = t + h
    return z

class ODEF(nn.Module):
    def forward_with_grad(self, z, t, grad_outputs):
        """Compute f and a df/dz, a df/dparams, a df/dt; df/dparams is abbreviated as df/dp fro convenience"""
        
        batch_size = z.shape[0]
        out = self.forward(z, t)

        a = grad_outputs
        adfdz, adfdt, *adfdp = torch.autograd.grad(
            (out,), 
            (z, t) + tuple(self.parameters()), 
            grad_outputs=(a),
            allow_unused=True, 
            retain_graph=True
        )
        # grad method automatically sums gradients for batch items, we have to expand them back
        if adfdp is not None:
            adfdp = torch.cat([p_grad.flatten() for p_grad in adfdp]).unsqueeze(0)
            adfdp = adfdp.expand(batch_size, -1) / batch_size
        if adfdt is not None:
            adfdt = adfdt.expand(batch_size, 1) / batch_size
        return out, adfdz, adfdt, adfdp

    def flatten_parameters(self):
        param_shapes = []
        flattened_params = []
        for param in self.parameters():
            param_shapes.append(param.size())
            flattened_params.append(param.flatten())
        return torch.cat(flattened_params)
