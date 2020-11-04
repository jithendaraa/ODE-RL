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
