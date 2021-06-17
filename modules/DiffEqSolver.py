import sys
sys.path.append('../')

import torch
import torch.nn as nn
import helpers.utils as utils

# git clone https://github.com/rtqichen/torchdiffeq.git
from torchdiffeq import odeint as odeint


class DiffEqSolver(nn.Module):
    def __init__(self, ode_func, method, odeint_rtol=1e-4, odeint_atol=1e-5, device=torch.device("cpu"), memory=False):
        super(DiffEqSolver, self).__init__()
        
        self.ode_func = ode_func
        self.ode_method = method
        self.device = device
        self.memory = memory
        
        self.odeint_rtol = odeint_rtol
        self.odeint_atol = odeint_atol
    
    def forward(self, first_point, time_steps_to_predict, backwards=False):
        """
		# Decode the trajectory through ODE Solver
		"""
        pred_y = None
        if self.memory is True:
            y_is = [first_point]
            b, c, h, w = first_point.size()
        
            for i in range(len(time_steps_to_predict)):
                h_prev = y_is[-1]
                timestep = time_steps_to_predict[i:i+1]
                pred_m = odeint(self.ode_func, h_prev, timestep, rtol=self.odeint_rtol, atol=self.odeint_atol, method=self.ode_method)
                pred_m = pred_m.view(b, c, h, w)
                h_next = h_prev + pred_m
                y_is.append(h_next)
            
            pred_y = torch.stack(y_is[1:]).permute(1, 0, 2, 3, 4)  # => [b, t, c, h0, w0]
        
        else:
            pred_y = odeint(self.ode_func, first_point, time_steps_to_predict,
                        rtol=self.odeint_rtol, atol=self.odeint_atol, method=self.ode_method)
            pred_y = pred_y.permute(1, 0, 2, 3, 4)  # => [b, t, c, h0, w0]
        
        return pred_y


#####################################################################################################

class ODEFunc(nn.Module):
    def __init__(self, n_inputs, n_outputs, n_layers, n_units, downsize, nonlinear, device=torch.device("cpu")):
        """
		input_dim: dimensionality of the input
		latent_dim: dimensionality used for ODE. Analog of a continous latent state
		"""
        super(ODEFunc, self).__init__()
        self.device = device
        self.gradient_net = utils.create_convnet(n_inputs, n_outputs, n_layers, n_units, downsize, nonlinear).to(device)
    
    def forward(self, t_local, y, backwards=False):
        """
		Perform one step in solving ODE. Given current data point y and current time point t_local, returns gradient dy/dt at this time point
		t_local: current time point
		y: value at the current time point
		"""
        grad = self.gradient_net(y)     # get_ode_gradient_nn
        if backwards:
            grad = -grad
        return grad
