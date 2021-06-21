import sys
sys.path.append('..')

import torch
import torch.nn as nn

from modules.DiffEqSolver import ODEFunc, DiffEqSolver
from modules.ODEConvGRUCell import ODEConvGRUCell

import helpers.utils as utils

class Encoder(nn.Module):
    
    def __init__(self, input_dim=3, ch=64, n_downs=2, device=None):
        super(Encoder, self).__init__()
        cnn_encoder = [nn.Conv2d(input_dim, ch, 3, 1, 1), nn.BatchNorm2d(ch), nn.ReLU()]
        
        for _ in range(n_downs):
            cnn_encoder += [nn.Conv2d(ch, ch * 2, 4, 2, 1), nn.BatchNorm2d(ch * 2), nn.ReLU()]
            ch *= 2
        
        self.cnn_encoder = nn.Sequential(*cnn_encoder).to(device) # CNN Embedding

    def forward(self, x):
        out = self.cnn_encoder(x)
        return out

class Decoder(nn.Module):
    def __init__(self, input_dim=256, output_dim=3, n_ups=2, device=None):
        super(Decoder, self).__init__()
        model = []
        ch = input_dim

        for _ in range(n_ups):
            model += [nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)]
            model += [nn.Conv2d(ch, ch // 2, 3, 1, 1), nn.BatchNorm2d(ch // 2), nn.ReLU()]
            ch = ch // 2
        
        model += [nn.Conv2d(ch, output_dim, 3, 1, 1)]
        self.cnn_decoder = nn.Sequential(*model).to(device)
    
    def forward(self, x):
        out = self.cnn_decoder(x)
        return out

class VidODE(nn.Module):
    def __init__(self, opt, device):
        super(VidODE, self).__init__()
        print("init VidODE")
        
        self.opt = opt
        self.device = device
        self.resize = 2 ** opt.n_downs
        h, w = opt.resolution, opt.resolution
        resolution_after_encoder = (h // self.resize, w // self.resize)
        ch = 32
        in_dim, out_dim = ch * self.resize, ch * self.resize

        self.conv_encoder = Encoder(input_dim=opt.in_channels, ch=ch, n_downs=opt.n_downs, device=device).to(device)

        self.ode_encoder_func = ODEFunc(n_inputs=in_dim, 
                                            n_outputs=out_dim, 
                                            n_layers=opt.n_layers, 
                                            n_units=in_dim // 2,
                                            downsize=False,
                                            nonlinear='relu',
                                            final_act=False,
                                            device=device)

        self.encoder_z0 = ODEConvGRUCell(self.ode_encoder_func, 
                                                    opt, 
                                                    resolution_after_encoder, 
                                                    out_dim, 
                                                    device=device)
        
        # Init decoder neural ODE with a convnet
        self.ode_decoder_func = ODEFunc(n_inputs=in_dim, 
                            n_outputs=out_dim, 
                            n_layers=opt.n_layers, 
                            n_units=in_dim // 2,
                            downsize=False,
                            nonlinear='relu',
                            final_act=False,
                            device=device)
        
        self.diffeq_solver = DiffEqSolver(self.ode_decoder_func, opt.decode_diff_method, device=device, memory=False)
        
        self.conv_decoder = Decoder(out_dim * 2, opt.in_channels + 3, opt.n_downs, device=device).to(device)

    
    def forward(self, inputs, batch_dict):
        
        b, t, c, h, w = inputs.size()
        observed_tp = batch_dict['observed_tp']
        time_steps_to_predict = batch_dict['tp_to_predict']
        mask = batch_dict["observed_mask"].to(self.device)
        out_mask = batch_dict["mask_predicted_data"].to(self.device)
        pred_t_len = len(time_steps_to_predict)

        # 1. ConvEncode the input frames
        encoded_inputs = self.conv_encoder(inputs.view(b*t, c, h, w))
        encoded_inputs = encoded_inputs.view(b, -1, encoded_inputs.size()[-3], encoded_inputs.size()[-2], encoded_inputs.size()[-1])

        # 2. ODEConvGRUCell to predict (first_point_mu, first_point_std) for z_0
        first_point_mu, first_point_std = self.encoder_z0(encoded_inputs, observed_tp, mask)
        first_point_enc = first_point_mu.unsqueeze(0).repeat(1, 1, 1, 1, 1).squeeze(0)

        # 3. ODE decoding: Given first_point_enc (z_0) and time_steps_to_predict([t_i,....t_(i+n)]), we predict sol_y([z_i,...z_(i+n)])
        sol_y = self.diffeq_solver(first_point_enc, time_steps_to_predict)
        sol_y = sol_y.contiguous().view(b, pred_t_len, -1, h // self.resize, w // self.resize)  

        skip_conn_embed = self.conv_encoder(inputs[:, -1, ...])
        pred_outputs = self.get_flowmaps(sol_out=sol_y, first_prev_embed=skip_conn_embed, mask=out_mask)
        pred_outputs = torch.cat(pred_outputs, dim=1)

        pred_flows, pred_intermediates, pred_masks = \
            pred_outputs[:, :, :2, ...], pred_outputs[:, :, 2:2+self.opt.in_channels, ...], torch.sigmoid(pred_outputs[:, :, 2+self.opt.in_channels:, ...])
        
        ### Warping first frame by using optical flow
        # Declare grid for warping
        grid_x = torch.linspace(-1.0, 1.0, w).view(1, 1, w, 1).expand(b, h, -1, -1)
        grid_y = torch.linspace(-1.0, 1.0, h).view(1, h, 1, 1).expand(b, -1, w, -1)
        grid = torch.cat([grid_x, grid_y], 3).float().to(self.device)  # [b, h, w, 2]
        
        # Warping
        last_frame = inputs[:, -1, ...] 
        warped_pred_x = self.get_warped_images(pred_flows=pred_flows, start_image=last_frame, grid=grid)
        warped_pred_x = torch.cat(warped_pred_x, dim=1)  # regular b, t, 6, h, w / irregular b, t * ratio, 6, h, w

        pred_x = pred_masks * warped_pred_x + (1 - pred_masks) * pred_intermediates
        pred_x = pred_x.view(b, -1, c, h, w)

        ### extra information
        extra_info = {}

        extra_info["optical_flow"] = pred_flows
        extra_info["warped_pred_x"] = warped_pred_x
        extra_info["pred_intermediates"] = pred_intermediates
        extra_info["pred_masks"] = pred_masks

        return pred_x, extra_info

    def get_flowmaps(self, sol_out, first_prev_embed, mask):
        b, _, c, h, w = sol_out.size()
        pred_time_steps = int(mask[0].sum())
        pred_flows = list()
    
        prev = first_prev_embed.clone()
        time_iter = range(pred_time_steps)
        
        if mask.size(1) == sol_out.size(1):
            sol_out = sol_out[mask.squeeze(-1).byte()].view(b, pred_time_steps, c, h, w)
        
        for t in time_iter:
            cur_and_prev = torch.cat([sol_out[:, t, ...], prev], dim=1)
            pred_flow = self.conv_decoder(cur_and_prev).unsqueeze(1)
            pred_flows += [pred_flow]
            prev = sol_out[:, t, ...].clone()
    
        return pred_flows

    def get_warped_images(self, pred_flows, start_image, grid):
        """ Get warped images recursively
        Input:
            pred_flows - Predicted flowmaps to use (b, time_steps_to_predict, c, h, w)
            start_image- Start image to warp
            grid - pre-defined grid

        Output:
            pred_x - List of warped (b, time_steps_to_predict, c, h, w)
        """
        warped_time_steps = pred_flows.size(1)
        pred_x = list()
        last_frame = start_image
        b, _, c, h, w = pred_flows.shape
        
        for t in range(warped_time_steps):
            pred_flow = pred_flows[:, t, ...]           # b, 2, h, w
            pred_flow = torch.cat([pred_flow[:, 0:1, :, :] / ((w - 1.0) / 2.0), pred_flow[:, 1:2, :, :] / ((h - 1.0) / 2.0)], dim=1)
            pred_flow = pred_flow.permute(0, 2, 3, 1)   # b, h, w, 2
            flow_grid = grid.clone() + pred_flow.clone()# b, h, w, 2
            warped_x = nn.functional.grid_sample(last_frame, flow_grid, padding_mode="border")
            pred_x += [warped_x.unsqueeze(1)]           # b, 1, 3, h, w
            last_frame = warped_x.clone()
        
        return pred_x 

    def get_prediction(self, input_frames, batch_dict):
        pred_x, extra_info = self(input_frames, batch_dict)
        self.extra_info = extra_info
        self.batch_dict = batch_dict
        return pred_x

    def get_mse(self, truth, pred_x, mask=None):
        b, _, c, h, w = truth.size()
        if mask is None:
            selected_time_len = truth.size(1)
            selected_truth = truth
        else:
            selected_time_len = int(mask[0].sum())
            selected_truth = truth[mask.squeeze(-1).byte()].view(b, selected_time_len, c, h, w)
        loss = torch.sum(torch.abs(pred_x - selected_truth)) / (b * selected_time_len * c * h * w)
        return loss

    def get_diff(self, data, mask=None):
        data_diff = data[:, 1:, ...] - data[:, :-1, ...]
        b, _, c, h, w = data_diff.size()
        selected_time_len = int(mask[0].sum())
        masked_data_diff = data_diff[mask.squeeze(-1).byte()].view(b, selected_time_len, c, h, w)
        return masked_data_diff

    def get_loss(self, pred_frames, truth, loss='MSE'):
        init_image = self.batch_dict["observed_data"][:, -1, ...]
        data = torch.cat([init_image.unsqueeze(1), self.batch_dict["data_to_predict"]], dim=1)
        data_diff = self.get_diff(data=data, mask=self.batch_dict["mask_predicted_data"])
        
        # batch-wise mean
        loss = torch.mean(self.get_mse(truth=self.batch_dict["data_to_predict"],
                                       pred_x=pred_frames,
                                       mask=self.batch_dict["mask_predicted_data"]))
        
        loss += torch.mean(self.get_mse(truth=data_diff, 
                                                pred_x=self.extra_info["pred_intermediates"], 
                                                mask=None))

        loss = torch.mean(loss)
        return loss