import torch
import torch.nn as nn

from models.base_conv_gru import *
from models.ode_func import ODEFunc, DiffeqSolver
from models.layers import create_convnet


class VidODE(nn.Module):
    
    def __init__(self, opt, device):
        super(VidODE, self).__init__()
        
        self.opt = opt
        self.device = device
        
        # initial function
        self.build_model()
        
        # tracker
        self.tracker = utils.Tracker()

    def build_model(self):
        
        # channels for encoder, ODE, init decoder
        init_dim = self.opt.init_dim
        resize = 2 ** self.opt.n_downs
        base_dim = init_dim * resize
        input_size = (self.opt.input_size // resize, self.opt.input_size // resize)
        ode_dim = base_dim
        slot_dim = self.opt.dim
        
        print(f"Building models... base_dim:{base_dim}")
        
        ##### Conv Encoder
        self.encoder = Encoder(input_dim=self.opt.input_dim,
                               ch=init_dim,
                               n_downs=self.opt.n_downs,
                               opt=self.opt, device=self.device).to(self.device)
        
        # Get CNN Encoder and Decoder
        ode_func_netE, ode_func_netD = self.set_ode_func_netED() # creates a CNN Encoding
        
        ##### ODE Encoder
        rec_ode_func = ODEFunc(opt=self.opt,
                               input_dim=ode_dim,
                               latent_dim=base_dim,  # channels after encoder, & latent dimension
                               ode_func_net=ode_func_netE,
                               device=self.device).to(self.device)
        
        z0_diffeq_solver = DiffeqSolver(base_dim,
                                        ode_func=rec_ode_func,
                                        method="euler",
                                        latents=base_dim,
                                        odeint_rtol=1e-3,
                                        odeint_atol=1e-4,
                                        device=self.device,
                                        nru=self.opt.nru,
                                        nru2=self.opt.nru2)
        
        self.encoder_z0 = self.set_Encoder_z0_ODE_ConvGRU(z0_diffeq_solver)
        
        ##### ODE Decoder
        gen_ode_func = ODEFunc(opt=self.opt,
                               input_dim=ode_dim,
                               latent_dim=base_dim,
                               ode_func_net=ode_func_netD,
                               device=self.device).to(self.device)
        
        self.diffeq_solver = DiffeqSolver(base_dim,
                                          gen_ode_func,
                                          self.opt.dec_diff, base_dim,
                                          odeint_rtol=1e-3,
                                          odeint_atol=1e-4,
                                          device=self.device,
                                          nru=self.opt.nru,
                                          nru2=self.opt.nru2)
        
        ##### Conv Decoder
        if self.opt.slot_attention is False:
            self.decoder = Decoder(input_dim=base_dim * 2, output_dim=self.opt.input_dim + 3, n_ups=self.opt.n_downs, opt=self.opt).to(self.device)

        elif self.opt.slot_attention is True:
            if self.opt.pos == 1:
                pass
            elif self.opt.pos == 2:
                self.decoder = Decoder(input_dim=slot_dim * 2, 
                                        output_dim=self.opt.input_dim + 3 + 1, # 3 channels for image, 3 channels for flow, image diff, warp, final channel is alpha mask for slot attention
                                        n_ups=self.opt.n_downs, 
                                        opt=self.opt).to(self.device)
    
    def set_ode_func_netED(self):
        
        init_dim = self.opt.init_dim
        num_slots = self.opt.num_slots
        resize = 2 ** self.opt.n_downs
        base_dim = init_dim * resize
        input_size = (self.opt.input_size // resize, self.opt.input_size // resize)
        ode_dim = base_dim
        slot_dim = self.opt.dim

        if self.opt.slot_attention is False:
            ode_func_netE = create_convnet(n_inputs=ode_dim,
                                            n_outputs=base_dim,
                                            n_layers=self.opt.n_layers,
                                            n_units=base_dim // 2).to(self.device)
            
            ode_func_netD = create_convnet(n_inputs=ode_dim,
                                            n_outputs=base_dim,
                                            n_layers=self.opt.n_layers,
                                            n_units=base_dim // 2).to(self.device)
            
            return ode_func_netE, ode_func_netD
        
        elif self.opt.slot_attention is True:
            if self.opt.pos == 1:
                pass
            
            elif self.opt.pos == 2:
                ode_func_netE = create_convnet(n_inputs=slot_dim,
                                                n_outputs=slot_dim,
                                                n_layers=self.opt.n_layers,
                                                n_units=slot_dim).to(self.device)

                ode_func_netD = create_convnet(n_inputs=slot_dim,
                                                n_outputs=slot_dim,
                                                n_layers=self.opt.n_layers,
                                                n_units=slot_dim).to(self.device)
                
                return ode_func_netE, ode_func_netD

        return None, None
    
    def set_Encoder_z0_ODE_ConvGRU(self, z0_diffeq_solver):
        
        init_dim = self.opt.init_dim
        resize = 2 ** self.opt.n_downs
        base_dim = init_dim * resize
        input_size = (self.opt.input_size // resize, self.opt.input_size // resize)
        ode_dim = base_dim
        num_slots = self.opt.num_slots
        slot_dim = self.opt.dim
        encoder_z0 = None

        encoder_z0 = Encoder_z0_ODE_ConvGRU(input_size=input_size,
                                                input_dim=base_dim,
                                                hidden_dim=base_dim,
                                                kernel_size=(3, 3),
                                                num_layers=self.opt.n_layers,
                                                dtype=torch.cuda.FloatTensor if self.device == 'cuda' else torch.FloatTensor,
                                                batch_first=True,
                                                bias=True,
                                                return_all_layers=True,
                                                z0_diffeq_solver=z0_diffeq_solver,
                                                run_backwards=self.opt.run_backwards,
                                                opt=self.opt).to(self.device)

        if self.opt.slot_attention is True:
            if self.opt.pos == 1:
                pass
            
            elif self.opt.pos == 2:
                encoder_z0 = Encoder_z0_ODE_ConvGRU(input_size=input_size,
                                                    input_dim=slot_dim,
                                                    hidden_dim=slot_dim,
                                                    kernel_size=(3, 3),
                                                    num_layers=self.opt.n_layers,
                                                    dtype=torch.cuda.FloatTensor if self.device == 'cuda' else torch.FloatTensor,
                                                    batch_first=True,
                                                    bias=True,
                                                    return_all_layers=True,
                                                    z0_diffeq_solver=z0_diffeq_solver,
                                                    run_backwards=self.opt.run_backwards).to(self.device)

        return encoder_z0

    def get_reconstruction(self, time_steps_to_predict, truth, truth_time_steps, mask=None, out_mask=None):
        
        truth = truth.to(self.device)
        num_slots = self.opt.num_slots
        slot_dim = self.opt.dim
        D = int(slot_dim ** 0.5)
        truth_time_steps = truth_time_steps.to(self.device)
        mask = mask.to(self.device)
        out_mask = out_mask.to(self.device)
        time_steps_to_predict = time_steps_to_predict.to(self.device)
        
        resize = 2 ** self.opt.n_downs
        b, t, c, h, w = truth.shape
        pred_t_len = len(time_steps_to_predict)
        
        ##### Skip connection forwarding
        skip_image = truth[:, -1, ...] if self.opt.extrap else truth[:, 0, ...]
        skip_conn_embed = self.encoder(skip_image)
        
        if self.opt.slot_attention is True:
            skip_conn_embed = skip_conn_embed.view(b, num_slots, -1, h // resize, w // resize).permute(1, 0, 2, 3, 4)

        elif self.opt.slot_attention is False:
            skip_conn_embed = skip_conn_embed.view(b, -1, h // resize, w // resize)
            
        ##### Conv encoding
        # print("Encoding e_truth")
        e_truth = self.encoder(truth.view(b * t, c, h, w))

        if self.opt.slot_attention is True:
            # print("e_truth (After encoder): ", e_truth.size(), "reshaped to", e_truth.view(b * num_slots, t, -1, h // resize, w // resize).size())
            e_truth = e_truth.view(b * num_slots, t, -1, h // resize, w // resize)

        elif self.opt.slot_attention is False:
            # print("e_truth (After encoder): ", e_truth.size(), "reshaped to", e_truth.view(b, t, -1, h // resize, w // resize).size())
            e_truth = e_truth.view(b, t, -1, h // resize, w // resize)
        
        ##### ODE encoding
        first_point_mus, first_point_stds, first_point_encs = [], [], []
        sol_ys, pred_outputs_s = [], []
        pred_flows_s, pred_intermediates_s, pred_masks_s, pred_xs, warped_pred_xs = [], [], [], [], []

        if self.opt.slot_attention is True:
            e_truths = e_truth.view(b, num_slots, t, -1, h // resize, w // resize).permute(1, 0, 2, 3, 4, 5)

            # Sampling latent features: e_truth for each slot
            for slot_e_truth, slot_skip_conn_embed in zip(e_truths, skip_conn_embed):

                first_point_mu, first_point_std = self.encoder_z0(input_tensor=slot_e_truth, time_steps=truth_time_steps, mask=mask, tracker=self.tracker)
                first_point_mus.append(first_point_mu)
                first_point_stds.append(first_point_std)

                first_point_enc = first_point_mu.unsqueeze(0).repeat(1, 1, 1, 1, 1).squeeze(0)
                first_point_encs.append(first_point_enc)
                
                # ODE decoding
                sol_y = self.diffeq_solver(first_point_enc, time_steps_to_predict)
                sol_ys.append(sol_y)
                
                # Conv decoding
                sol_y = sol_y.contiguous().view(b, pred_t_len, -1, h // resize, w // resize)
                # print("got sol_y", slot_skip_conn_embed.size(), sol_y.size(), out_mask.size())
                # regular b, t, 6, h, w / irregular b, t * ratio, 6, h, w
                pred_outputs = self.get_flowmaps(sol_out=sol_y, first_prev_embed=slot_skip_conn_embed, mask=out_mask) # b, t, 6, h, w
                pred_outputs = torch.cat(pred_outputs, dim=1)
                pred_outputs_s.append(pred_outputs)
                # print("pred_outputs", pred_outputs.size())

                pred_flows, pred_intermediates, pred_masks = \
                pred_outputs[:, :, :2, ...], pred_outputs[:, :, 2:2+self.opt.input_dim, ...], torch.sigmoid(pred_outputs[:, :, 2+self.opt.input_dim:, ...])

                pred_flows_s.append(pred_flows)
                pred_intermediates_s.append(pred_intermediates)
                pred_masks_s.append(pred_masks)

                ### Warping first frame by using optical flow
                # Declare grid for warping
                grid_x = torch.linspace(-1.0, 1.0, w).view(1, 1, w, 1).expand(b, h, -1, -1)
                grid_y = torch.linspace(-1.0, 1.0, h).view(1, h, 1, 1).expand(b, -1, w, -1)
                grid = torch.cat([grid_x, grid_y], 3).float().to(self.device)  # [b, h, w, 2]

                # Warping
                last_frame = truth[:, -1, ...] if self.opt.extrap else truth[:, 0, ...]
                warped_pred_x = self.get_warped_images(pred_flows=pred_flows, start_image=last_frame, grid=grid)
                warped_pred_x = torch.cat(warped_pred_x, dim=1)  # regular b, t, 6, h, w / irregular b, t * ratio, 6, h, w
                warped_pred_xs.append(warped_pred_x)

                pred_x = pred_masks * warped_pred_x + (1 - pred_masks) * pred_intermediates
                pred_x = pred_x.view(b, -1, c, h, w)
                pred_xs.append(pred_x)

            first_point_mus = torch.stack(first_point_mus)
            first_point_stds = torch.stack(first_point_stds)
            first_point_encs = torch.stack(first_point_encs)
            sol_ys = torch.stack(sol_ys)
            self.tracker.write_info(key="sol_ys", value=sol_ys.clone().cpu())
            pred_outputs_s = torch.stack(pred_outputs_s)

            pred_flows_s = torch.stack(pred_flows_s)
            pred_intermediates_s = torch.stack(pred_intermediates_s)
            pred_masks_s = torch.stack(pred_masks_s)
            warped_pred_xs = torch.stack(warped_pred_xs)
            pred_xs = torch.stack(pred_xs)

            ### extra information
            extra_info = {}

            extra_info["optical_flow"] = pred_flows_s
            extra_info["warped_pred_x"] = warped_pred_xs
            extra_info["pred_intermediates"] = pred_intermediates_s
            extra_info["pred_masks"] = pred_masks_s

            return pred_xs, extra_info

        else:
            # Sampling latent features
            first_point_mu, first_point_std = self.encoder_z0(input_tensor=e_truth, time_steps=truth_time_steps, mask=mask, tracker=self.tracker)
            first_point_enc = first_point_mu.unsqueeze(0).repeat(1, 1, 1, 1, 1).squeeze(0)

            # ODE decoding
            sol_y = self.diffeq_solver(first_point_enc, time_steps_to_predict)
            self.tracker.write_info(key="sol_y", value=sol_y.clone().cpu())
            
            # Conv decoding
            sol_y = sol_y.contiguous().view(b, pred_t_len, -1, h // resize, w // resize)
            # regular b, t, 6, h, w / irregular b, t * ratio, 6, h, w
            # print("got sol_y", skip_conn_embed.size(), sol_y.size(), out_mask.size())
            pred_outputs = self.get_flowmaps(sol_out=sol_y, first_prev_embed=skip_conn_embed, mask=out_mask) # b, t, 6, h, w
            pred_outputs = torch.cat(pred_outputs, dim=1)
            # print("pred_outputs", pred_outputs.size())
            pred_flows, pred_intermediates, pred_masks = \
            pred_outputs[:, :, :2, ...], pred_outputs[:, :, 2:2+self.opt.input_dim, ...], torch.sigmoid(pred_outputs[:, :, 2+self.opt.input_dim:, ...])

            ### Warping first frame by using optical flow
            # Declare grid for warping
            grid_x = torch.linspace(-1.0, 1.0, w).view(1, 1, w, 1).expand(b, h, -1, -1)
            grid_y = torch.linspace(-1.0, 1.0, h).view(1, h, 1, 1).expand(b, -1, w, -1)
            grid = torch.cat([grid_x, grid_y], 3).float().to(self.device)  # [b, h, w, 2]

            # Warping
            last_frame = truth[:, -1, ...] if self.opt.extrap else truth[:, 0, ...]
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

    def export_infos(self):
        infos = self.tracker.export_info()
        self.tracker.clean_info()
        return infos
    
    def get_flowmaps(self, sol_out, first_prev_embed, mask):
        """ Get flowmaps recursively
        Input:
            sol_out - Latents from ODE decoder solver (b, time_steps_to_predict, c, h, w)
            first_prev_embed - Latents of last frame (b, c, h, w)
        
        Output:
            pred_flows - List of predicted flowmaps (b, time_steps_to_predict, c, h, w)
        """
        b, _, c, h, w = sol_out.size()
        pred_time_steps = int(mask[0].sum())
        pred_flows = list()
    
        prev = first_prev_embed.clone()
        time_iter = range(pred_time_steps)
        
        if mask.size(1) == sol_out.size(1):
            sol_out = sol_out[mask.squeeze(-1).byte()].view(b, pred_time_steps, c, h, w)
        
        for t in time_iter:
            cur_and_prev = torch.cat([sol_out[:, t, ...], prev], dim=1)
            pred_flow = self.decoder(cur_and_prev).unsqueeze(1)
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
    
    def compute_all_losses(self, batch_dict):
        
        batch_dict["tp_to_predict"] = batch_dict["tp_to_predict"].to(self.device)
        batch_dict["observed_data"] = batch_dict["observed_data"].to(self.device)
        batch_dict["observed_tp"] = batch_dict["observed_tp"].to(self.device)
        batch_dict["observed_mask"] = batch_dict["observed_mask"].to(self.device)
        batch_dict["data_to_predict"] = batch_dict["data_to_predict"].to(self.device)
        batch_dict["mask_predicted_data"] = batch_dict["mask_predicted_data"].to(self.device)

        pred_x, extra_info = self.get_reconstruction(
            time_steps_to_predict=batch_dict["tp_to_predict"],
            truth=batch_dict["observed_data"],
            truth_time_steps=batch_dict["observed_tp"],
            mask=batch_dict["observed_mask"],
            out_mask=batch_dict["mask_predicted_data"])
        
        # batch-wise mean
        loss = torch.mean(self.get_mse(truth=batch_dict["data_to_predict"],
                                       pred_x=pred_x,
                                       mask=batch_dict["mask_predicted_data"]))

        if not self.opt.extrap:
            init_image = batch_dict["observed_data"][:, 0, ...]
        else:
            init_image = batch_dict["observed_data"][:, -1, ...]

        data = torch.cat([init_image.unsqueeze(1), batch_dict["data_to_predict"]], dim=1)
        data_diff = self.get_diff(data=data, mask=batch_dict["mask_predicted_data"])

        loss = loss + torch.mean(self.get_mse(truth=data_diff, pred_x=extra_info["pred_intermediates"], mask=None))

        results = {}
        results["loss"] = torch.mean(loss)
        results["pred_y"] = pred_x
        
        return results


# python main.py -nl 2 -b 4 -e 100 -is 10 -os 10 --unequal -d kth --extrap -p train