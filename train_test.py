import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
torch.autograd.set_detect_anomaly(True)
from torch.optim import lr_scheduler

import numpy as np
import time
import os
import math
import wandb

import helpers.utils as utils
import helpers.loggers as loggers


def train(opt, model, loader_objs, device, exp_config_dict):
    step = 0
    start_time = time.time()

    # Data loaders
    train_dataloader = loader_objs['train_dataloader']
    n_train_batches = loader_objs['n_train_batches']
    total_steps = n_train_batches * opt.epochs
    loggers.print_exp_details(opt, n_train_batches)

    if opt.offline is True: os.system('wandb offline')
    else:   os.system('wandb online')

    if opt.off_wandb is False:
        # 1. Start a new run
        wandb.init(project=opt.wandb_project, entity=opt.wandb_entity, config=exp_config_dict)
        # 2. Save model inputs and hyperparameters
        config = wandb.config
        # 3. Log gradients and model parameters
        wandb.watch(model)

    print(f"Logging to {opt.logdir} {n_train_batches}")
    optimizer = optim.Adam(model.parameters(), lr=opt.lr)
    
    for epoch in range(opt.epochs):
        epoch_loss = 0

        for it in range(n_train_batches):   # n_train_batches steps
            pred, gt, step_loss, loss_dict = train_batch(model, train_dataloader, optimizer, opt, device)
            pred_gt = torch.cat((pred.detach().cpu(), gt.cpu()), 0).numpy()
            epoch_loss += step_loss
            step += 1

            if opt.off_wandb is False:
                # Log losses and pred, gt videos
                if step % opt.loss_log_freq == 0:
                    wandb.log(loss_dict, step=step)

                if step == 1 or step % opt.video_log_freq == 0:
                    wandb.log({ 'Pred_GT': wandb.Video(pred_gt) }, step=step)
                    print("Logged video")
                
            print(f"step {step}; Step loss {step_loss}")

            # Save model params
            utils.save_model_params(model, optimizer, epoch, opt, step, opt.ckpt_save_freq)
            
        epoch_loss /= n_train_batches # Avg loss over all batches for this epoch
        wandb.log({"Per Epoch Loss": epoch_loss})
        loggers.log_after_epoch(epoch, epoch_loss, step, start_time, total_steps, opt=opt)

def test(opt, model, loader_objs, device, exp_config_dict, step=None, metrics=None, lr_schedule=None):
    test_loss = 0
    step = 0
    pred_timesteps = opt.test_out_seq
    dataloader = loader_objs['test_dataloader']
    batches = loader_objs['n_test_batches'] 
    print(f"Testing model on {batches} batches ({opt.batch_size*batches} steps)")

    if opt.offline is True: os.system('wandb offline')
    
    if opt.off_wandb is False:
        wandb.init(project=opt.wandb_project, entity=opt.wandb_entity, config=exp_config_dict)
        config = wandb.config
        wandb.watch(model)
    
    mse_vals, ssim_vals, psnr_vals = [], [], []

    with torch.no_grad():
        model.eval()
            
        for it in range(batches):
            pred, gt, loss = test_batch(model, dataloader, opt, device) # pred, gt are in -0.5, 0.5
            test_loss += loss
            step += 1
            frame_mse = []
            frame_ssim = []
            frame_psnr = []

            for t in range(pred_timesteps):
                pred_ = pred[:, t, :, :, :]
                gt_ = gt[:, t, :, :, :]
                mse = F.mse_loss(pred_, gt_).item()
                psnr = 10 * math.log10(1 / mse)
                pred_255, gt_255 = (pred_ + 0.5) * 255.0, (gt_ + 0.5) * 255.0
                pred_255, gt_255 = pred_255, gt_255
                ssim = utils.get_normalized_ssim(pred_255, gt_255)   # normalized by b
                frame_mse.append(mse)
                frame_psnr.append(psnr)
                frame_ssim.append(ssim)

            mse_vals.append(frame_mse)
            psnr_vals.append(frame_psnr)
            ssim_vals.append(frame_ssim)

            print(f'Testing step {step}....')

        pred_gt = torch.cat(((pred.cpu() + 0.5) * 255.0, (gt.cpu() + 0.5) * 255.0), 0).numpy()
        test_loss /= batches # avg test loss over all batches

        mse_vals = torch.FloatTensor(mse_vals).mean(dim=0)
        psnr_vals = torch.FloatTensor(psnr_vals).mean(dim=0)
        ssim_vals = torch.FloatTensor(ssim_vals).mean(dim=0)

        for i in range(pred_timesteps):
            if opt.off_wandb is False:
                step=i+1+opt.test_in_seq
                if opt.model in ['S3VAE']:
                    step = i+1
                wandb.log({"PSNR": psnr_vals[i], 
                            "MSE": mse_vals[i],  
                            "SSIM": ssim_vals[i],
                            'Pred_GT': wandb.Video(pred_gt)}, step=step)

        avg_mse, avg_psnr, avg_ssim = mse_vals[-1], psnr_vals[-1], ssim_vals[-1]
        loggers.log_final_test_metrics(test_loss, avg_mse, avg_psnr, avg_ssim, opt.id)    # Logs final MSE, PSNR, SSIM

def test_batch(model, test_dataloader, opt, device):
    data_dict = utils.get_data_dict(test_dataloader)
    batch_dict = utils.get_next_batch(data_dict, opt)
    input_frames = batch_dict['observed_data'].to(device)   # [-0.5, 0.5]
    ground_truth = batch_dict['data_to_predict'].to(device) # [-0.5, 0.5]
    
    # inputs and gt in [0, 1]
    input_frames, ground_truth = (input_frames + 0.5).to(device), (ground_truth + 0.5).to(device)
    predicted_frames = model.get_prediction(input_frames, batch_dict=batch_dict)
    
    if opt.model in ['ConvGRU', 'ODEConv']:  # preds are [0, 1]
        loss = model.get_loss(predicted_frames, ground_truth)
        
    elif opt.model in ['S3VAE']:
        loss_function = nn.MSELoss().cuda()
        b, t, c, h, w = predicted_frames.size()
        ground_truth = torch.cat((input_frames, ground_truth), 1)
        loss = loss_function(predicted_frames.reshape(b*t, c, h, w), ground_truth.reshape(b*t, c, h, w)) 
    
    # Convert to [-0.5, 0.5] range
    ground_truth, predicted_frames = (ground_truth - 0.5).to(device), (predicted_frames - 0.5).to(device) 

    return predicted_frames, ground_truth, loss

def train_batch(model, train_dataloader, optimizer, opt, device):
    # Get batch data & Get input sequence and output ground truth 
    data_dict = utils.get_data_dict(train_dataloader)
    batch_dict = utils.get_next_batch(data_dict, opt)
    input_frames = batch_dict['observed_data'].to(device)   # [-0.5, 0.5]
    ground_truth = batch_dict['data_to_predict'].to(device) # [-0.5, 0.5]
    loss_dict = {}
    optimizer.zero_grad()
    
    # change input_frames and ground_truth from [-0.5, 0.5] to [0, 1]
    input_frames, ground_truth = (input_frames + 0.5).to(device), (ground_truth + 0.5).to(device) 
    predicted_frames = model.get_prediction(input_frames, batch_dict=batch_dict)
    
    if opt.model in ['S3VAE']:  
        train_loss, loss_dict = model.get_loss()
        train_loss.backward()
        optimizer.step()
        return predicted_frames * 255.0, input_frames * 255.0, train_loss.item(), loss_dict

    else: 
        train_loss = model.get_loss(predicted_frames, ground_truth)
        train_loss.backward()
        optimizer.step()
        loss_dict = {'Per Step Loss': train_loss.item()}
        return predicted_frames * 255.0, ground_truth * 255.0, train_loss.item(), loss_dict
