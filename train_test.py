import torch
import torch.optim as optim
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
from torchvision.utils import save_image

def train(opt, model, loader_objs, device, exp_config_dict):

    step = 0
    start_time = time.time()
    losses = []

    # Data loaders
    train_dataloader = loader_objs['train_dataloader']
    
    n_train_batches = loader_objs['n_train_batches']
    total_steps = n_train_batches * opt.epochs
    loggers.print_exp_details(opt, n_train_batches)

    if opt.offline is True:
        os.system('wandb offline')
    # 1. Start a new run
    run = wandb.init(project='VideoODE', entity='pranav2109', config=exp_config_dict)
    
    # 2. Save model inputs and hyperparameters
    config = wandb.config
    
    # 3. Log gradients and model parameters
    wandb.watch(model)

    print(f"Logging to {opt.logdir}")
    optimizer = optim.Adamax(model.parameters(), lr=opt.lr)
    pla_lr_scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=4, verbose=True)
    
    for epoch in range(opt.epochs):
        epoch_loss = 0
        utils.update_learning_rate(optimizer, decay_rate=0.99, lowest=opt.lr / 10)
        
        for it in range(n_train_batches):   # n_train_batches steps
            pred, gt, step_loss = train_batch(model, train_dataloader, optimizer, opt, device)
            epoch_loss += step_loss
            step += 1
            
            if step % opt.test_log_freq == 0:
                log_orig = gt.detach().cpu().numpy()
                log_pred = pred.detach().cpu().numpy()
                wandb.log( {'Predicted': wandb.Video(log_pred)}, step=step)
                wandb.log( {'Original': wandb.Video(log_orig)}, step=step)    
                wandb.log( {'Train Loss (per step)': step_loss}, step=step)

            print(f"step {step}")
            #utils.save_model_params(model, optimizer, epoch, opt, step, opt.ckpt_save_freq)    

        epoch_loss /= n_train_batches # Avg loss over all batches for this epoch
        # log epoch number and train_loss after every epoch

        wandb.log({"Train Loss (per epoch)": epoch_loss}, step=epoch)
        loggers.log_after_epoch(epoch, epoch_loss, step, start_time, total_steps, opt=opt)

def train_batch(model, train_dataloader, optimizer, opt, device):
    # Get bacth data
    data_dict = utils.get_data_dict(train_dataloader)
    batch_dict = utils.get_next_batch(data_dict, opt, opt.train_in_seq, opt.train_out_seq)

    # Get input sequence and output ground truth 
    input_frames, ground_truth = batch_dict['observed_data'].to(device), batch_dict['data_to_predict'].to(device)
    
    # Forward pass
    predicted_frames = model.get_prediction(input_frames, batch_dict=batch_dict)
    train_loss = model.get_loss(predicted_frames, ground_truth)

    # Backward pass
    optimizer.zero_grad()
    train_loss.backward()
    #torch.nn.utils.clip_grad_value_(model.parameters(), clip_value=10.0)

    # Step with optimizer
    optimizer.step()

    return ((predicted_frames/2.0) + 0.5) * 255.0, (ground_truth + 0.5) * 255.0, train_loss.item()

def test(opt, model, loader_objs, device, arg_dict, step=None, metrics=None, lr_schedule=None):

    test_loss = 0
    avg_mse = 0
    pred_timesteps = opt.test_out_seq
    avg_mses = [0] * pred_timesteps
    avg_psnrs = [0] * pred_timesteps
    avg_ssims = [0] * pred_timesteps

    # 1. Start a new run
    wandb.init(project=opt.wandb_project, entity=opt.wandb_entity)
    
    # 2. Save model inputs and hyperparameters
    config = wandb.config
    for key, val in arg_dict.items():
      setattr(config, key, val)

    dataloader = loader_objs['test_dataloader']
    batches = loader_objs['n_test_batches'] 
    print(f"Testing model on {batches} batches ({opt.batch_size*batches} steps)")


    with torch.no_grad():
        model.eval()
        if step is None:
            step = 0

        for it in range(batches):
            q = time.time()
            data_dict = utils.get_data_dict(dataloader)
            batch_dict = utils.get_next_batch(data_dict, opt)

            input_frames = batch_dict['observed_data'].to(device)
            ground_truth = batch_dict['data_to_predict'].to(device)
            b = ground_truth.size()[0]

            predicted_frames = model.get_prediction(input_frames)
            loss = model.get_loss(predicted_frames, ground_truth)
            test_loss += loss.item()
            step += 1
            _, _, c, h, w = predicted_frames.size()

            for i in range(pred_timesteps):
                pred = predicted_frames[:, i:i+1, :, :, :].view(b, c, h, w)
                gt = ground_truth[:, i:i+1, :, :, :].view(b, c, h, w)
                avg_mses[i] += F.mse_loss(pred, gt).item()         # normalized by b
                avg_ssims[i] += utils.get_normalized_ssim(pred, gt)   # normalized by b
                
            loggers.log_test_loss(opt, step, loss.item())
            # utils.save_video(predicted_frames.cpu(), ground_truth.cpu(), step, batches // 20, tb) # Log video 20 times

        test_loss /= batches # avg test loss over all batches
        
        for i in range(1, pred_timesteps):
            avg_mses[i] += avg_mses[i-1] 
            avg_ssims[i] += avg_ssims[i-1] 
            
        for i in range(pred_timesteps):
            avg_mses[i] /= (batches * (i+1))    # normalize by batches and pred_len
            avg_ssims[i] /= (batches * (i+1))   # normalize by batches and pred_len
            avg_psnrs[i] = 10 * math.log10(1 / avg_mses[i])

        avg_mse, avg_psnr, avg_ssim = avg_mses[-1], avg_psnrs[-1], avg_ssims[-1]
        loggers.plot_metrics_vs_n_frames(avg_psnrs, avg_mses, avg_ssims,  opt.id, opt)   # plots metrics vs #predicted frames
        loggers.log_final_test_metrics(test_loss, avg_mse, avg_psnr, avg_ssim, opt.id)    # Logs final MSE, PSNR, SSIM
        # loggers.log_metrics_to_tb(avg_mses, avg_psnrs, avg_ssims, tb)               # log metrics to tensorboard