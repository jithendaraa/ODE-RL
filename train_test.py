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

    print(f"Logging to {opt.logdir}")
    optimizer = optim.Adamax(model.parameters(), lr=opt.lr)
    pla_lr_scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=4, verbose=True)
    
    for epoch in range(opt.epochs):
        epoch_loss = 0
        utils.update_learning_rate(optimizer, decay_rate=0.99, lowest=opt.lr / 10)

        for it in range(n_train_batches):   # n_train_batches steps
            pred, gt, step_loss = train_batch(model, train_dataloader, optimizer, opt, device)
            pred_gt = torch.cat((pred.detach().cpu(), gt.cpu()), 0).numpy()
            epoch_loss += step_loss
            step += 1

            if opt.off_wandb is False:
                # Log losses and pred, gt videos
                if step % opt.loss_log_freq == 0:
                    wandb.log( {'Per Step Loss': step_loss}, step=step)

                if step == 1 or step % opt.video_log_freq == 0:
                    wandb.log({ 'Pred_GT': wandb.Video(pred_gt) }, step=step)
                    print("Logged video")
                
            print(f"step {step}")

            # Save model params
            # utils.save_model_params(model, optimizer, epoch, opt, step, opt.ckpt_save_freq)
            
        # epoch_loss /= n_train_batches # Avg loss over all batches for this epoch
        # wandb.log({"Per Epoch Loss": epoch_loss})
        # loggers.log_after_epoch(epoch, epoch_loss, step, start_time, total_steps, opt=opt)


def test(opt, model, loader_objs, device, exp_config_dict, step=None, metrics=None, lr_schedule=None):

    test_loss = 0
    step = 0
    pred_timesteps = opt.test_out_seq
    avg_mses = [0] * pred_timesteps
    avg_psnrs = [0] * pred_timesteps
    avg_ssims = [0] * pred_timesteps

    dataloader = loader_objs['test_dataloader']
    batches = loader_objs['n_test_batches'] 
    print(f"Testing model on {batches} batches ({opt.batch_size*batches} steps)")

    if opt.offline is True: os.system('wandb offline')
    
    if opt.off_wandb is False:
        # 1. Start a new run
        wandb.init(project=opt.wandb_project, entity=opt.wandb_entity, config=exp_config_dict)
        # 2. Save model inputs and hyperparameters
        config = wandb.config
        # 3. Log gradients and model parameters
        wandb.watch(model)

    with torch.no_grad():
        model.eval()
            
        for it in range(batches):
            pred, gt, loss = test_batch(model, dataloader, opt, device) # pred, gt are in -0.5, 0.5
            pred_gt = torch.cat(((pred.cpu() + 0.5) * 255.0, (gt.cpu() + 0.5) * 255.0), 0).numpy()
            b, _, c, h, w = pred.size()
            test_loss += loss
            step += 1

            for i in range(pred_timesteps):
                pred_ = pred[:, i:i+1, :, :, :].view(b, 1, c, h, w)
                gt_ = gt[:, i:i+1, :, :, :].view(b, 1, c, h, w)

                mse_loss = model.get_loss(pred_, gt_).item()
                avg_mses[i] += mse_loss          # normalized by b

                pred_255, gt_255 = (pred_ + 0.5) * 255.0, (gt_ + 0.5) * 255.0
                pred_255, gt_255 = pred_255.view(b, c, h, w), gt_255.view(b, c, h, w)
                avg_ssims[i] += utils.get_normalized_ssim(pred_255, gt_255)   # normalized by b
                
            print(f'Testing step {step}....')

            if step % 500 == 0 and opt.off_wandb is False:
                wandb.log({'Pred_GT': wandb.Video(pred_gt)}, step=it+1)

        test_loss /= batches # avg test loss over all batches

        for i in range(1, pred_timesteps):
            avg_mses[i] += avg_mses[i-1]         # normalized by b
            avg_ssims[i] += avg_ssims[i-1]
        
        for i in range(pred_timesteps):
            avg_mses[i] /= (batches * (i+1))    # normalize by batches and pred_len
            avg_ssims[i] /= (batches * (i+1))   # normalize by batches and pred_len
            avg_psnrs[i] = 10 * math.log10(1 / avg_mses[i])

            print(avg_mses[i], avg_psnrs[i], avg_ssims[i])

            if opt.off_wandb is False:
                wandb.log({"PSNR": avg_psnrs[i],
                            "MSE": avg_mses[i], 
                            "SSIM": avg_ssims[i]})

        avg_mse, avg_psnr, avg_ssim = avg_mses[-1], avg_psnrs[-1], avg_ssims[-1]
        loggers.log_final_test_metrics(test_loss, avg_mse, avg_psnr, avg_ssim, opt.id)    # Logs final MSE, PSNR, SSIM

def test_batch(model, test_dataloader, opt, device):
    data_dict = utils.get_data_dict(test_dataloader)
    batch_dict = utils.get_next_batch(data_dict, opt)
    
    input_frames = batch_dict['observed_data'].to(device) 
    ground_truth = batch_dict['data_to_predict'].to(device) 

    predicted_frames = model.get_prediction(input_frames, batch_dict=batch_dict)
    loss = model.get_loss(predicted_frames, 2.0*ground_truth).item()

    return predicted_frames/2.0, ground_truth, loss

def train_batch(model, train_dataloader, optimizer, opt, device):
    # Get bacth data
    data_dict = utils.get_data_dict(train_dataloader)
    batch_dict = utils.get_next_batch(data_dict, opt, opt.train_in_seq, opt.train_out_seq)

    # Get input sequence and output ground truth 
    input_frames, ground_truth = batch_dict['observed_data'].to(device), batch_dict['data_to_predict'].to(device) 

    # Forward pass
    predicted_frames = model.get_prediction(input_frames, batch_dict=batch_dict)
    train_loss = model.get_loss(predicted_frames, 2.0*ground_truth)

    # Backward pass
    optimizer.zero_grad()
    train_loss.backward()
    # torch.nn.utils.clip_grad_value_(model.parameters(), clip_value=10.0)

    # Step with optimizer
    optimizer.step()

    return ((predicted_frames/2.0) + 0.5) * 255.0, (ground_truth + 0.5) * 255.0, train_loss.item()
