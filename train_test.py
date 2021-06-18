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

    os.environ["WANDB_API_KEY"] = "73b7aa2bb830c99a8a3e6228588c6587d037ee96"
    os.environ["WANDB_MODE"] = "dryrun"

    # 1. Start a new run
    run = wandb.init(project='VideoODE', config=exp_config_dict, reinit=True)
    
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
            
        epoch_loss /= n_train_batches # Avg loss over all batches for this epoch
        
        wandb.log({"Train Loss (per epoch)": epoch_loss}, step=epoch)
    run.finish()    

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
    
    # Step with optimizer
    optimizer.step()

    return ((predicted_frames/2.0) + 0.5) * 255.0, (ground_truth + 0.5) * 255.0, train_loss.item()
