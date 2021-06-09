import torch
import torch.optim as optim
torch.autograd.set_detect_anomaly(True)
from torch.optim import lr_scheduler

import numpy as np
import time
import os
from torch.utils.tensorboard import SummaryWriter

import helpers.utils as utils


def train(opt, model, loader_objs, device):

    tb = SummaryWriter(opt.rundir)
    print(f"Logging to {opt.logdir}")
    optimizer = optim.Adamax(model.parameters(), lr=opt.lr)
    pla_lr_scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=4, verbose=True)

    # Data loaders
    train_dataloader = loader_objs['train_dataloader']
    n_train_batches = loader_objs['n_train_batches']

    total_step = 0
    start_time = time.time()
    losses = []
    utils.print_exp_details(opt, n_train_batches)

    metrics = {
        "train_losses_x": [],  # steps
        "train_losses": [],    # Avg loss over all train_batches for all epochs
        "valid_losses_x": [],  # steps
        "valid_losses": []     # Avg loss over all valid_batches after every validate_frequency epochs
    }

    print("n_train_batches:", n_train_batches)
    
    for epoch in range(opt.epochs):
        epoch_train_loss = 0
        utils.update_learning_rate(optimizer, decay_rate=0.99, lowest=opt.lr / 10)
        a = time.time()
        for it in range(n_train_batches):
            data_dict = utils.get_data_dict(train_dataloader)
            batch_dict = utils.get_next_batch(data_dict, opt, opt.train_in_seq, opt.train_out_seq)

            input_frames = batch_dict['observed_data'].to(device)
            ground_truth = batch_dict['data_to_predict'].to(device)

            predicted_frames = model.get_prediction(input_frames, batch_dict=batch_dict)
            train_loss = model.get_loss(predicted_frames, ground_truth)
            epoch_train_loss += train_loss.item()
            
            optimizer.zero_grad()
            train_loss.backward()
            torch.nn.utils.clip_grad_value_(model.parameters(), clip_value=10.0)
            optimizer.step()
            total_step += 1
            # print(it, (time.time()-a)/3600, "hours")
            
            """
                # 1. Validate model every `validate_freq` epochs
                # 2. Save model params every `ckpt_save_freq` step
                # 3. #TODO: Save video every `log_video_freq` step. See save_video() in helpers/utils.py
            """

            validate_model(model, opt, loader_objs, metrics, pla_lr_scheduler, total_step, device, tb, opt.validate_freq)
            utils.save_model_params(model, optimizer, epoch, opt, total_step, opt.ckpt_save_freq)
            utils.save_video(predicted_frames.cpu(), ground_truth.cpu(), total_step, opt.log_video_freq, tb)
            tb.add_scalar('Train Loss', train_loss.item(), total_step)

        epoch_train_loss /= n_train_batches # Avg loss over all batches for this epoch
        # log epoch number and train_loss for every `log_freq` steps
        metrics = utils.log_after_epoch(epoch, epoch_train_loss, metrics, total_step, start_time, opt=opt)


def validate_model(model, opt, loader_objs, metrics, lr_schedule, step, device, tb, validate_freq, type_='validate'):
    if step % validate_freq == 0:
        metrics = test(opt, model, loader_objs, device, step, type_, metrics, lr_schedule, tb=tb, valid_batches=opt.valid_batches)
    return metrics

def test(opt, model, loader_objs, device, step=None, type_='test', metrics=None, lr_schedule=None, valid_batches=5, tb=None):

    if type_ == 'test':
        tb = SummaryWriter(opt.rundir)
        dataloader = loader_objs['test_dataloader']
        batches = loader_objs['n_test_batches'] 
        print(f"Testing model on {batches} batches")

    elif type_ == 'validate':
        print(f"Validating model at step {step} on {valid_batches} batches")
        dataloader = loader_objs['valid_dataloader']
        batches = loader_objs['n_valid_batches']

    test_loss = 0

    with torch.no_grad():
        model.eval()
        if step is None:
            step = 0

        for it in range(batches):
            data_dict = utils.get_data_dict(dataloader)
            batch_dict = utils.get_next_batch(data_dict, opt)

            input_frames = batch_dict['observed_data'].to(device)
            ground_truth = batch_dict['data_to_predict'].to(device)

            predicted_frames = model.get_prediction(input_frames)
            loss = model.get_loss(predicted_frames, ground_truth)
            test_loss += loss.item()
            
            if it == (valid_batches - 1): 
                test_loss /= valid_batches
                break

            if type_ == 'test': 
                step += 1
                utils.log_test_loss(opt, step, loss.item())
                log_video_freq = batches // 20
                utils.save_video(predicted_frames.cpu(), ground_truth.cpu(), total_step, log_video_freq, tb)

        if type_ == 'test': 
            test_loss /= batches # avg test loss over all batches
            print("Final Test loss on entire dataset after evaluation:", test_loss)

        if type_ == 'validate':
            metrics['valid_losses_x'].append(step)
            metrics['valid_losses'].append(test_loss)
            lr_schedule.step(test_loss)
            print("Validation loss: ", test_loss)
            tb.add_scalar('Validation Loss', test_loss, step)
            return metrics