import torch
import torch.nn as nn
import os
import gzip
import numpy as np
import pickle
import time
import datetime
import re

def args_type(default):
  def parse_string(x):
    if default is None:
      return x
    if isinstance(default, bool):
      return bool(['False', 'True'].index(x))
    if isinstance(default, int):
      return float(x) if ('e' in x or '.' in x) else int(x)
    if isinstance(default, (list, tuple)):
      return tuple(args_type(default[0])(y) for y in x.split(','))
    return type(default)(x)
  def parse_object(x):
    if isinstance(default, (list, tuple)):
      return tuple(x)
    return x
  return lambda x: parse_string(x) if isinstance(x, str) else parse_object(x)

# Get device
def get_device(tensor):
    device = torch.device("cpu")
    if tensor.is_cuda:
        device = tensor.get_device()
    return device

# Data loading for MMNIST: load_mnist, load_fixed_set
def load_mnist(data_dir):
    # Load MNIST dataset for generating training data.
    path = os.path.join(data_dir, 'train-images-idx3-ubyte.gz')
    with gzip.open(path, 'rb') as f:
        mnist = np.frombuffer(f.read(), np.uint8, offset=16)
        mnist = mnist.reshape(-1, 28, 28)
    return mnist

def load_fixed_set(root, is_train):
    # Load the fixed dataset
    filename = 'mnist_test_seq.npy'
    path = os.path.join(root, filename)
    dataset = np.load(path)
    dataset = dataset[..., np.newaxis]
    return dataset
# _____________________________________________________________________________

# Dataloader unpackers: inf_generator, get_data_dict, get_dict_template, get_next_batch
def inf_generator(iterable):
    """Allows training with DataLoaders in a single infinite loop:
        for i, (x, y) in enumerate(inf_generator(train_loader)):
    """
    iterator = iterable.__iter__()
    while True:
        try:
            yield iterator.__next__()
        except StopIteration:
            iterator = iterable.__iter__()

def get_data_dict(dataloader):
    data_dict = dataloader.__next__()
    return data_dict

def get_dict_template():
    return {"observed_data": None,
            "data_to_predict": None,
            'timesteps': None,
            'observed_tp': None,
            'tp_to_predict': None}

def get_next_batch(data_dict, test_interp=False, opt=None, in_len=None, out_len=None):
    device = get_device(data_dict["observed_data"])
    batch_dict = get_dict_template()
    batch_dict["observed_data"] = data_dict["observed_data"]
    batch_dict["data_to_predict"] = data_dict["data_to_predict"]
    
    input_t = data_dict["observed_data"].size()[1]
    output_t = data_dict["data_to_predict"].size()[1]
    total_t = input_t + output_t
    
    batch_dict["timesteps"] = torch.tensor(np.arange(0, total_t) / total_t).to(device)
    batch_dict["observed_tp"] = torch.tensor(batch_dict["timesteps"][:input_t]).to(device)
    batch_dict["tp_to_predict"] = torch.tensor(batch_dict["timesteps"][input_t:]).to(device)
    return batch_dict
# ______________________________________________________________________________

def update_learning_rate(optimizer, decay_rate=0.999, lowest=1e-3):
    for param_group in optimizer.param_groups:
        lr = param_group['lr']
        lr = max(lr * decay_rate, lowest)
        param_group['lr'] = lr

def get_norm_layer(ch):
    norm_layer = nn.BatchNorm2d(ch)
    return norm_layer

def create_convnet(n_inputs, n_outputs, n_layers=1, n_units=128, downsize=False, nonlinear='tanh'):
    if nonlinear == 'tanh':
        nonlinear = nn.Tanh()
    elif nonlinear == 'relu':
        nonlinear = nn.ReLU()
    else:
        raise NotImplementedError('There is no named')

    layers = []
    layers.append(nn.Conv2d(n_inputs, n_units, 3, 1, 1, dilation=1))
    
    for i in range(n_layers):
        layers.append(nonlinear)
        if downsize is False:
            layers.append(nn.Conv2d(n_units, n_units, 3, 1, 1, dilation=1))
        else:
            layers.append(nn.Conv2d(n_units, n_units, 4, 2, 1, dilation=1))
    
    layers.append(nonlinear)
    layers.append(nn.Conv2d(n_units, n_outputs, 3, 1, 1, dilation=1))

    return nn.Sequential(*layers)

def create_transpose_convnet(n_inputs, n_outputs, n_layers=1, n_units=128, upsize=False, nonlinear='tanh'):
    if nonlinear == 'tanh':
        nonlinear = nn.Tanh()
    elif nonlinear == 'relu':
        nonlinear = nn.ReLU()
    else:
        raise NotImplementedError('There is no named')

    layers = []
    layers.append(nn.ConvTranspose2d(n_inputs, n_units, 3, 1, 1, dilation=1))
    
    for i in range(n_layers):
        layers.append(nonlinear)
        if upsize is False:
            layers.append(nn.ConvTranspose2d(n_units, n_units, 3, 1, 1, dilation=1))
        else:
            layers.append(nn.ConvTranspose2d(n_units, n_units, 4, 2, 1, dilation=1))
    
    layers.append(nonlinear)
    layers.append(nn.ConvTranspose2d(n_units, n_outputs, 3, 1, 1, dilation=1))

    return nn.Sequential(*layers)

def log_after_epoch(epoch_num, loss, metrics, step, start_time, opt=None):
    
    if (epoch_num % opt.log_freq) == 0:
        metrics['train_losses_x'].append(step)
        metrics['train_losses'].append(loss)
        et = time.time() - start_time
        et = str(datetime.timedelta(seconds=et))[:-7]
        log = f"Elapsed [{et}] Epoch [{epoch_num:03d}/{opt.epochs:03d}]\t"\
                        f"Iterations [{(step):6d}] \t"\
                        f"Mse [{loss:.10f}]\t"
        print(log)
        # TODO: Output train loss to tensorboard

    return metrics

def log_test_loss(opt, step, loss):
    if (step % opt.test_log_freq) == 0:
        print(f"Test loss for step {step}: {loss}")

# Save model params every `ckpt_save_freq` steps as model_params_logdir/ID_00000xxxxx.pickle
def save_model_params(model, optimizer, epoch, opt, step, ckpt_save_freq):
    
    if step > 0 and (step % ckpt_save_freq == 0):
        padded_zeros = '0' * (10 - len(str(step)))
        padded_step = padded_zeros + str(step)
        model_params_file_name = os.path.join(opt.model_params_logdir, opt.id + '_' + padded_step + '.pickle')

        model_dict = {
            'epoch': epoch,
            'step': step,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict()
        }

        with open(model_params_file_name, 'wb') as handle:
            pickle.dump(model_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
            print("Saved model parameters at step", step, "->", model_params_file_name)

def load_model_params(model, opt):
    
    # Load model params of this exp id
    load_id = opt.model + '_' + opt.dataset + '_train_' + str(opt.train_in_seq) + '_' + str(opt.train_out_seq) + '_' 
    saved_params_list = os.listdir(opt.model_params_logdir)
    # Required saved_params_list shoudl start with load_id
    r = re.compile(load_id + "*")
    filtered_saved_params_list = list(filter(r.match, saved_params_list))
    filtered_saved_params_list.sort()

    model_params_file_path = os.path.join(opt.model_params_logdir, filtered_saved_params_list[-1])  # select params_file with highest steps

    objects = {}
    with (open(model_params_file_path, "rb")) as openfile:
        while True:
            try:
                objects = pickle.load(openfile)
                print("Loaded model at", model_params_file_path)
                break
            except EOFError:
                print("Error reading params pickle at", model_params_file_path)
                break
    print()
    model.load_state_dict(objects['state_dict'])
    return model

def set_opts(opt):

    # Get exp ID
    if opt.phase == 'train':
        opt.id += '_' + str(opt.train_in_seq) + '_' + str(opt.train_out_seq) 
    else:
        opt.id += '_' +  str(opt.test_in_seq) + '_' + str(opt.test_out_seq) 
    print("ID:", opt.id, '\n')

    # Set run directory
    if os.path.isdir(opt.rundir) is False: os.mkdir(opt.rundir)     # create runs/ if needed
    opt.rundir = os.path.join(opt.rundir, opt.model)                
    if os.path.isdir(opt.rundir) is False: os.mkdir(opt.rundir)     # create runs/ConvGRU if needed
    opt.rundir = os.path.join(opt.rundir, opt.id)
    print("rundir:", opt.rundir)

    # Set logdir and create dirs along the way
    if os.path.isdir(opt.logdir) is False:  os.mkdir(opt.logdir)    # mkdir logs
    opt.logdir = os.path.join(opt.logdir, opt.model)                # logs/ConvGRU
    if os.path.isdir(opt.logdir) is False:  os.mkdir(opt.logdir)    # mkdir logs/ConvGRU
    print("logdir:", opt.logdir)

    opt.storage_dir = os.path.join(opt.user_dir, opt.storage_dir)
    opt.data_dir = os.path.join(opt.storage_dir, opt.data_dir)
    print("data_dir:", opt.data_dir)

    # Set video logdir and create dirs along the way
    opt.video_logdir = os.path.join(opt.storage_dir, opt.video_logdir)          # /home/jithen/scratch/videos
    if os.path.isdir(opt.video_logdir) is False:  os.mkdir(opt.video_logdir)    
    opt.video_logdir = os.path.join(opt.video_logdir, opt.model)                # /home/jithen/scratch/videos/ConvGRU
    if os.path.isdir(opt.video_logdir) is False:  os.mkdir(opt.video_logdir)
    print("video_dir:", opt.video_logdir)

    # Set model_params_logdir and create dirs along the way
    opt.model_params_logdir = os.path.join(opt.logdir, opt.model_params_logdir)     # logs/ConvGRU/model_params
    if os.path.isdir(opt.model_params_logdir) is False:  os.mkdir(opt.model_params_logdir)
    print("model_params_logdir:", opt.model_params_logdir)

    return opt

def print_exp_details(opt, n_batches):
    print()
    print("Exp ID: ", opt.id)
    if opt.phase == 'train':
        print("Training the", opt.model, "model on", opt.dataset, 'for', n_batches, 'batches of batch size', opt.batch_size)
        print("Input frames:", opt.train_in_seq)
        print("Output frames:", opt.train_out_seq)

    else:
        print("Training the", opt.model, "model on", opt.dataset, 'for', n_batches, 'batches of batch size', opt.batch_size)
        print("Input frames:", opt.test_in_seq)
        print("Output frames:", opt.test_out_seq)
    print()

def save_video(pred, truth, step, log_video_freq, tb):

    if (step % log_video_freq) == 0:
        pred, truth = pred[0], truth[0] # extract first batch
        # TODO: save as GIF
        # TODO: send GIF to tensorboard

    pass

def plot_psnr_vs_n_frames(ground_truth, predicted_frames, tb):
    # TODO: plot PSNR vs. #frames
    # TODO: send plot to tensorboard
    # TODO: call this function in test() method of train_test.py
    pass