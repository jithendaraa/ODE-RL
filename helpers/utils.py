import torch
import torch.nn as nn
import os
import gzip
import numpy as np
import pickle
import time
import datetime
import re
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim
from torchvision.utils import save_image
from torchvision.utils import make_grid

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

def set_opts(opt):

    # Get exp ID
    if opt.phase == 'train':
        opt.id += '_' + str(opt.train_in_seq) + '_' + str(opt.train_out_seq) 
    else:
        opt.id += '_' +  str(opt.test_in_seq) + '_' + str(opt.test_out_seq) 
    print("ID:", opt.id, '\n')

    # # Set model_params_logdir and create dirs along the way
    # opt.model_params_logdir = os.path.join(opt.logdir, opt.model_params_logdir)     # logs/ConvGRU/model_params
    # if os.path.isdir(opt.model_params_logdir) is False:  os.mkdir(opt.model_params_logdir)
    # print("model_params_logdir:", opt.model_params_logdir)

    return opt

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

def create_convnet(n_inputs, n_units):
    n_units=[16,32,64,128]
    layers = []
    layers.append(nn.Conv2d(n_inputs, n_units[0], 4, 2, 1, dilation=1))
    layers.append(nn.ReLU())

    layers.append(nn.Conv2d(n_units[0], n_units[1], 4, 2, 1, dilation=1))
    layers.append(nn.ReLU())

    layers.append(nn.Conv2d(n_units[1], n_units[2], 4, 2, 1, dilation=1))
    layers.append(nn.ReLU())

    layers.append(nn.Conv2d(n_units[2], n_units[3], 4, 2, 1, dilation=1))
    layers.append(nn.ReLU())

    return nn.Sequential(*layers)

def create_odeconvnet(n_inputs, n_units):
    layers = []
    layers.append(nn.Conv2d(n_inputs, n_units[0], 3, 1, 1, dilation=1))
    layers.append(nn.ReLU())

    layers.append(nn.Conv2d(n_units[0], n_units[0], 3, 1, 1, dilation=1))
    layers.append(nn.ReLU())

    layers.append(nn.Conv2d(n_units[0], n_units[0], 3, 1, 1, dilation=1))
    layers.append(nn.ReLU())

    layers.append(nn.Conv2d(n_units[0], n_units[1], 3, 1, 1, dilation=1))
    layers.append(nn.ReLU())

    return nn.Sequential(*layers)

def create_transpose_convnet(n_inputs, n_units):
    
    layers = []
    layers.append(nn.ConvTranspose2d(n_inputs, n_units[0], 4, 2, 1, dilation=1))
    layers.append(nn.ReLU())

    layers.append(nn.ConvTranspose2d(n_units[0], n_units[1], 4, 2, 1, dilation=1))
    layers.append(nn.ReLU())

    layers.append(nn.ConvTranspose2d(n_units[1], n_units[2], 4, 2, 1, dilation=1))
    layers.append(nn.ReLU())

    layers.append(nn.ConvTranspose2d(n_units[2], n_units[3], 4, 2, 1, dilation=1))
    layers.append(nn.Tanh())

    return nn.Sequential(*layers)

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
    padded_zeros = '0' * (10 - len(str(opt.step)))
    padded_step = padded_zeros + str(opt.step)
    file_to_load = load_id + padded_step + '.pickle'

    saved_params_list = os.listdir(opt.model_params_logdir)
    # Required saved_params_list should start with load_id
    # r = re.compile(load_id + "*")
    # filtered_saved_params_list = list(filter(r.match, saved_params_list))
    # filtered_saved_params_list.sort()

    assert file_to_load in saved_params_list
    model_params_file_path = os.path.join(opt.model_params_logdir, file_to_load)  # select params_file with highest steps

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

def get_normalized_ssim(pred, gt):

    pred_np = pred.cpu()    # b, t, c, h, w
    gt_np = gt.cpu()
    ssim_val = 0
    b, t = pred_np.shape[0], pred_np.shape[1]

    for pred_, gt_ in zip(pred_np, gt_np):  # For every batch
        for pred_t, gt_t in zip(pred_, gt_):    # For every time step
            ssim_val += ssim(pred_t, gt_t, data_range=255, gaussian_weights=True, use_sample_covariance=False, channel_axis=0,multichannel=True)
    
    return (ssim_val / (b*t))

