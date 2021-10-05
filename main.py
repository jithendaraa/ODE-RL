import argparse
from models.DS2VAE import DS2VAE
import numpy as np
import torch
import ruamel.yaml as yaml
import sys
import pathlib
import helpers.utils as utils
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

from dataloader import parse_datasets
# Models
from models.ConvGRU import ConvGRU
from models.ODEConvGRU import ODEConvGRU
# from models.VidODE import VidODE
from models.S3VAE import S3VAE
from train_test import train, test
from torch.utils.tensorboard import SummaryWriter


def get_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--configs', nargs='+', required=True)
    args, remaining = parser.parse_known_args()
    configs = yaml.safe_load(
      (pathlib.Path(sys.argv[0]).parent / 'configs.yaml').read_text())
    defaults = {}
    for name in args.configs:
        defaults.update(configs[name])

    parser = argparse.ArgumentParser()
    for key, value in sorted(defaults.items(), key=lambda x: x[0]):
      arg_type = utils.args_type(value)
      parser.add_argument(f'--{key}', type=arg_type, default=arg_type(value))
    
    opt = parser.parse_args(remaining)
    opt = utils.set_opts(opt)

    exp_config = {}

    for arg in vars(opt):
      val = getattr(opt, arg)
      exp_config[arg] = val
      # print(arg, val)

    print()
    return opt, exp_config
    
def init_model(opt, device):

    implemented_models = ['ConvGRU', 'cgrudecODE', 'ODEConv', 'S3VAE', 'DS2VAE']
    
    if opt.model in ['ConvGRU', 'cgrudecODE']:
      model = ConvGRU(opt, device, decODE=opt.decODE)
    
    elif opt.model in ['ODEConv']:
      model = ODEConvGRU(opt, device)
      print("Initialised ODEConv model")

    elif opt.model in ['S3VAE']:
      model = S3VAE(opt, device)

    elif opt.model in ['DS2VAE']:
      model = DS2VAE(opt, device)

    elif opt.model in ['VidODE']:
      raise NotImplementedError(f'Model {opt.model} is not implemented. Try one of {implemented_models}')
    
    else: 
      raise NotImplementedError(f'Model {opt.model} is not implemented. Try one of {implemented_models}')

    return model

def main(opt, exp_config):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Dataloader
    loader_objs = parse_datasets(opt, device)
    print("Loaded", opt.dataset, "dataset")
    print("Args:", opt)

    writer = SummaryWriter(opt.rundir)
    model = init_model(opt, device)

    if opt.load_model is True:
      model = utils.load_model_params(model, opt)

    if opt.phase == 'train':
      train(opt, model, loader_objs, device, exp_config, writer)
      
    elif opt.phase == 'test':
      test(opt, model, loader_objs, device, exp_config)


if __name__ == '__main__':
    opt, exp_config = get_opt()
    main(opt, exp_config)