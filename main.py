import argparse
import numpy as np
import torch
import ruamel.yaml as yaml
import sys
import pathlib
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

from dataloader import parse_datasets
from models.ConvGRU import ConvGRU
from train_test import train, test
from tensorboardX import SummaryWriter

import helpers.utils as utils


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
    print()
    opt = utils.set_opts(opt)
    print()
    return opt
    

def main(opt):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Dataloader
    loader_objs = parse_datasets(opt, device)
    print("Loaded", opt.dataset, "dataset")

    if opt.model == 'ConvGRU':
      model = ConvGRU(opt, device)
    
    if opt.load_model is True:
      model = utils.load_model_params(model, opt)

    if opt.phase == 'train':
      train(opt, model, loader_objs, device)
      
    elif opt.phase == 'test':
      test(opt, model, loader_objs, device)

if __name__ == '__main__':
    opt = get_opt()
    main(opt)