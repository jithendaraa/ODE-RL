import torch
import torch.optim as optim
torch.autograd.set_detect_anomaly(True)
import argparse
import os
import time
import datetime
import json
from pathlib import Path
import numpy as np

from dataloader import parse_datasets
from models.conv_odegru import *
from models.gan import *
from tester import Tester
import utils
import visualize
import warnings
warnings.filterwarnings("ignore", category=UserWarning)


def get_opt():
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--name", default="vid_ode", help='Specify experiment')
    parser.add_argument("--jobid", default="sample", help='Specify experiment')
    parser.add_argument('-j', '--workers', type=int, default=4)
    parser.add_argument('-b', '--batch_size', type=int, default=6)
    parser.add_argument('-e', '--epoch', type=int, default=500, help='epoch')
    parser.add_argument('-p', '--phase', default="train", choices=["train", "test_met"])
    
    # Hyper-parameters
    parser.add_argument('--lr', type=float, default=1e-3, help="Starting learning rate.")
    parser.add_argument('--window_size', type=int, default=20, help="Window size to sample")
    parser.add_argument('--sample_size', type=int, default=10, help="Number of time points to sub-sample")
    
    # Hyper-parameters
    parser.add_argument('--lamb_adv', type=float, default=0.003, help="Adversarial Loss lambda")

    # Model architectures
    parser.add_argument('--nru', default=False) # alternatively predict m_t and h_t
    parser.add_argument('--nru2', default=False) # predict all m_t at once, then use that to get h_t's
    parser.add_argument('-sa', '--slot_attention', action='store_true', default=False) 
    
    # SLot attention network variants
    parser.add_argument('--pos', type=int, default=2) # For slot attention: pos 1 -> slot attention after Encoder before ConvGRU
    parser.add_argument('--num_slots', type=int, default=4)
    parser.add_argument('--dim', type=int, default=256)
    parser.add_argument('--slot_iters', type=int, default=3)
    
    # Network variants for experiment..
    parser.add_argument('-fd', '--frame_dims', type=int, default=64)
    parser.add_argument('--input_size', type=int, default=128)
    parser.add_argument('--dec_diff', type=str, default='dopri5', choices=['dopri5', 'euler', 'adams', 'rk4'])
    parser.add_argument('-nl', '--n_layers', type=int, default=2, help='A number of layer of ODE func')
    parser.add_argument('-cc', '--convGRU_cells', type=int, default=1, help='A number of layer of ODE func')
    parser.add_argument('--n_downs', type=int, default=2)
    parser.add_argument('--init_dim', type=int, default=32)
    parser.add_argument('--input_norm', action='store_true', default=False)
    
    parser.add_argument('--run_backwards', action='store_true', default=True)
    parser.add_argument('--irregular', action='store_true', default=False, help="Train with irregular time-steps")
    parser.add_argument("--sample_from_beg", type=bool, default=False)

    # Need to be tested...
    parser.add_argument('--extrap', action='store_true', default=True, help="Set extrapolation mode. If this flag is not set, run interpolation mode.")

    # Test argument:
    parser.add_argument('--split_time', default=10, type=int, help='Split time for extrapolation or interpolation ')
    
    # Sequence parameters
    parser.add_argument('-is', '--input_sequence', default=3, type=int, help='Input frame sequence length')
    parser.add_argument('-os', '--output_sequence', default=14, type=int, help='Ouput frame sequence length')
    parser.add_argument('-u', '--unequal', action='store_true', default=False)

    # Log
    parser.add_argument("--ckpt_save_freq", type=int, default=10000)
    parser.add_argument("--log_print_freq", type=int, default=4000)
    parser.add_argument("--image_print_freq", type=int, default=5000)
    
    # Path (Data & Checkpoint & Tensorboard)
    parser.add_argument('-d', '--dataset', type=str, default='kth', choices=["mgif", "hurricane", "kth", "penn", "minerl", 'cater', 'moving_mnist', 'box2D', "phyre"])
    parser.add_argument('--log_dir', type=str, default='./logs', help='save tensorboard infos')
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints', help='save checkpoint infos')
    parser.add_argument('-td', '--test_dir', type=str, help='load saved model')
    
    opt = parser.parse_args()
    if opt.dataset == 'phyre':
        opt.input_size = 64
        opt.extrap = True
        opt.sample_from_beg = True
    
    elif opt.dataset == 'kth':
        # Clockwork VAE experiment conditions
        if opt.unequal is True:
            opt.sample_size = opt.input_sequence + opt.output_sequence
            opt.window_size = opt.sample_size

    elif opt.dataset == 'minerl':
        if opt.unequal is True:
            opt.sample_size = opt.input_sequence + opt.output_sequence
            opt.window_size = opt.sample_size

    opt.input_dim = 3
    
    if opt.phase == 'train':
        # Make Directory
        STORAGE_PATH = utils.create_folder_ifnotexist("./storage")
        STORAGE_PATH = STORAGE_PATH.resolve()
        LOG_PATH = utils.create_folder_ifnotexist(STORAGE_PATH / "logs")
        CKPT_PATH = utils.create_folder_ifnotexist(STORAGE_PATH / "checkpoints")

        # Modify Desc
        # now = datetime.datetime.now()
        # month_day = f"{now.month:02d}{now.day:02d}"
        opt.name = f"dataset{opt.dataset}_{opt.convGRU_cells}c_{opt.n_layers}l_extrap{opt.extrap}_f{opt.frame_dims}_nru{opt.nru}_nru2{opt.nru2}_e{opt.epoch}_b{opt.batch_size}_unequal{opt.unequal}_{opt.input_sequence}_{opt.output_sequence}_{opt.sample_size}"
        # opt.log_dir = utils.create_folder_ifnotexist(LOG_PATH / month_day / opt.name)
        # opt.checkpoint_dir = utils.create_folder_ifnotexist(CKPT_PATH / month_day / opt.name)
        opt.log_dir = utils.create_folder_ifnotexist(LOG_PATH / opt.name)
        opt.checkpoint_dir = utils.create_folder_ifnotexist(CKPT_PATH / opt.name)

        # Write opt information
        with open(str(opt.log_dir / 'options.json'), 'w') as fp:
            opt.log_dir = str(opt.log_dir)
            opt.checkpoint_dir = str(opt.checkpoint_dir)
            json.dump(opt.__dict__, fp=fp)
            print("option.json dumped!")
            opt.log_dir = Path(opt.log_dir)
            opt.checkpoint_dir = Path(opt.checkpoint_dir)
        
        opt.train_image_path = utils.create_folder_ifnotexist(opt.log_dir / "train_images")
        opt.test_image_path = utils.create_folder_ifnotexist(opt.log_dir / "test_images")
    else:
        print("[Info] In test phase, skip dumping options.json..!")
    
    return opt

def set_seed(seed_value, use_cuda=True):
    np.random.seed(seed_value) # cpu vars
    torch.manual_seed(seed_value) # cpu  vars
    random.seed(seed_value) # Python
    os.environ['PYTHONHASHSEED'] = str(seed_value) # Python hash buildin
    if use_cuda:
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value) # gpu vars
        torch.backends.cudnn.deterministic = True  #needed
        torch.backends.cudnn.benchmark = False
        
def main():
    # Option
    opt = get_opt()
    
    if opt.phase != 'train':
        tester = Tester()
        opt = tester._load_json(opt)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # print(f"device:{device}")
    
    # Dataloader
    loader_objs = parse_datasets(opt, device)
    
    # Model
    model = VidODE(opt, device)
    
    # print("NRU:", opt.nru, opt.input_sequence, opt.output_sequence, opt.extrap)
    # print("NRU2:", opt.nru2)
    # print("Dataset:", opt.dataset)
    # print("Batch size:", opt.batch_size)
    # print("runback:", opt.run_backwards)
    # print("Slot attention:", opt.slot_attention, opt.pos)
    
    # Set tester
    if opt.phase != 'train':
        tester._load_model(opt, model)
        tester._set_properties(opt, model, loader_objs, device)
    
    # Phase
    if opt.phase == 'train':
        train(opt, model, loader_objs, device)
    if opt.phase == 'test_met':
        tester.infer_and_metrics()


def train(opt, netG, loader_objs, device):
    # Optimizer
    optimizer_netG = optim.Adamax(netG.parameters(), lr=opt.lr)
    
    train_dataloader = loader_objs['train_dataloader']
    test_dataloader = loader_objs['test_dataloader']
    n_train_batches = loader_objs['n_train_batches']
    n_test_batches = loader_objs['n_test_batches']
    total_step = 0
    start_time = time.time()

    # Discriminator
    sample_data = utils.get_next_batch(utils.get_data_dict(train_dataloader), opt=opt)['data_to_predict']

    _, opt.output_sequence, _, _, _ = sample_data.size()
    
    if opt.extrap and opt.input_sequence != opt.output_sequence: 
        opt.output_sequence -= 1

    netD_img, netD_seq, optimizer_netD = create_netD(opt, device)

    for epoch in range(opt.epoch):
        utils.update_learning_rate(optimizer_netG, decay_rate=0.99, lowest=opt.lr / 10)
        utils.update_learning_rate(optimizer_netD, decay_rate=0.99, lowest=opt.lr / 10)
        
        for it in range(n_train_batches):
            data_dict = utils.get_data_dict(train_dataloader)
            batch_dict = utils.get_next_batch(data_dict, opt=opt)

            res = netG.compute_all_losses(batch_dict)
            loss_netG = res['loss']
            
            # Compute Adversarial Loss
            real = batch_dict["data_to_predict"]
            fake = res["pred_y"]
            input_real = batch_dict["observed_data"]

            # Filter out mask
            if opt.irregular:
                b, _, c, h, w = real.size()
                observed_mask = batch_dict["observed_mask"]
                mask_predicted_data = batch_dict["mask_predicted_data"]

                selected_timesteps = int(observed_mask[0].sum())
                input_real = input_real[observed_mask.squeeze(-1).byte(), ...].view(b, selected_timesteps, c, h, w)
                real = real[mask_predicted_data.squeeze(-1).byte(), ...].view(b, selected_timesteps, c, h, w)

            loss_netD = opt.lamb_adv * netD_seq.netD_adv_loss(real, fake, input_real, opt.unequal)  
            loss_netD += opt.lamb_adv * netD_img.netD_adv_loss(real, fake, None)

            # Train D
            optimizer_netD.zero_grad()
            loss_netD.backward()
            optimizer_netD.step()
            
            loss_adv_netG = opt.lamb_adv * netD_seq.netG_adv_loss(real, fake, input_real, opt.unequal)
            loss_adv_netG += opt.lamb_adv * netD_img.netG_adv_loss(None, fake, None)
            loss_netG += loss_adv_netG
            
            # Train G
            optimizer_netG.zero_grad()
            loss_netG.backward()
            optimizer_netG.step()
            
            if (total_step + 1) % opt.log_print_freq == 0 or total_step == 0:
                et = time.time() - start_time
                et = str(datetime.timedelta(seconds=et))[:-7]
                log = f"Elapsed [{et}] Epoch [{epoch:03d}/{opt.epoch:03d}]\t"\
                        f"Iterations [{(total_step + 1):6d}] \t"\
                        f"Mse [{res['loss'].item():.4f}]\t"\
                        f"Adv_G [{loss_adv_netG.item():.4f}]\t"\
                        f"Adv_D [{loss_netD.item():.4f}]"
                
                print(log)

            if (total_step + 1) % opt.ckpt_save_freq == 0 or (epoch + 1 == opt.epoch and it + 1 == n_train_batches) or total_step == 0:
                utils.save_checkpoint(netG, os.path.join(opt.checkpoint_dir, f"ckpt_{(total_step + 1):08d}.pth"))
            
            if (total_step + 1) % opt.image_print_freq == 0 or total_step == 0:
                
                gt, pred, time_steps = visualize.make_save_sequence(opt, batch_dict, res)
                
                if opt.extrap:
                    visualize.save_extrap_images(opt=opt, gt=gt, pred=pred, path=opt.train_image_path, total_step=total_step)
                else:
                    visualize.save_interp_images(opt=opt, gt=gt, pred=pred, path=opt.train_image_path, total_step=total_step)
            
            total_step += 1
        
        # Test
        if (epoch + 1) % 100 == 0:
            test(netG, epoch, test_dataloader, opt, n_test_batches)

def test(netG, epoch, test_dataloader, opt, n_test_batches):
    
    # Select random index to save
    random_saving_idx = np.random.randint(0, n_test_batches, size=1)
    fix_saving_idx = 2
    test_losses = 0.0
    
    with torch.no_grad():
        for i in range(n_test_batches):
            data_dict = utils.get_data_dict(test_dataloader)
            batch_dict = utils.get_next_batch(data_dict)

            res = netG.compute_all_losses(batch_dict)
            test_losses += res["loss"].detach()

            if i == fix_saving_idx or i == random_saving_idx:
    
                gt, pred, time_steps = visualize.make_save_sequence(opt, batch_dict, res)

                if opt.extrap:
                    visualize.save_extrap_images(opt=opt, gt=gt, pred=pred, path=opt.test_image_path, total_step=100 * (epoch + 1) + i)
                else:
                    visualize.save_interp_images(opt=opt, gt=gt, pred=pred, path=opt.test_image_path, total_step=100 * (epoch + 1) + i)
                    
        test_losses /= n_test_batches

    print(f"[Test] Epoch [{epoch:03d}/{opt.epoch:03d}]\t" f"Loss {test_losses:.4f}\t")

if __name__ == '__main__':
    main()
