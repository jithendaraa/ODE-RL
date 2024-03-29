import sys
sys.path.append('..')

from helpers.utils import get_data_dict, get_next_batch
import torch
import torch.optim as optim
import torch.nn as nn
import argparse
import os
import json
import random
from torch.autograd import Variable
from torch.utils.data import DataLoader
# from utils_plot import plot_rec_new, plot_rec_exchange, plot_rec_fixed_motion, plot_rec_fixed_content, \
#     plot_rec_generating, plot_rec_generating2, plot_rec_exchange_paper, plot_rec_generating_paper, plot_rec_generating2_paper
import utils
import itertools
import progressbar
import numpy as np
import torch.nn.functional as F
import math
import time
from models.DS_VAE import DisentangledVAE_ICLR_V1
import models.dcgan_64 as model


parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='Sprite', help='dataset to train with(smmnist_fixed, smmnist_triplet, mmnist)')
parser.add_argument('--lr', default=0.002, type=float, help='learning rate')
parser.add_argument('--beta1', default=0.9, type=float, help='momentum term for adam')
parser.add_argument('--batch_size', default=4, type=int, help='batch size')
parser.add_argument('--log_dir', default='logs/lp', help='base directory to save logs')
parser.add_argument('--model_dir', default='', help='base directory to save logs')
parser.add_argument('--name', default='', help='identifier for directory')
parser.add_argument('--data_root', default='data', help='root directory for data')
parser.add_argument('--optimizer', default='adam', help='optimizer to train with')
parser.add_argument('--niter', type=int, default=300, help='number of epochs to train for')
parser.add_argument('--seed', default=1, type=int, help='manual seed')
parser.add_argument('--image_width', type=int, default=64, help='the height / width of the input image to network')

# mmnist args
parser.add_argument('--total_frames', default=2000000, type=int)
parser.add_argument('--train_test_split', default=0.8, type=float)
parser.add_argument('--test_seq', default=200, type=int)
parser.add_argument('--user_dir', default="/home/jithen" , type=str)
parser.add_argument('--storage_dir', default="scratch", type=str)
parser.add_argument('--data_dir', default="/home/jithen/scratch/datasets/MovingMNIST_video", type=str)
parser.add_argument('--train_in_seq', default=20, type=int)
parser.add_argument('--train_out_seq', default=20, type=int)
parser.add_argument('--test_in_seq', default=20, type=int)
parser.add_argument('--test_out_seq', default=180, type=int)
parser.add_argument('--frozen', default=True, type=bool)
parser.add_argument('--in_channels', default=1, type=int)






parser.add_argument('--channels', default=3, type=int)
parser.add_argument('--frames', type=int, default=8, help='number of frames, 8 for sprite, 15 for digits')

parser.add_argument('--rnn_size', type=int, default=256, help='dimensionality of hidden layer')
parser.add_argument('--prior_rnn_layers', type=int, default=1, help='number of layers')

parser.add_argument('--beta', type=float, default=0.0001, help='weighting on KL to prior')
parser.add_argument('--model', default='dcgan', help='model type (dcgan | vgg)')
parser.add_argument('--data_threads', type=int, default=1, help='number of data loading threads')
parser.add_argument('--num_digits', type=int, default=2, help='number of digits for moving mnist')
parser.add_argument('--last_frame_skip', action='store_true', help='if true, skip connections go between frame t and frame t+t rather than last ground truth frame')

parser.add_argument('--f_rnn_layers', type=int, default=1, help='number of layers (content lstm)')
parser.add_argument('--f_dim', type=int, default=256, help='dim of f')
parser.add_argument('--z_dim', type=int, default=32, help='dimensionality of z_t')
parser.add_argument('--g_dim', type=int, default=128, help='dimensionality of encoder output vector and decoder input vector')
parser.add_argument('--gpu', default='0', type=str, help='index of GPU to use')

parser.add_argument('--weight_OT', type=float, default=0, help='weighting on Optimal Transport, seq-level z distance')
parser.add_argument('--weight_f', type=float, default=1, help='weighting on KL to prior, content vector')
parser.add_argument('--weight_z', type=float, default=1, help='weighting on KL to prior, motion vector')
parser.add_argument('--weight_triple', type=float, default=100, help='100, weighting on triple loss of content vector')
parser.add_argument('--weight_motion_area', type=float, default=0.1, help='0.1, weighting on motion area loss of motion vector')
parser.add_argument('--weight_motion_dir', type=float, default=0.1, help='0.1, weighting on motion direction loss of motion vector')
parser.add_argument('--weight_MI', type=float, default=0.01, help='weighting on Mutual infomation of f and z')
parser.add_argument('--weight_GT_cls', type=float, default=0, help='weighting on digit recogniation classification loss')



opt = parser.parse_args()


os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpu
def get_batch_fixed(train_loader, dtype):
    while True:
        for sequence in train_loader:
            # sequence is uint8, range(0, 255),
            # convert is to float range(0, 1)
            data = sequence[0].float()
            batch = utils.normalize_data(opt, dtype, data)
            yield batch, sequence[1], sequence[2], sequence[-1]

def logsumexp(value, dim=None, keepdim=False):
    """Numerically stable implementation of the operation
    value.exp().sum(dim, keepdim).log()
    """
    if dim is not None:
        m, _ = torch.max(value, dim=dim, keepdim=True)
        value0 = value - m
        if keepdim is False:
            m = m.squeeze(dim)
        return m + torch.log(torch.sum(torch.exp(value0),
                                       dim=dim, keepdim=keepdim))
    else:
        raise ValueError('Must specify the dimension.')

def log_density(sample, mu, logsigma):
    mu = mu.type_as(sample)
    logsigma = logsigma.type_as(sample)
    c = torch.Tensor([np.log(2 * np.pi)]).type_as(sample.data)

    inv_sigma = torch.exp(-logsigma)
    tmp = (sample - mu) * inv_sigma
    return -0.5 * (tmp * tmp + 2 * logsigma + c)

def log_importance_weight_matrix(batch_size, dataset_size):
    N = dataset_size
    M = batch_size - 1
    strat_weight = (N - M) / (N * M)
    W = torch.Tensor(batch_size, batch_size).fill_(1 / M)
    W.view(-1)[::M+1] = 1 / N
    W.view(-1)[1::M+1] = strat_weight
    W[M-1, 0] = strat_weight
    return W.log()

def print_log(print_string, log=None):
  print("{}".format(print_string))
  if log is not None:
    log = open(log, 'a')
    log.write('{}\n'.format(print_string))
    log.close()

#from sinkhorn_OT import SinkhornDistance
def loss_fn_new(original_seq,recon_seq,f_mean,f_logvar,z_post_mean,z_post_logvar, z_post,
                z_prior_mean, z_prior_logvar, z_prior, opt):
    """
    Loss function consists of 3 parts, 
    the reconstruction term that is the MSE loss between the generated and the original images
    the KL divergence of f, 
    and the sum over the KL divergence of each z_t, with the sum divided by batch_size

    Loss = {mse + KL of f + sum(KL of z_t)} / batch_size
    Prior of f is a spherical zero mean unit variance Gaussian and the prior of each z_t is a Gaussian whose mean and variance
    are given by the LSTM
    """
    batch_size = original_seq.size(0)
    mse = F.mse_loss(recon_seq, original_seq, reduction='sum')

    f_mean = f_mean.view((-1, f_mean.shape[-1]))
    f_logvar = f_logvar.view((-1, f_logvar.shape[-1]))
    kld_f = -0.5 * torch.sum(1 + f_logvar - torch.pow(f_mean,2) - torch.exp(f_logvar))

    z_post_var = torch.exp(z_post_logvar)
    z_prior_var = torch.exp(z_prior_logvar)
    kld_z = 0.5 * torch.sum(z_prior_logvar - z_post_logvar + ((z_post_var + torch.pow(z_post_mean - z_prior_mean, 2)) / z_prior_var) - 1)
    ot_loss = 0

    ot_loss = torch.zeros((1)).cuda()
    return {'mse': mse/batch_size, 'kld_f': kld_f/batch_size, 'kld_z': kld_z/batch_size,
            'ot': ot_loss/batch_size}


triplet_loss = nn.TripletMarginLoss(margin=1.0, p=2).cuda()
CE_loss = nn.CrossEntropyLoss().cuda()

# --------- training funtions ------------------------------------
def train(x, label_A, label_D, label, mask, model, classifier, optimizer, optimizer_cls, opt):
    model.zero_grad()
    if classifier:  classifier.zero_grad()

    f_mean, f_logvar, f, z_post_mean, z_post_logvar,\
    z_post, z_prior_mean, z_prior_logvar, z_prior,\
    recon_x , pred_dir, pred_area = model(x) #pred

    if isinstance(x, list):
        x = x[0]
        f_mean_pos = f_mean[1]
        f_mean_neg = f_mean[2]
        f_mean = f_mean[0]

    x = x[0] if isinstance(x, list) else x
    loss_dict = loss_fn_new(x, recon_x, f_mean, f_logvar, z_post_mean, z_post_logvar, z_post, z_prior_mean,
                                 z_prior_logvar, z_prior, opt)

    mse = loss_dict['mse']
    kld_f = loss_dict['kld_f']
    kld_z = loss_dict['kld_z']
    ot = loss_dict['ot']

    if opt.weight_triple:
        trp_loss = triplet_loss(f_mean, f_mean_pos, f_mean_neg) 
    else:
        trp_loss = torch.zeros((1)).cuda() * opt.weight_triple
    
    motion_dir_loss = torch.zeros((1)).cuda()
    motion_area_loss = torch.zeros((1)).cuda()

    # print("Motion weights", opt.weight_motion_dir, opt.weight_motion_area)
    # print("MI", opt.weight_MI)
    
    if opt.weight_motion_dir or opt.weight_motion_area:
        if opt.dataset not in ['mmnist']:
            # predict the motion direction of each area
            print("Label", label.size())
            label_dir = label.permute(2, 0, 1).flatten()
            print("label_dir", label_dir.size(), label.size())

            print("mask", mask.size())
            mask_dir = mask.permute(2, 0, 1).flatten()
            print("mask_dir", mask_dir.size())
            
            pred_dir = pred_dir[mask_dir]
            label_dir = label_dir[mask_dir]
            print(pred_dir.size(), label_dir.size())

            motion_dir_loss = F.cross_entropy(pred_dir, label_dir.long())
            print("motion_dir_loss", motion_dir_loss.size(), label_dir[0], pred_dir[0])

        # predict which area are moving, remove the first frame(dummy)

        if opt.dataset in ['mmnist']:
            mask_area = mask.contiguous().view(-1, mask.shape[2]).cuda()
        else:
            mask_area = mask[:,1:,:].contiguous().view(-1, mask.shape[2]).cuda()
        
        pred_area = pred_area.view(opt.batch_size, -1, 9)[:,1:,:].contiguous().view(-1, mask.shape[2])
        
        # 3x3 grid
        motion_area_loss = F.binary_cross_entropy(torch.sigmoid(pred_area), mask_area.float())

    # calculate the mutual infomation of f and z
    batch_size, n_frame, z_dim = z_post_mean.size()
    mi_fz = torch.zeros((1)).cuda()

    if opt.weight_MI:
        # compute log q(z) ~= log 1/(NM) sum_m=1^M q(z|x_m) = - log(MN) + logsumexp_m(q(z|x_m))
        # batch_size x batch_size x f_dim
        # print(f.size(), f_mean.size(), f_logvar.size())

        _logq_f_tmp = log_density(f.unsqueeze(0).repeat(n_frame, 1, 1).view(n_frame, batch_size, 1, opt.f_dim),
                                  f_mean.unsqueeze(0).repeat(n_frame, 1, 1).view(n_frame, 1, batch_size, opt.f_dim),
                                  f_logvar.unsqueeze(0).repeat(n_frame, 1, 1).view(n_frame, 1, batch_size, opt.f_dim))

        # print("_logq_f_tmp", _logq_f_tmp.size())
        # print(z_post.size(), z_post_mean.size(), z_post_logvar.size())
        # n_frame x batch_size x batch_size x f_dim
        _logq_z_tmp = log_density(z_post.transpose(0, 1).view(n_frame, batch_size, 1, z_dim),
                                  z_post_mean.transpose(0, 1).view(n_frame, 1, batch_size, z_dim),
                                  z_post_logvar.transpose(0, 1).view(n_frame, 1, batch_size, z_dim))
        _logq_fz_tmp = torch.cat((_logq_f_tmp, _logq_z_tmp), dim=3)

        # print("log_f, log_z, log_zf", _logq_f_tmp.size(), _logq_z_tmp.size(), _logq_fz_tmp.size())

        logq_f = (logsumexp(_logq_f_tmp.sum(3), dim=2, keepdim=False) - math.log(batch_size * opt.dataset_size))
        logq_z = (logsumexp(_logq_z_tmp.sum(3), dim=2, keepdim=False) - math.log(batch_size * opt.dataset_size))
        logq_fz = (logsumexp(_logq_fz_tmp.sum(3), dim=2, keepdim=False) - math.log(batch_size * opt.dataset_size))
        # print("HA", logq_f.size(), logq_z.size(), logq_fz.size())

        # n_frame x batch_size
        # some sample are wired negative
        mi_fz = F.relu(logq_fz - logq_f - logq_z).mean()


    # multple label classification loss
    # if there is supervision with ID and attributes
    if classifier and optimizer_cls:
        # print("classifier", classifier)
        pred_action, pred_a1, pred_a2, pred_a3, pred_a4 =  classifier(z_post, f)
        cls_ｄ_loss = CE_loss(pred_action, label_D.cuda())
        cls_a1_loss = CE_loss(pred_a1, label_A[:,0].cuda())
        cls_a2_loss = CE_loss(pred_a2, label_A[:,1].cuda())
        cls_a3_loss = CE_loss(pred_a3, label_A[:,2].cuda())
        cls_a4_loss = CE_loss(pred_a4, label_A[:,3].cuda())
        cls_loss = (cls_ｄ_loss*4 + cls_a1_loss + cls_a2_loss + cls_a3_loss + cls_a4_loss)/8
    else:
        cls_loss = torch.zeros(1).cuda()

    # print("weights f z", opt.weight_f, opt.weight_z, opt.weight_motion_area, opt.weight_motion_dir, opt.weight_MI, opt.weight_GT_cls)
    # print("unweighted losses", mse, kld_f, kld_z, trp_loss / opt.weight_triple, motion_area_loss, motion_dir_loss, mi_fz, cls_loss)
    print("Unweighted losses")
    print("MSE:", mse.data.cpu().item(), "| Static KL:", kld_f.data.cpu().numpy(), "| Dynamic KL:", kld_z.data.cpu().numpy(), "| SCC Loss:", trp_loss.data.cpu().numpy(), "| Motion area loss:", motion_area_loss.data.cpu().numpy(), "| Motion dir loss:", motion_dir_loss.data.cpu().numpy(), "MI Loss:", mi_fz.data.cpu().numpy(), "| CLS Loss:", cls_loss.data.cpu().numpy())
    print()

    kld_f = kld_f * opt.weight_f
    kld_z = kld_z * opt.weight_z
    trp_loss = trp_loss * opt.weight_triple
    motion_area_loss = motion_area_loss * opt.weight_motion_area
    motion_dir_loss  = motion_dir_loss * opt.weight_motion_dir
    mi_fz = mi_fz * opt.weight_MI
    cls_loss = cls_loss * opt.weight_GT_cls

    loss = mse + kld_f + kld_z + trp_loss + motion_area_loss + motion_dir_loss + mi_fz + cls_loss
    loss.backward()
    optimizer.step()
    if optimizer_cls:
        optimizer_cls.step()
    ot_data = np.array(ot) if isinstance(ot, float) else ot.data.cpu().numpy()

    # print()
    # print("Weighted losses")
    # print("MSE:", mse.data.cpu(), "| Static KL:", kld_f.data.cpu().numpy(), "| Dynamic KL:", kld_z.data.cpu().numpy(), "| OT:", ot_data, "SCC Loss:", trp_loss.data.cpu().numpy(), "| Motion area loss:", motion_area_loss.data.cpu().numpy(), "| Motion dir loss:", motion_dir_loss.data.cpu().numpy(), "MI Loss:", mi_fz.data.cpu().numpy())
    # print("Total loss:", loss.item())
    # print()

    return mse.data.cpu().numpy(), kld_f.data.cpu().numpy(), kld_z.data.cpu().numpy(), ot_data,\
           trp_loss.data.cpu().numpy(), motion_area_loss.data.cpu().numpy(), motion_dir_loss.data.cpu().numpy(), mi_fz.data.cpu().numpy(), \
           cls_loss.data.cpu().numpy()


def get_dataloader(train_data, test_data, opt):
    if opt.dataset in ['mmnist']:
        return train_data, test_data
    else:
        train_loader = DataLoader(train_data,
                              num_workers=opt.data_threads,
                              batch_size=opt.batch_size,
                              shuffle=True,
                              drop_last=True,
                              pin_memory=True)
    
        test_loader = DataLoader(test_data,
                             num_workers=opt.data_threads,
                             batch_size=opt.batch_size,#8
                             shuffle=False,
                             drop_last=True,
                             pin_memory=True)
        
    return train_loader, test_loader


def main(opt):

    if opt.dataset in ['mmnist']:
        opt.channels = 1
        opt.frames = opt.train_in_seq
        opt.weight_motion_dir = 0
        # opt.weight_triple = 1000
        # opt.weight_motion_area = 100
        # opt.weight_MI = 1
    
    classifier = None
    optimizer_cls = None
    name = 'ICLR_V1_%s_model=%s%dx%d-rnn_size=%d-lr=%.4f-g_dim=%d-z_dim=%d' \
               '-weight:kl_f=%.2f-kl_z=%.2f-OT_z=%.2f-triple=%.2f-m_area=%.2f-m_dir=%.2f-mi=%.2f-cls=%.2f-%s' % (
               'triple' if opt.weight_triple else '', opt.model, opt.image_width, opt.image_width, opt.rnn_size,
               opt.lr, opt.g_dim, opt.z_dim, opt.weight_f, opt.weight_z, opt.weight_OT, opt.weight_triple, opt.weight_motion_area,
               opt.weight_motion_dir, opt.weight_MI, opt.weight_GT_cls, opt.name)

    opt.log_dir = '%s/%s/%s' % (opt.log_dir, opt.dataset, name)
    print("name:", name)

    log = os.path.join(opt.log_dir, 'log.txt')
    os.makedirs('%s/gen/' % opt.log_dir, exist_ok=True)
    os.makedirs('%s/plots/' % opt.log_dir, exist_ok=True)

    print_log("Random Seed: {}".format(opt.seed), log)
    random.seed(opt.seed)
    torch.manual_seed(opt.seed)
    torch.cuda.manual_seed_all(opt.seed)
    dtype = torch.cuda.FloatTensor

    print_log('Running parameters:')
    print_log(json.dumps(vars(opt), indent=4, separators=(',', ':')), log)

    # ---------------- optimizers ----------------
    opt.optimizer = optim.Adam

    ds_vae = DisentangledVAE_ICLR_V1(model, opt)
    ds_vae.apply(utils.init_weights)

    optimizer = opt.optimizer(ds_vae.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))

    # --------- transfer to gpu ------------------------------------
    if torch.cuda.device_count() > 1:
        print_log("Let's use {} GPUs!".format(torch.cuda.device_count()), log)
        ds_vae = nn.DataParallel(ds_vae)
    ds_vae = ds_vae.cuda()
    print_log(ds_vae, log)
    print("classifier", classifier)
    if classifier:
        classifier = classifier.cuda()
        print_log(classifier, log)

    # --------- load a dataset ------------------------------------

    if opt.dataset in ['mmnist']: 
        opt.frames = opt.train_in_seq
        loader_objs = utils.load_dataset(opt)
        train_data, test_data = loader_objs['train_dataloader'], loader_objs['test_dataloader']
        opt.dataset_size = loader_objs['n_train_batches']
        train_loader, test_loader = get_dataloader(train_data, test_data, opt)
    else:
        train_data, test_data = utils.load_dataset(opt)
        train_loader, test_loader = get_dataloader(train_data, test_data, opt)
        opt.dataset_size = len(train_loader)
        
    # testing_batch_generator = get_batch_fixed(test_loader, dtype)


    # --------- training loop ------------------------------------
    for epoch in range(opt.niter):
        ds_vae.train()
        epoch_mse = 0
        epoch_kld_f = 0
        epoch_kld_z = 0
        epoch_ot_z = 0
        epoch_m_area_loss = 0
        epoch_m_dir_loss = 0
        epoch_triple_loss = 0
        epoch_MI_loss = 0
        epoch_cls_loss = 0
        epoch_size = opt.dataset_size
        progress = progressbar.ProgressBar(max_value=epoch_size).start()
        start = time.time()
        print(f'epoch {epoch}')
        
        for i, batch in enumerate(train_loader):
            progress.update(i+1)

            if opt.weight_triple:
                x, x_pos, x_neg, label_A, label_D, label, mask = utils.get_data(batch, opt)
                x = [x, x_pos, x_neg]  # list
            else:
                data = batch[0].float()
                x = utils.normalize_data(opt, dtype, data)
                label_A = batch[1]
                label_D = batch[2]
                label = batch[3].cuda()
                mask = batch[4].cuda()
                x = torch.stack(x, dim=1).cuda()

            mse, kld_f, kld_z, ot_z, triple_loss, m_area_loss, m_dir_loss, mi_loss, cls_loss = train(x, label_A, label_D, label, mask, ds_vae, classifier, optimizer, optimizer_cls, opt)
            epoch_mse += mse
            epoch_kld_f += kld_f
            epoch_kld_z += kld_z
            epoch_ot_z += ot_z
            epoch_m_area_loss += m_area_loss
            epoch_m_dir_loss += m_dir_loss
            epoch_triple_loss += triple_loss
            epoch_MI_loss += mi_loss
            epoch_cls_loss += cls_loss

            if i%100 == 0 and i:
                print_log('[%02d] mse: %.5f | kld_f: %.5f | kld_z: %.5f | ot_z: %.5f | m_area: %.5f | m_dir: %.5f | trp: %.5f | mi: %.5f | cls: %.5f' % (
                epoch, mse.item(), kld_f.item(), kld_z.item(), ot_z.item(), m_area_loss.item(), m_dir_loss.item(), triple_loss.item(), mi_loss.item(),
                cls_loss.item()), log)
        
        print("Time for epoch", epoch, ":", time.time() - start, "s")
        progress.finish()
        utils.clear_progressbar()
        print_log('[%02d] mse: %.5f | kld_f: %.5f | kld_z: %.5f | ot_z: %.5f | m_area: %.5f | m_dir: %.5f | trp: %.5f | mi: %.5f | cls: %.5f (%d)' %
                                                                                (epoch, epoch_mse/epoch_size,
                                                                                epoch_kld_f/epoch_size,
                                                                                epoch_kld_z/epoch_size,
                                                                                epoch_ot_z/epoch_size,
                                                                                epoch_m_area_loss / epoch_size,
                                                                                epoch_m_dir_loss / epoch_size,
                                                                                epoch_triple_loss / epoch_size,
                                                                                epoch_MI_loss / epoch_size,
                                                                                epoch_cls_loss / epoch_size,
                                                                                epoch*epoch_size*opt.batch_size), log)
        # plot some stuff
        ds_vae.eval()
        # save the model
        net2save = ds_vae.module if torch.cuda.device_count() > 1 else ds_vae
        torch.save({
            'ds_vae': net2save},
            '%s/model.pth' % opt.log_dir)

        # x, label, mask, index = next(testing_batch_generator)
        # x = torch.stack(x[:15], dim=1).cuda()

        # net2test = ds_vae.module if torch.cuda.device_count() > 1 else ds_vae
        # plot_rec_new(x, epoch, opt, net2test)
        # plot_rec_exchange(x, epoch, opt, net2test)

        # plot_rec_fixed_motion(x, epoch, opt, net2test)
        # plot_rec_fixed_content(x, epoch, opt, net2test)
        # plot_rec_generating(x, epoch, opt, net2test)

        print('log dir: %s' % opt.log_dir)
        print()



if __name__ == '__main__':
    main(opt)
