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
#     plot_rec_generating, plot_rec_generating2
import utils
import itertools
import progressbar
import numpy as np
import torch.nn.functional as F
# from OP import IPOT_distance, cost_matrix

parser = argparse.ArgumentParser()
parser.add_argument('--lr', default=0.002, type=float, help='learning rate')
parser.add_argument('--beta1', default=0.9, type=float, help='momentum term for adam')
parser.add_argument('--batch_size', default=128, type=int, help='batch size')
parser.add_argument('--log_dir', default='logs/lp', help='base directory to save logs')
parser.add_argument('--model_dir', default='', help='base directory to save logs')
parser.add_argument('--name', default='', help='identifier for directory')
parser.add_argument('--data_root', default='data', help='root directory for data')
parser.add_argument('--optimizer', default='adam', help='optimizer to train with')
parser.add_argument('--niter', type=int, default=300, help='number of epochs to train for')
parser.add_argument('--seed', default=1, type=int, help='manual seed')
parser.add_argument('--image_width', type=int, default=64, help='the height / width of the input image to network')

parser.add_argument('--channels', default=3, type=int)
parser.add_argument('--dataset', default='Sprite', help='dataset to train with(smmnist_fixed, smmnist_triplet)')
parser.add_argument('--frames', type=int, default=8, help='number of frames, 8 for sprite, 15 for digits')


parser.add_argument('--rnn_size', type=int, default=256, help='dimensionality of hidden layer')
parser.add_argument('--prior_rnn_layers', type=int, default=1, help='number of layers')

parser.add_argument('--beta', type=float, default=0.0001, help='weighting on KL to prior')
parser.add_argument('--model', default='dcgan', help='model type (dcgan | vgg)')
parser.add_argument('--data_threads', type=int, default=5, help='number of data loading threads')
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
parser.add_argument('--weight_motion_area', type=float, default=0.1, help='weighting on motion area loss of motion vector')
parser.add_argument('--weight_motion_dir', type=float, default=0.1, help='weighting on motion direction loss of motion vector')
parser.add_argument('--weight_GT_cls', type=float, default=0, help='weighting on digit recogniation classification loss')
parser.add_argument('--type_gt',  type=str, default='action', help='action, skin, top, pant, hair')

opt = parser.parse_args()

opt.model_dir = "saved_model/Sprite/"


os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpu
def get_batch_fixed(train_loader, dtype):
    while True:
        for sequence in train_loader:
            # sequence is uint8, range(0, 255),
            # convert is to float range(0, 1)
            data = sequence[0].float()
            batch = utils.normalize_data(opt, dtype, data)
            yield batch, sequence[1], sequence[2]



def main(opt):
    if opt.model_dir != '':
        # load model and continue training from checkpoint
        saved_model = torch.load('%s/model.pth' % opt.model_dir)
        optimizer = opt.optimizer
        model_dir = opt.model_dir
        # opt = saved_model['opt']
        opt.optimizer = optimizer
        opt.model_dir = model_dir
        opt.log_dir = '%s/continued' % opt.log_dir
    else:
        name = 'ICLR_V1_%s_model=%s%dx%d-rnn_size=%d-lr=%.4f-g_dim=%d-z_dim=%d' \
               '-weight:kl_f=%.2f-kl_z=%.2f-OT_z=%.2f-triple=%.2f-m_area=%.2f-m_dir=%.2f-cls=%.2f-%s' % (
               'triple' if opt.weight_triple else '', opt.model, opt.image_width, opt.image_width, opt.rnn_size,
               opt.lr, opt.g_dim, opt.z_dim, opt.weight_f, opt.weight_z, opt.weight_OT, opt.weight_triple, opt.weight_motion_area,
               opt.weight_motion_dir, opt.weight_GT_cls, opt.name)

        opt.log_dir = '%s/%s/%s' % (opt.log_dir, opt.dataset, name)

    log = os.path.join(opt.log_dir, 'log.txt')
    os.makedirs('%s/gen/' % opt.log_dir, exist_ok=True)
    os.makedirs('%s/plots/' % opt.log_dir, exist_ok=True)

    # print_log("Random Seed: {}".format(opt.seed), log)
    # random.seed(opt.seed)
    # torch.manual_seed(opt.seed)
    # torch.cuda.manual_seed_all(opt.seed)
    dtype = torch.cuda.FloatTensor

    print_log('Running parameters:')
    print_log(json.dumps(vars(opt), indent=4, separators=(',', ':')), log)

    # ---------------- optimizers ----------------
    if opt.optimizer == 'adam':
        opt.optimizer = optim.Adam
    elif opt.optimizer == 'rmsprop':
        opt.optimizer = optim.RMSprop
    elif opt.optimizer == 'sgd':
        opt.optimizer = optim.SGD
    else:
        raise ValueError('Unknown optimizer: %s' % opt.optimizer)

    # import models.dcgan_64 as model
    #
    #
    # from models.DS_VAE import DisentangledVAE_New, DisentangledVAE_New_fixed,\
    #             DisentangledVAE_New_fixed_Z_notDependonF_PlusLSTM, DisentangledVAE_ICLR, DisentangledVAE_ICLR_V1
    #
    # ds_vae = DisentangledVAE_ICLR_V1(model, opt)
    # ds_vae.apply(utils.init_weights)

    if opt.model_dir != '':
        ds_vae =  saved_model['ds_vae']


    # --------- transfer to gpu ------------------------------------
    if torch.cuda.device_count() > 1:
        print_log("Let's use {} GPUs!".format(torch.cuda.device_count()), log)
        ds_vae = nn.DataParallel(ds_vae)
    ds_vae = ds_vae.cuda()
    #print_log(ds_vae, log)

    # --------- load a dataset ------------------------------------
    train_data, test_data = utils.load_dataset(opt)
    # train_loader = DataLoader(train_data,
    #                           num_workers=opt.data_threads,
    #                           batch_size=opt.batch_size,
    #                           shuffle=True,
    #                           drop_last=True,
    #                           pin_memory=True)
    test_loader = DataLoader(test_data,
                             num_workers=opt.data_threads,
                             batch_size=opt.batch_size,
                             shuffle=True,
                             drop_last=True,
                             pin_memory=True)


    # testing_batch_generator = get_batch_fixed(test_loader, dtype)
    # for epoch in range(opt.niter):
    #     for sequence in test_loader:
    #         data1 = sequence[0].float()
    #         batch1 = utils.normalize_data(opt, dtype, data1)
    #         x = torch.stack(batch1[:15], dim=1).cuda()
    #
    #         # plot(x, epoch)
    #
    #         net2test = ds_vae.module if torch.cuda.device_count() > 1 else ds_vae
    #         plot_rec_new(x, epoch, opt, net2test)
    #         plot_rec_exchange(x, epoch, opt, net2test)
    #         plot_rec_fixed_motion(x, epoch, opt, net2test)
    #         plot_rec_fixed_content(x, epoch, opt, net2test)
    #         plot_rec_generating(x, epoch, opt, net2test)
    #         plot_rec_generating2(x, epoch, opt, net2test)
    #         a = 1

    print("here1")

    from video_classifier_Sprite_all import classifier_Sprite_all
    print("here2")
    # if opt.type_gt == 'action':
    #     opt.nlabel = 9
    # else:
    #     opt.nlabel = 6
    opt.g_dim = 128
    opt.rnn_size = 256
    classifier = classifier_Sprite_all(opt)
    opt.resume = 'saved_model/video_cls/Sprite/5branch/Epoch_4_100.00.tar'
    loaded_dict = torch.load(opt.resume)
    classifier.load_state_dict(loaded_dict['state_dict'])
    classifier = classifier.cuda().eval()
    
   
    
    # --------- training loop ------------------------------------
    for epoch in range(opt.niter):

        ds_vae.eval()
        label1_all, label2_all, label3_all = list(), list(), list()
        pred1_all, pred2_all, pred3_all = list(), list(), list()
        label_gt = list()
        for sequence in test_loader:
            #print(sequence.keys())
            data1 = sequence[0].float()
            label_A = sequence[1]
            label_D = sequence[2]

            batch1 = utils.normalize_data(opt, dtype, data1)
            x = torch.stack(batch1, dim=1).cuda()

            """ #1 change"""
            if opt.type_gt == "action":
                recon_x_sample, recon_x = ds_vae.forward_fixed_action_for_classification(x)
            else:
                recon_x_sample, recon_x = ds_vae.forward_fixed_content_for_classification(x)
            
            with torch.no_grad():
                """ #2 change"""
                pred1_1, pred1_2, pred1_3, pred1_4, pred1_5 = classifier(x)
                pred2_1, pred2_2, pred2_3, pred2_4, pred2_5 = classifier(recon_x_sample)
                pred3_1, pred3_2, pred3_3, pred3_4, pred3_5 = classifier(recon_x)
                if opt.type_gt == "action":
                    pred1, pred2, pred3 = pred1_1, pred2_1, pred3_1
                elif opt.type_gt == "skin":
                    pred1, pred2, pred3 = pred1_2, pred2_2, pred3_2
                elif opt.type_gt == "top":
                    pred1, pred2, pred3 = pred1_3, pred2_3, pred3_3
                elif opt.type_gt == "pant":
                    pred1, pred2, pred3 = pred1_4, pred2_4, pred3_4
                elif opt.type_gt == "hair":
                    pred1, pred2, pred3 = pred1_5, pred2_5, pred3_5

                pred1 = F.softmax(pred1, dim = 1)
                pred2 = F.softmax(pred2, dim = 1)
                pred3 = F.softmax(pred3, dim = 1)

            label1 = np.argmax(pred1.detach().cpu().numpy(), axis=1)
            label2 = np.argmax(pred2.detach().cpu().numpy(), axis=1)
            label3 = np.argmax(pred3.detach().cpu().numpy(), axis=1)

            pred1_all.append(pred1.detach().cpu().numpy())
            pred2_all.append(pred2.detach().cpu().numpy())
            pred3_all.append(pred3.detach().cpu().numpy())

            """ #3 change"""
            if opt.type_gt == "action":
                label_gt.append(label_D.numpy())
            elif opt.type_gt == "skin":
                label_gt.append(label_A[:,0].numpy())
            elif opt.type_gt == "top":
                label_gt.append(label_A[:,1].numpy())
            elif opt.type_gt == "pant":
                label_gt.append(label_A[:,2].numpy())
            elif opt.type_gt == "hair":
                label_gt.append(label_A[:,3].numpy())
            
            label1_all.append(label1)
            label2_all.append(label2)
            label3_all.append(label3)
        label1_all = np.hstack(label1_all)
        label2_all = np.hstack(label2_all)
        label3_all = np.hstack(label3_all)
        label_gt = np.hstack(label_gt)

        pred1_all = np.vstack(pred1_all)
        pred2_all = np.vstack(pred2_all)
        pred3_all = np.vstack(pred3_all)

        #acc = (label_gt == label2_all).mean()
        acc = (label1_all == label2_all).mean()
        kl  = KL_divergence(pred2_all, pred1_all)

        """These scores are influented by label distribution. select pred2_all with uniform label distribution"""
        nSample_per_cls = min([(label_gt==i).sum() for i in np.unique(label_gt)])
        #print(nSample_per_cls)
        index  = np.hstack([np.nonzero(label_gt == i)[0][:nSample_per_cls] for i in np.unique(label_gt)]).squeeze()
        pred2_selected = pred2_all[index]

        IS  = inception_score(pred2_selected)
        H_yx = entropy_Hyx(pred2_selected)
        H_y = entropy_Hy(pred2_selected)

        print('acc: {:.2f}%, kl: {:.4f}, IS: {:.4f}, H_yx: {:.4f}, H_y: {:.4f}'.format(acc*100, kl, IS, H_yx, H_y))
        a = 1

def entropy_Hy(p_yx, eps=1E-16):
    p_y = p_yx.mean(axis=0)
    sum_h = (p_y * np.log(p_y + eps)).sum() * (-1)
    return sum_h

def entropy_Hyx(p, eps=1E-16):
    sum_h = (p * np.log(p + eps)).sum(axis = 1)
    # average over images
    avg_h = np.mean(sum_h) * (-1)
    return avg_h

def inception_score(p_yx,  eps=1E-16):
    # calculate p(y)
    p_y = np.expand_dims(p_yx.mean(axis=0), 0)
    # kl divergence for each image
    kl_d = p_yx * (np.log(p_yx + eps) - np.log(p_y + eps))
    # sum over classes
    sum_kl_d = kl_d.sum(axis=1)
    # average over images
    avg_kl_d = np.mean(sum_kl_d)
    # undo the logs
    is_score = np.exp(avg_kl_d)
    return is_score

def KL_divergence(P, Q, eps=1E-16):
    kl_d = P * (np.log(P + eps) - np.log(Q + eps))
    # sum over classes
    sum_kl_d = kl_d.sum(axis=1)
    # average over images
    avg_kl_d = np.mean(sum_kl_d)
    return avg_kl_d

def print_log(print_string, log=None):
    print("{}".format(print_string))
    if log is not None:
        log = open(log, 'a')
        log.write('{}\n'.format(print_string))
        log.close()

if __name__ == '__main__':
    main(opt)
