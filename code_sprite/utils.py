import math
import torch
import socket
import argparse
import os
import numpy as np
from sklearn.manifold import TSNE
import scipy.misc
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import functools
from skimage.measure import compare_psnr as psnr_metric
from skimage.measure import compare_ssim as ssim_metric
from scipy import signal
from scipy import ndimage
from PIL import Image, ImageDraw


from torchvision import datasets, transforms
from torch.autograd import Variable
import imageio


hostname = socket.gethostname()

def load_dataset(opt):
    if opt.dataset == 'smmnist':
        from data.moving_mnist import MovingMNIST, MovingMNIST_Label_Motion
        train_data = MovingMNIST_Label_Motion(
                train=True,
                data_root=opt.data_root,
                seq_len=opt.n_past+opt.n_future,
                image_size=opt.image_width,
                deterministic=False,
                num_digits=opt.num_digits)
        test_data = MovingMNIST_Label_Motion(
                train=False,
                data_root=opt.data_root,
                seq_len=opt.n_eval,
                image_size=opt.image_width,
                deterministic=False,
                num_digits=opt.num_digits)
    elif opt.dataset == 'smmnist_fixed':
        from data.moving_mnist import MovingMNIST_Fixed
        train_data = MovingMNIST_Fixed(train=True, data=opt.train_data, GT_label = opt.train_digit,
                                       OF_label=opt.train_OF_label, mask=opt.train_OF_mask, triple=opt.weight_triple)
        test_data = MovingMNIST_Fixed(train=False, data=opt.test_data, GT_label = opt.train_digit,
                                       OF_label=opt.train_OF_label, mask=opt.train_OF_mask, triple=opt.weight_triple)

    elif opt.dataset == 'smmnist_fixed_cls':
        from data.moving_mnist import MovingMNIST_Fixed_Cls
        train_data = MovingMNIST_Fixed_Cls(data=opt.train_data, GT_label = opt.train_digit)
        test_data  = MovingMNIST_Fixed_Cls(data=opt.test_data,  GT_label = opt.test_digit)


    elif opt.dataset == 'Sprite':
        from data.sprite import Sprite
        from load_sprites import sprites_act
        X_train, X_test, A_train, A_test, D_train, D_test = sprites_act('dataset/Sprite/', return_labels=True)
        A_train = A_train[:, 0, :, :] # Frame 0 to 8 repeated
        A_test  = A_test[:, 0, :, :]

        D_train = D_train[:, 0, :]
        D_test = D_test[:, 0, :]

        # convert one-hot vector to index label
        A_train = np.argmax(A_train, axis=2)
        A_test = np.argmax(A_test, axis=2)
        D_train = np.argmax(D_train, axis=1)
        D_test = np.argmax(D_test, axis=1)

        import pickle
        # pickle.dump({'train': X_train, 'test': X_test}, open('sprite_data.pkl', 'wb'))
        OF_label = pickle.load(open('dataset/dsprite_train_OF_Label_3block.pkl', 'rb'))
        print("OF_Label", OF_label.shape)

        dummy = np.ones((OF_label.shape[0], 1, OF_label.shape[2])) * (-1)
        opt.train_OF_label = np.concatenate((dummy, OF_label), axis=1)
        opt.train_OF_mask = opt.train_OF_label != -1

        train_data = Sprite(train=True, data = X_train, A_label = A_train ,
                            D_label = D_train, OF_label = opt.train_OF_label, mask = opt.train_OF_mask, triple = opt.weight_triple)
        test_data = Sprite(train=False, data = X_test, A_label = A_test, D_label = D_test,
                           OF_label = opt.train_OF_label, mask = opt.train_OF_mask,  triple = opt.weight_triple)


    elif opt.dataset == 'MUG':
        import pickle
        from data.MUG import MUG
        data_store = pickle.load(open('dataset/MUG/MUG_dataset_video_label_identity.pkl', 'rb'))
        X_train = list()
        X_motion = list()
        X_content = list()
        for data in data_store:
            X_train.append(data['video'][:15])
            X_motion.append(data['label'])
            X_content.append(data['identity'])
        X_train = np.stack(X_train, axis=0)
        X_motion = np.array(X_motion)
        X_content = np.array(X_content)

        # so far, only provide a dummy info
        opt.train_OF_label = np.ones((X_train.shape[0], X_train.shape[1], 9)) * (-1)
        opt.train_OF_mask = opt.train_OF_label != -1

        train_data = MUG(train=True, data = X_train, GT_label = X_content, # content groundtruth
                            OF_label = opt.train_OF_label, mask = opt.train_OF_mask, triple = opt.weight_triple)

        test_data  = MUG(train=True, data = X_train, GT_label = X_content, # content groundtruth
                            OF_label = opt.train_OF_label, mask = opt.train_OF_mask, triple = opt.weight_triple)

    return train_data, test_data

def sequence_input(seq, dtype):
    return [Variable(x.type(dtype)) for x in seq]

def normalize_data(opt, dtype, sequence):
    if opt.dataset == 'smmnist' or opt.dataset == 'smmnist_triplet' or opt.dataset == 'smmnist_fixed'\
        or opt.dataset == 'kth' or opt.dataset == 'bair' or opt.dataset == 'Sprite' or opt.dataset == 'MUG'  :
        sequence.transpose_(0, 1)
        sequence.transpose_(3, 4).transpose_(2, 3)
    else:
        sequence.transpose_(0, 1)

    return sequence_input(sequence, dtype)

def is_sequence(arg):
    return (not hasattr(arg, "strip") and
            not type(arg) is np.ndarray and
            not hasattr(arg, "dot") and
            (hasattr(arg, "__getitem__") or
            hasattr(arg, "__iter__")))

def image_tensor(inputs, padding=1):
    # assert is_sequence(inputs)
    assert len(inputs) > 0
    # print(inputs)

    # if this is a list of lists, unpack them all and grid them up
    if is_sequence(inputs[0]) or (hasattr(inputs, "dim") and inputs.dim() > 4):
        images = [image_tensor(x) for x in inputs]
        if images[0].dim() == 3:
            c_dim = images[0].size(0)
            x_dim = images[0].size(1)
            y_dim = images[0].size(2)
        else:
            c_dim = 1
            x_dim = images[0].size(0)
            y_dim = images[0].size(1)

        result = torch.ones(c_dim,
                            x_dim * len(images) + padding * (len(images)-1),
                            y_dim)
        for i, image in enumerate(images):
            result[:, i * x_dim + i * padding :
                   (i+1) * x_dim + i * padding, :].copy_(image)

        return result

    # if this is just a list, make a stacked image
    else:
        images = [x.data if isinstance(x, torch.autograd.Variable) else x
                  for x in inputs]
        # print(images)
        if images[0].dim() == 3:
            c_dim = images[0].size(0)
            x_dim = images[0].size(1)
            y_dim = images[0].size(2)
        else:
            c_dim = 1
            x_dim = images[0].size(0)
            y_dim = images[0].size(1)

        result = torch.ones(c_dim,
                            x_dim,
                            y_dim * len(images) + padding * (len(images)-1))
        for i, image in enumerate(images):
            result[:, :, i * y_dim + i * padding :
                   (i+1) * y_dim + i * padding].copy_(image)
        return result

def save_np_img(fname, x):
    if x.shape[0] == 1:
        x = np.tile(x, (3, 1, 1))
    img = scipy.misc.toimage(x,
                             high=255*x.max(),
                             channel_axis=0)
    img.save(fname)

def make_image(tensor):
    tensor = tensor.cpu().clamp(0, 1)
    if tensor.size(0) == 1:
        tensor = tensor.expand(3, tensor.size(1), tensor.size(2))
    # pdb.set_trace()
    return scipy.misc.toimage(tensor.numpy(),
                              high=255*float(tensor.max()),
                              channel_axis=0)

def draw_text_tensor(tensor, text):
    np_x = tensor.transpose(0, 1).transpose(1, 2).data.cpu().numpy()
    pil = Image.fromarray(np.uint8(np_x*255))
    draw = ImageDraw.Draw(pil)
    draw.text((4, 64), text, (0,0,0))
    img = np.asarray(pil)
    return Variable(torch.Tensor(img / 255.)).transpose(1, 2).transpose(0, 1)

def save_gif(filename, inputs, duration=0.25):
    images = []
    for tensor in inputs:
        img = image_tensor(tensor, padding=0)
        img = img.cpu()
        img = img.transpose(0,1).transpose(1,2).clamp(0,1)
        images.append(img.numpy())
    imageio.mimsave(filename, images, duration=duration)

def save_gif_with_text(filename, inputs, text, duration=0.25):
    images = []
    for tensor, text in zip(inputs, text):
        img = image_tensor([draw_text_tensor(ti, texti) for ti, texti in zip(tensor, text)], padding=0)
        img = img.cpu()
        img = img.transpose(0,1).transpose(1,2).clamp(0,1).numpy()
        images.append(img)
    imageio.mimsave(filename, images, duration=duration)

def save_image(filename, tensor):
    img = make_image(tensor)
    img.save(filename)

def save_tensors_image(filename, inputs, padding=1):
    images = image_tensor(inputs, padding)
    return save_image(filename, images)

def prod(l):
    return functools.reduce(lambda x, y: x * y, l)

def batch_flatten(x):
    return x.resize(x.size(0), prod(x.size()[1:]))

def clear_progressbar():
    # moves up 3 lines
    print("\033[2A")
    # deletes the whole line, regardless of character position
    print("\033[2K")
    # moves up two lines again
    print("\033[2A")

def mse_metric(x1, x2):
    err = np.sum((x1 - x2) ** 2)
    err /= float(x1.shape[0] * x1.shape[1] * x1.shape[2])
    return err

def eval_seq(gt, pred):
    T = len(gt)
    bs = gt[0].shape[0]
    ssim = np.zeros((bs, T))
    psnr = np.zeros((bs, T))
    mse = np.zeros((bs, T))
    for i in range(bs):
        for t in range(T):
            for c in range(gt[t][i].shape[0]):
                ssim[i, t] += ssim_metric(gt[t][i][c], pred[t][i][c])
                psnr[i, t] += psnr_metric(gt[t][i][c], pred[t][i][c])
            ssim[i, t] /= gt[t][i].shape[0]
            psnr[i, t] /= gt[t][i].shape[0]
            mse[i, t] = mse_metric(gt[t][i], pred[t][i])

    return mse, ssim, psnr

# ssim function used in Babaeizadeh et al. (2017), Fin et al. (2016), etc.
def finn_eval_seq(gt, pred):
    T = len(gt)
    bs = gt[0].shape[0]
    ssim = np.zeros((bs, T))
    psnr = np.zeros((bs, T))
    mse = np.zeros((bs, T))
    for i in range(bs):
        for t in range(T):
            for c in range(gt[t][i].shape[0]):
                res = finn_ssim(gt[t][i][c], pred[t][i][c]).mean()
                if math.isnan(res):
                    ssim[i, t] += -1
                else:
                    ssim[i, t] += res
                psnr[i, t] += finn_psnr(gt[t][i][c], pred[t][i][c])
            ssim[i, t] /= gt[t][i].shape[0]
            psnr[i, t] /= gt[t][i].shape[0]
            mse[i, t] = mse_metric(gt[t][i], pred[t][i])

    return mse, ssim, psnr


def finn_psnr(x, y):
    mse = ((x - y)**2).mean()
    return 10*np.log(1/mse)/np.log(10)


def gaussian2(size, sigma):
    A = 1/(2.0*np.pi*sigma**2)
    x, y = np.mgrid[-size//2 + 1:size//2 + 1, -size//2 + 1:size//2 + 1]
    g = A*np.exp(-((x**2/(2.0*sigma**2))+(y**2/(2.0*sigma**2))))
    return g

def fspecial_gauss(size, sigma):
    x, y = np.mgrid[-size//2 + 1:size//2 + 1, -size//2 + 1:size//2 + 1]
    g = np.exp(-((x**2 + y**2)/(2.0*sigma**2)))
    return g/g.sum()
  
def finn_ssim(img1, img2, cs_map=False):
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    size = 11
    sigma = 1.5
    window = fspecial_gauss(size, sigma)
    K1 = 0.01
    K2 = 0.03
    L = 1 #bitdepth of image
    C1 = (K1*L)**2
    C2 = (K2*L)**2
    mu1 = signal.fftconvolve(img1, window, mode='valid')
    mu2 = signal.fftconvolve(img2, window, mode='valid')
    mu1_sq = mu1*mu1
    mu2_sq = mu2*mu2
    mu1_mu2 = mu1*mu2
    sigma1_sq = signal.fftconvolve(img1*img1, window, mode='valid') - mu1_sq
    sigma2_sq = signal.fftconvolve(img2*img2, window, mode='valid') - mu2_sq
    sigma12 = signal.fftconvolve(img1*img2, window, mode='valid') - mu1_mu2
    if cs_map:
        return (((2*mu1_mu2 + C1)*(2*sigma12 + C2))/((mu1_sq + mu2_sq + C1)*
                    (sigma1_sq + sigma2_sq + C2)), 
                (2.0*sigma12 + C2)/(sigma1_sq + sigma2_sq + C2))
    else:
        return ((2*mu1_mu2 + C1)*(2*sigma12 + C2))/((mu1_sq + mu2_sq + C1)*
                    (sigma1_sq + sigma2_sq + C2))

import torch.nn as nn
def init_weights(model):
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.normal_(m.weight, 0, 0.01)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, 0, 0.01)
            nn.init.constant_(m.bias, 0)
