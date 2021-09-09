import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader

import argparse
import os
import pickle
import numpy as np
import progressbar

import utils

parser = argparse.ArgumentParser()
parser.add_argument('--lr', default=0.0002, type=float, help='learning rate')
parser.add_argument('--beta1', default=0.9, type=float, help='momentum term for adam')
parser.add_argument('--batch_size', default=128, type=int, help='batch size')
parser.add_argument('--log_dir', default='logs/video_cls', help='base directory to save logs')
parser.add_argument('--model_dir', default='', help='base directory to save logs')
parser.add_argument('--name', default='', help='identifier for directory')
parser.add_argument('--data_root', default='data', help='root directory for data')
parser.add_argument('--nEpoch', type=int, default=100, help='number of epochs to train for')
parser.add_argument('--seed', default=1, type=int, help='manual seed')
parser.add_argument('--epoch_size', type=int, default=600, help='epoch size')
parser.add_argument('--image_width', type=int, default=64, help='the height / width of the input image to network')
parser.add_argument('--resume', default='', help='base directory to trained model')


parser.add_argument('--dataset', default='Sprite', help='dataset to train with(smmnist_fixed, smmnist_triplet)')
parser.add_argument('--frames', type=int, default=8, help='number of frames, 8 for sprite, 15 for digits')
parser.add_argument('--data_threads', type=int, default=5, help='number of data loading threads')
parser.add_argument('--gpu', default='0', type=str, help='index of GPU to use')

parser.add_argument('--g_dim',    type=int, default=128, help='dimensionality of encoder output vector and decoder input vector')
parser.add_argument('--rnn_size', type=int, default=256, help='dimensionality of hidden layer')
parser.add_argument('--channels', type=int, default=3)
parser.add_argument('--weight_triple', type=float, default=0)
parser.add_argument('--type_gt',  type=str, default='dummy', help='dummy')

opt = parser.parse_args()
os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpu
class classifier_Sprite(nn.Module):
    def __init__(self, opt):
        super(classifier_Sprite, self).__init__()
        self.g_dim = opt.g_dim  # frame feature
        self.channels = opt.channels  # frame feature
        self.hidden_dim = opt.rnn_size
        self.frames = opt.frames
        from models.dcgan_64_new import encoder
        self.encoder = encoder(self.g_dim, self.channels)
        self.bilstm = nn.LSTM(self.g_dim, self.hidden_dim, 1, bidirectional=True, batch_first=True)

        self.FCs = nn.Sequential(
            nn.Linear(self.hidden_dim * 2, self.hidden_dim),
            nn.ReLU(True),
            nn.Linear(self.hidden_dim, opt.nlabel))


    def encoder_frame(self, x):
        # input x is list of length Frames [batchsize, channels, size, size]
        # convert it to [batchsize, frames, channels, size, size]
        # x = torch.stack(x, dim=1)
        # [batch_size, frames, channels, size, size] to [batch_size * frames, channels, size, size]
        x_shape = x.shape
        x = x.view(-1, x_shape[-3], x_shape[-2], x_shape[-1])
        x_embed = self.encoder(x)[0]
        # to [batch_size , frames, embed_dim]
        return x_embed.view(x_shape[0], x_shape[1], -1)

    def forward(self, x):
        conv_x = self.encoder_frame(x)
        # pass the bidirectional lstm
        lstm_out, _ = self.bilstm(conv_x)
        backward = lstm_out[:, 0, self.hidden_dim:2 * self.hidden_dim]
        frontal = lstm_out[:, self.frames - 1, 0:self.hidden_dim]
        lstm_out_f = torch.cat((frontal, backward), dim=1)
        return self.FCs(lstm_out_f)

class classifier_Sprite_all(nn.Module):
    def __init__(self, opt):
        super(classifier_Sprite_all, self).__init__()
        self.g_dim = opt.g_dim  # frame feature
        self.channels = opt.channels  # frame feature
        self.hidden_dim = opt.rnn_size
        self.frames = opt.frames
        from models.dcgan_64 import encoder
        self.encoder = encoder(self.g_dim, self.channels)
        self.bilstm = nn.LSTM(self.g_dim, self.hidden_dim, 1, bidirectional=True, batch_first=True)
        self.cls_skin = nn.Sequential(
            nn.Linear(self.hidden_dim * 2, self.hidden_dim),
            nn.ReLU(True),
            nn.Linear(self.hidden_dim, 6))
        self.cls_top = nn.Sequential(
            nn.Linear(self.hidden_dim * 2, self.hidden_dim),
            nn.ReLU(True),
            nn.Linear(self.hidden_dim, 6))
        self.cls_pant = nn.Sequential(
            nn.Linear(self.hidden_dim * 2, self.hidden_dim),
            nn.ReLU(True),
            nn.Linear(self.hidden_dim, 6))
        self.cls_hair = nn.Sequential(
            nn.Linear(self.hidden_dim * 2, self.hidden_dim),
            nn.ReLU(True),
            nn.Linear(self.hidden_dim, 6))
        self.cls_action = nn.Sequential(
            nn.Linear(self.hidden_dim * 2, self.hidden_dim),
            nn.ReLU(True),
            nn.Linear(self.hidden_dim, 9))


    def encoder_frame(self, x):
        # input x is list of length Frames [batchsize, channels, size, size]
        # convert it to [batchsize, frames, channels, size, size]
        # x = torch.stack(x, dim=1)
        # [batch_size, frames, channels, size, size] to [batch_size * frames, channels, size, size]
        x_shape = x.shape
        x = x.view(-1, x_shape[-3], x_shape[-2], x_shape[-1])
        x_embed = self.encoder(x)[0]
        # to [batch_size , frames, embed_dim]
        return x_embed.view(x_shape[0], x_shape[1], -1)

    def forward(self, x):
        conv_x = self.encoder_frame(x)
        # pass the bidirectional lstm
        lstm_out, _ = self.bilstm(conv_x)
        backward = lstm_out[:, 0, self.hidden_dim:2 * self.hidden_dim]
        frontal = lstm_out[:, self.frames - 1, 0:self.hidden_dim]
        lstm_out_f = torch.cat((frontal, backward), dim=1)
        return self.cls_action(lstm_out_f), self.cls_skin(lstm_out_f), self.cls_pant(lstm_out_f), \
               self.cls_top(lstm_out_f), self.cls_hair(lstm_out_f)

def get_accuracy(pred, label):
    # it's 1 only when two digits are correctly predicted
    pred = pred.detach().cpu().numpy()
    idx_mat = np.argsort(-1 * pred, axis=1)
    label_mat = (idx_mat[:, 0:2]).astype(int)
    label_mat = np.sort(label_mat, axis=1) #e.g., [6,2] => [2,6]
    label = label.numpy()
    acc = ((label_mat == label).sum(1) == 2).mean()
    return acc


def main():
    opt.log_dir = os.path.join(opt.log_dir, opt.dataset, '5branch')
    os.makedirs(opt.log_dir, exist_ok=True)
    # --------- load a dataset ------------------------------------
    train_data, test_data = utils.load_dataset(opt)
    train_loader = DataLoader(train_data,
                              num_workers=opt.data_threads,
                              batch_size=opt.batch_size,
                              shuffle=True,
                              drop_last=True,
                              pin_memory=True)
    test_loader = DataLoader(test_data,
                             num_workers=opt.data_threads,
                             batch_size=opt.batch_size,
                             shuffle=True,
                             drop_last=True,
                             pin_memory=True)


    net = classifier_Sprite_all(opt)

    if opt.resume:
        loaded_dict = torch.load(opt.resume)
        net.load_state_dict(loaded_dict['state_dict'])

    net = net.cuda()
    print(net)
    criterion = nn.CrossEntropyLoss().cuda()
    dtype = torch.cuda.FloatTensor
    optimizer = optim.Adam(net.parameters(), lr=opt.lr)
    for epoch in range(opt.nEpoch):
        net.train()
        mean_loss = 0
        mean_acc0, mean_acc1, mean_acc2, mean_acc3, mean_acc4 = 0, 0, 0, 0, 0
        cnt = 0
        progress = progressbar.ProgressBar(max_value=len(train_loader)).start()
        for idx, batch in enumerate(train_loader):
            progress.update(idx + 1)
            net.zero_grad()
            data = batch[0].float()
            x = utils.normalize_data(opt, dtype, data)
            label_A = batch[1].cuda()
            label_D = batch[2].cuda()
            # label = batch[3].cuda()
            # mask = batch[4].cuda()
            x = torch.stack(x, dim=1).cuda()
            pred_action, pred_skin, pred_pant, pred_top, pred_hair = net(x)
            # ['body', 'bottomwear', 'topwear', 'hair']

            loss0 = criterion(pred_action, label_D)
            loss1 = criterion(pred_skin, label_A[:, 0])
            loss2 = criterion(pred_pant, label_A[:, 1])
            loss3 = criterion(pred_top,  label_A[:, 2])
            loss4 = criterion(pred_hair, label_A[:, 3])

            loss = (loss0 + loss1 + loss2 + loss3 + loss4)/4

            loss.backward()
            optimizer.step()
            # print(pred.detach().cpu().numpy().shape)
            # print(np.argmax(pred.detach().cpu().numpy(), axis=1))
            # print(label.numpy())

            acc0 = (np.argmax(pred_action.detach().cpu().numpy(), axis=1)==label_D.cpu().numpy()).mean()
            acc1 = (np.argmax(pred_skin.detach().cpu().numpy(), axis=1) == label_A[:, 0].cpu().numpy()).mean()
            acc2 = (np.argmax(pred_pant.detach().cpu().numpy(), axis=1) == label_A[:, 1].cpu().numpy()).mean()
            acc3 = (np.argmax(pred_top.detach().cpu().numpy(), axis=1) ==  label_A[:, 2].cpu().numpy()).mean()
            acc4 = (np.argmax(pred_hair.detach().cpu().numpy(), axis=1) == label_A[:, 3].cpu().numpy()).mean()
            # print(acc)

            mean_loss += loss.item()
            mean_acc0 += acc0
            mean_acc1 += acc1
            mean_acc2 += acc2
            mean_acc3 += acc3
            mean_acc4 += acc4
            cnt += 1
            if idx % 20 == 0 and idx:
                print('Epoch: {} Iter: {}  loss: {:.4f}  Acc: {:.2f}% Acc: {:.2f}% Acc: {:.2f}% Acc: {:.2f}% Acc: {:.2f}% '.format(
                                                                             epoch, idx, mean_loss/cnt, mean_acc0/cnt*100,
                                                                             mean_acc1/cnt*100, mean_acc2/cnt*100,
                                                                             mean_acc3/cnt*100, mean_acc4/cnt*100))
                mean_loss = 0
                mean_acc0, mean_acc1, mean_acc2, mean_acc3, mean_acc4 = 0, 0, 0, 0, 0
                cnt = 0
        progress.finish()
        utils.clear_progressbar()

        net.eval()

        eval_loss = 0
        eval_acc0, eval_acc1, eval_acc2, eval_acc3, eval_acc4= 0, 0, 0, 0, 0
        for idx, batch in enumerate(train_loader):
            data = batch[0].float()
            x = utils.normalize_data(opt, dtype, data)
            label_A = batch[1].cuda()
            label_D = batch[2].cuda()
            x = torch.stack(x, dim=1).cuda()

            with torch.no_grad():
                pred_action, pred_skin, pred_pant, pred_top, pred_hair = net(x)
                loss0 = criterion(pred_action, label_D)
                loss1 = criterion(pred_skin, label_A[:, 0])
                loss2 = criterion(pred_pant, label_A[:, 1])
                loss3 = criterion(pred_top, label_A[:, 2])
                loss4 = criterion(pred_hair, label_A[:, 3])
                loss = (loss0 + loss1 + loss2 + loss3 + loss4) / 4


            acc0 = (np.argmax(pred_action.detach().cpu().numpy(), axis=1) == label_D.cpu().numpy()).mean()
            acc1 = (np.argmax(pred_skin.detach().cpu().numpy(), axis=1) == label_A[:, 0].cpu().numpy()).mean()
            acc2 = (np.argmax(pred_pant.detach().cpu().numpy(), axis=1) == label_A[:, 1].cpu().numpy()).mean()
            acc3 = (np.argmax(pred_top.detach().cpu().numpy(), axis=1) == label_A[:, 2].cpu().numpy()).mean()
            acc4 = (np.argmax(pred_hair.detach().cpu().numpy(), axis=1) == label_A[:, 3].cpu().numpy()).mean()
            eval_loss += loss.item()
            eval_acc0 += acc0
            eval_acc1 += acc1
            eval_acc2 += acc2
            eval_acc3 += acc3
            eval_acc4 += acc4

        print('Train loss: {:.4f} Acc: {:.2f}% Acc: {:.2f}% Acc: {:.2f}% Acc: {:.2f}% Acc: {:.2f}% '.format(
                                                       eval_loss / len(train_loader), eval_acc0 / len(train_loader)*100,
                                                       eval_acc1 / len(train_loader)*100, eval_acc2 / len(train_loader)*100,
                                                       eval_acc3 / len(train_loader)*100, eval_acc4 / len(train_loader)*100))

        eval_loss = 0
        eval_acc0, eval_acc1, eval_acc2, eval_acc3, eval_acc4= 0, 0, 0, 0, 0
        for idx, batch in enumerate(test_loader):
            data = batch[0].float()
            x = utils.normalize_data(opt, dtype, data)
            label_A = batch[1].cuda()
            label_D = batch[2].cuda()
            # label = batch[3].cuda()
            # mask = batch[4].cuda()
            x = torch.stack(x, dim=1).cuda()


            with torch.no_grad():
                pred_action, pred_skin, pred_pant, pred_top, pred_hair = net(x)
                loss0 = criterion(pred_action, label_D)
                loss1 = criterion(pred_skin, label_A[:, 0])
                loss2 = criterion(pred_pant, label_A[:, 1])
                loss3 = criterion(pred_top, label_A[:, 2])
                loss4 = criterion(pred_hair, label_A[:, 3])
                loss = (loss0 + loss1 + loss2 + loss3 + loss4) / 4

            acc0 = (np.argmax(pred_action.detach().cpu().numpy(), axis=1) == label_D.cpu().numpy()).mean()
            acc1 = (np.argmax(pred_skin.detach().cpu().numpy(), axis=1) == label_A[:, 0].cpu().numpy()).mean()
            acc2 = (np.argmax(pred_pant.detach().cpu().numpy(), axis=1) == label_A[:, 1].cpu().numpy()).mean()
            acc3 = (np.argmax(pred_top.detach().cpu().numpy(), axis=1) == label_A[:, 2].cpu().numpy()).mean()
            acc4 = (np.argmax(pred_hair.detach().cpu().numpy(), axis=1) == label_A[:, 3].cpu().numpy()).mean()
            eval_loss += loss.item()
            eval_acc0 += acc0
            eval_acc1 += acc1
            eval_acc2 += acc2
            eval_acc3 += acc3
            eval_acc4 += acc4
        print('Test loss: {:.4f} Acc: {:.2f}% Acc: {:.2f}% Acc: {:.2f}% Acc: {:.2f}% Acc: {:.2f}% '.format(
                                                       eval_loss / len(test_loader), eval_acc0 / len(test_loader)*100,
                                                       eval_acc1 / len(test_loader)*100, eval_acc2 / len(test_loader)*100,
                                                       eval_acc3 / len(test_loader)*100, eval_acc4 / len(test_loader)*100))

        torch.save({
            'epoch': epoch + 1,
            'state_dict': net.state_dict(),
        }, opt.log_dir + '/Epoch_{}_{:.2f}.tar'.format(epoch, eval_acc1 / len(test_loader)*100))


if __name__ == '__main__':
    main()
