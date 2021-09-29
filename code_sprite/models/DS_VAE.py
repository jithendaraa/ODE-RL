

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import OrderedDict



class Content_Disc(nn.Module):
    def __init__(self, in_features):
        super(Content_Disc, self).__init__()
        self.main = nn.Sequential(
                    nn.Linear(in_features, 256),
                    nn.ReLU(inplace=True),
                    nn.Linear(256, 256),
                    nn.ReLU(inplace=True),
                    nn.Linear(256, 128),
                    nn.ReLU(inplace=True),
                    nn.Linear(128, 1))

    def forward(self, x):
        return self.main(x)

# A block consisting of an affine layer, batch normalization (optional) followed by a nonlinearity (defaults to Leaky ReLU)
class LinearUnit(nn.Module):
    def __init__(self, in_features, out_features, batchnorm=True, nonlinearity=nn.LeakyReLU(0.2)):
        super(LinearUnit, self).__init__()
        if batchnorm is True:
            self.model = nn.Sequential(
                    nn.Linear(in_features, out_features),
                    nn.BatchNorm1d(out_features), nonlinearity)
        else:
            self.model = nn.Sequential(
                    nn.Linear(in_features, out_features), nonlinearity)

    def forward(self, x):
        return self.model(x)


class DisentangledVAE_New(nn.Module):
    def __init__(self, endecoder_model, opt):
        super(DisentangledVAE_New, self).__init__()
        self.f_dim = opt.f_dim  # content
        self.z_dim = opt.z_dim  # motion
        self.g_dim = opt.g_dim  # frame feature
        self.channels = opt.channels  # frame feature
        self.hidden_dim = opt.rnn_size
        self.f_rnn_layers = opt.f_rnn_layers
        self.frames = 15

        # Frame encoder and decoder
        self.encoder = endecoder_model.encoder(self.g_dim, self.channels)
        self.decoder = endecoder_model.decoder_woSkip(self.z_dim + self.f_dim, self.channels)

        # Prior of content is a uniform Gaussian and prior of the dynamics is an LSTM
        self.z_prior_lstm = nn.LSTMCell(self.z_dim, self.hidden_dim)
        self.z_prior_mean = nn.Linear(self.hidden_dim, self.z_dim)
        self.z_prior_logvar = nn.Linear(self.hidden_dim, self.z_dim)

        # POSTERIOR DISTRIBUTION NETWORKS
        # -------------------------------
        # content lstm
        self.f_lstm = nn.LSTM(self.g_dim, self.hidden_dim, self.f_rnn_layers,
                              bidirectional=True, batch_first=True)
        self.f_mean = LinearUnit(self.hidden_dim * 2, self.f_dim, False)
        self.f_logvar = LinearUnit(self.hidden_dim * 2, self.f_dim, False)

        # motion lstm
        self.z_lstm = nn.LSTM(self.g_dim + self.f_dim, self.hidden_dim, 1, bidirectional=True, batch_first=True)
        self.z_rnn = nn.RNN(self.hidden_dim * 2, self.hidden_dim, batch_first=True)
        # Each timestep is for each z so no reshaping and feature mixing
        self.z_mean = nn.Linear(self.hidden_dim, self.z_dim)
        self.z_logvar = nn.Linear(self.hidden_dim, self.z_dim)




    def forward(self, x):
        z_mean_prior, z_logvar_prior, _ = self.sample_z(x.size(0), random_sampling=self.training)
        conv_x = self.encoder_frame(x)
        f_mean, f_logvar, f = self.encode_f(conv_x)
        z_mean, z_logvar, z = self.encode_z(conv_x, f)

        f_expand = f.unsqueeze(1).expand(-1, self.frames, self.f_dim)
        zf = torch.cat((z, f_expand), dim=2)
        recon_x = self.decoder(zf)
        return f_mean, f_logvar, f, z_mean, z_logvar, z, z_mean_prior, z_logvar_prior, recon_x

    # Fixed the content and sample the motion vector.
    def forward_generating(self, x):
        z_mean_prior, z_logvar_prior, z_out = self.sample_z(x.size(0), random_sampling=self.training)
        conv_x = self.encoder_frame(x)
        f_mean, f_logvar, f = self.encode_f(conv_x)
        # z_mean, z_logvar, z = self.encode_z(conv_x, f)
        z_mean, z_logvar, z = z_mean_prior, z_logvar_prior, z_out
        f_repeat = f[0].repeat(100, 1)
        f_expand = f_repeat.unsqueeze(1).expand(-1, self.frames, self.f_dim)
        zf = torch.cat((z, f_expand), dim=2)
        recon_x = self.decoder(zf)
        return f_mean, f_logvar, f, z_mean, z_logvar, z, z_mean_prior, z_logvar_prior, recon_x



    def forward_exchange(self, x):
        z_mean_prior, z_logvar_prior, _ = self.sample_z(x.size(0), random_sampling=self.training)
        conv_x = self.encoder_frame(x)
        f_mean, f_logvar, f = self.encode_f(conv_x)
        z_mean, z_logvar, z = self.encode_z(conv_x, f)

        # perm = torch.LongTensor(np.random.permutation(f.shape[0]))
        # f_mix = f[perm]

        a = f[np.arange(0, f.shape[0], 2)]
        b = f[np.arange(1, f.shape[0], 2)]
        f_mix = torch.stack((b, a), dim=1).view((-1, f.shape[1]))
        # mix = torch.stack((b[0], a[0], b[1], a[1], b[2], a[2], b[3], a[3], b[4], a[4]), dim=0)
        # f_mix = torch.cat((mix, a[5:], b[5:]))

        f_expand = f_mix.unsqueeze(1).expand(-1, self.frames, self.f_dim)
        
        zf = torch.cat((z, f_expand), dim=2)
        recon_x = self.decoder(zf)
        return f_mean, f_logvar, f, z_mean, z_logvar, z, z_mean_prior, z_logvar_prior, recon_x

    def forward_fixed_motion(self, x):
        z_mean_prior, z_logvar_prior, _ = self.sample_z(x.size(0), random_sampling=self.training)
        conv_x = self.encoder_frame(x)
        f_mean, f_logvar, f = self.encode_f(conv_x)
        z_mean, z_logvar, z = self.encode_z(conv_x, f)

        # perm = torch.LongTensor(np.random.permutation(f.shape[0]))
        # f_mix = f[perm]
        z_repeat = z[0].repeat(100, 1, 1)
        # a = f[np.arange(0, f.shape[0], 2)]
        # b = f[np.arange(1, f.shape[0], 2)]
        # f_mix = torch.stack((b, a), dim=1).view((-1, f.shape[1]))
        # mix = torch.stack((b[0], a[0], b[1], a[1], b[2], a[2], b[3], a[3], b[4], a[4]), dim=0)
        # f_mix = torch.cat((mix, a[5:], b[5:]))

        f_expand = f.unsqueeze(1).expand(-1, self.frames, self.f_dim)

        zf = torch.cat((z_repeat, f_expand), dim=2)
        recon_x = self.decoder(zf)
        return f_mean, f_logvar, f, z_mean, z_logvar, z, z_mean_prior, z_logvar_prior, recon_x


    def forward_fixed_content(self, x):
        z_mean_prior, z_logvar_prior, _ = self.sample_z(x.size(0), random_sampling=self.training)
        conv_x = self.encoder_frame(x)
        f_mean, f_logvar, f = self.encode_f(conv_x)
        z_mean, z_logvar, z = self.encode_z(conv_x, f)

        # perm = torch.LongTensor(np.random.permutation(f.shape[0]))
        # f_mix = f[perm]
        # z_repeat = z[0].repeat(100, 1, 1)
        # a = f[np.arange(0, f.shape[0], 2)]
        # b = f[np.arange(1, f.shape[0], 2)]
        # f_mix = torch.stack((b, a), dim=1).view((-1, f.shape[1]))
        # mix = torch.stack((b[0], a[0], b[1], a[1], b[2], a[2], b[3], a[3], b[4], a[4]), dim=0)
        # f_mix = torch.cat((mix, a[5:], b[5:]))
        f_repeat = f[0].repeat(100, 1)
        f_expand = f_repeat.unsqueeze(1).expand(-1, self.frames, self.f_dim)

        zf = torch.cat((z, f_expand), dim=2)
        recon_x = self.decoder(zf)
        return f_mean, f_logvar, f, z_mean, z_logvar, z, z_mean_prior, z_logvar_prior, recon_x

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


    def encode_f(self, x):
        lstm_out, _ = self.f_lstm(x)
        # The features of the last timestep of the forward RNN is stored at the end of lstm_out in the first half, and the features
        # of the "first timestep" of the backward RNN is stored at the beginning of lstm_out in the second half
        # For a detailed explanation, check: https://gist.github.com/ceshine/bed2dadca48fe4fe4b4600ccce2fd6e1
        backward = lstm_out[:, 0, self.hidden_dim:2 * self.hidden_dim]
        frontal = lstm_out[:, self.frames - 1, 0:self.hidden_dim]
        lstm_out = torch.cat((frontal, backward), dim=1)
        mean = self.f_mean(lstm_out)
        logvar = self.f_logvar(lstm_out)
        return mean, logvar, self.reparameterize(mean, logvar, self.training)

    def encode_z(self, x, f):

        # The expansion is done to match the dimension of x and f, used for concatenating f to each x_t
        f_expand = f.unsqueeze(1).expand(-1, self.frames, self.f_dim)
        lstm_out, _ = self.z_lstm(torch.cat((x, f_expand), dim=2))
        features, _ = self.z_rnn(lstm_out)
        mean = self.z_mean(features)
        logvar = self.z_logvar(features)
        return mean, logvar, self.reparameterize(mean, logvar, self.training)

    def reparameterize(self, mean, logvar, random_sampling=True):
        # Reparametrization occurs only if random sampling is set to true, otherwise mean is returned
        if random_sampling is True:
            eps = torch.randn_like(logvar)
            std = torch.exp(0.5 *logvar)
            z = mean + eps *std
            return z
        else:
            return mean


    # If random sampling is true, reparametrization occurs else z_t is just set to the mean
    def sample_z(self, batch_size, random_sampling=True):
        z_out = None # This will ultimately store all z_s in the format [batch_size, frames, z_dim]
        z_means = None
        z_logvars = None

        # All states are initially set to 0, especially z_0 = 0
        z_t = torch.zeros(batch_size, self.z_dim).cuda()
        # z_mean_t = torch.zeros(batch_size, self.z_dim)
        # z_logvar_t = torch.zeros(batch_size, self.z_dim)
        h_t = torch.zeros(batch_size, self.hidden_dim).cuda()
        c_t = torch.zeros(batch_size, self.hidden_dim).cuda()

        for _ in range(self.frames):
            h_t, c_t = self.z_prior_lstm(z_t, (h_t, c_t))
            z_mean_t = self.z_prior_mean(h_t)
            z_logvar_t = self.z_prior_logvar(h_t)
            z_t = self.reparameterize(z_mean_t, z_logvar_t, random_sampling)
            if z_out is None:
                # If z_out is none it means z_t is z_1, hence store it in the format [batch_size, 1, z_dim]
                z_out = z_t.unsqueeze(1)
                z_means = z_mean_t.unsqueeze(1)
                z_logvars = z_logvar_t.unsqueeze(1)
            else:
                # If z_out is not none, z_t is not the initial z and hence append it to the previous z_ts collected in z_out
                z_out = torch.cat((z_out, z_t.unsqueeze(1)), dim=1)
                z_means = torch.cat((z_means, z_mean_t.unsqueeze(1)), dim=1)
                z_logvars = torch.cat((z_logvars, z_logvar_t.unsqueeze(1)), dim=1)

        return z_means, z_logvars, z_out

import copy
class DisentangledVAE_New_fixed(nn.Module):
    def __init__(self, endecoder_model, opt):
        super(DisentangledVAE_New_fixed, self).__init__()
        self.f_dim = opt.f_dim  # content
        self.z_dim = opt.z_dim  # motion
        self.g_dim = opt.g_dim  # frame feature
        self.channels = opt.channels  # frame feature
        self.hidden_dim = opt.rnn_size
        self.f_rnn_layers = opt.f_rnn_layers
        self.frames = 15

        # Frame encoder and decoder
        self.encoder = endecoder_model.encoder(self.g_dim, self.channels)
        self.decoder = endecoder_model.decoder_woSkip(self.z_dim + self.f_dim, self.channels)

        # Prior of content is a uniform Gaussian and prior of the dynamics is an LSTM
        self.z_prior_lstm_ly1 = nn.LSTMCell(self.z_dim, self.hidden_dim)
        self.z_prior_lstm_ly2 = nn.LSTMCell(self.hidden_dim, self.hidden_dim)

        self.z_prior_mean = nn.Linear(self.hidden_dim, self.z_dim)
        self.z_prior_logvar = nn.Linear(self.hidden_dim, self.z_dim)

        # POSTERIOR DISTRIBUTION NETWORKS
        # -------------------------------
        # content lstm
        self.f_lstm = nn.LSTM(self.g_dim, self.hidden_dim, self.f_rnn_layers,
                              bidirectional=True, batch_first=True)
        self.f_mean = LinearUnit(self.hidden_dim * 2, self.f_dim, False)
        self.f_logvar = LinearUnit(self.hidden_dim * 2, self.f_dim, False)

        # motion lstm
        self.z_lstm = nn.LSTM(self.g_dim + self.f_dim, self.hidden_dim, 1, bidirectional=True, batch_first=True)
        self.z_rnn = nn.RNN(self.hidden_dim * 2, self.hidden_dim, batch_first=True)
        # Each timestep is for each z so no reshaping and feature mixing
        self.z_mean = nn.Linear(self.hidden_dim, self.z_dim)
        self.z_logvar = nn.Linear(self.hidden_dim, self.z_dim)

        # nine block, label range (0,7)
        # self.z_motion_predictor = [nn.Sequential(OrderedDict([
        #     ('fc1', nn.Linear(self.z_dim, self.z_dim * 2)),
        #     ('relu', nn.LeakyReLU(0.2)),
        #     ('fc2', nn.Linear(self.z_dim * 2, 8)),
        # ])).cuda() for i in range(9)]

        self.z_motion_predictor0 = nn.Sequential(OrderedDict([
            ('fc1', nn.Linear(self.z_dim, self.z_dim * 2)),
            ('relu', nn.LeakyReLU(0.2)),
            ('fc2', nn.Linear(self.z_dim * 2, 8)),
            ]))
        self.z_motion_predictor1 = copy.deepcopy(self.z_motion_predictor0)
        self.z_motion_predictor2 = copy.deepcopy(self.z_motion_predictor0)
        self.z_motion_predictor3 = copy.deepcopy(self.z_motion_predictor0)
        self.z_motion_predictor4 = copy.deepcopy(self.z_motion_predictor0)
        self.z_motion_predictor5 = copy.deepcopy(self.z_motion_predictor0)
        self.z_motion_predictor6 = copy.deepcopy(self.z_motion_predictor0)
        self.z_motion_predictor7 = copy.deepcopy(self.z_motion_predictor0)
        self.z_motion_predictor8 = copy.deepcopy(self.z_motion_predictor0)
        # self.z_motion_predictor_list = [self.z_motion_predictor0, self.z_motion_predictor1, self.z_motion_predictor2,
        #                            self.z_motion_predictor3, self.z_motion_predictor4, self.z_motion_predictor5,
        #                            self.z_motion_predictor6, self.z_motion_predictor7, self.z_motion_predictor8]

    def forward(self, x):
        # z_mean_prior, z_logvar_prior, _ = self.sample_z(x.size(0), random_sampling=self.training)
        conv_x = self.encoder_frame(x)
        f_mean, f_logvar, f = self.encode_f(conv_x)
        z_mean_post, z_logvar_post, z_post = self.encode_z(conv_x, f)

        z_mean_prior, z_logvar_prior, z_prior = self.sample_z_prior_train(z_post, random_sampling=self.training)


        z_flatten = z_post.view(-1, z_post.shape[2])
        # pred = list()

        pred0 = self.z_motion_predictor0(z_flatten)
        pred1 = self.z_motion_predictor1(z_flatten)
        pred2 = self.z_motion_predictor2(z_flatten)
        pred3 = self.z_motion_predictor3(z_flatten)
        pred4 = self.z_motion_predictor4(z_flatten)
        pred5 = self.z_motion_predictor5(z_flatten)
        pred6 = self.z_motion_predictor6(z_flatten)
        pred7 = self.z_motion_predictor7(z_flatten)
        pred8 = self.z_motion_predictor8(z_flatten)

        # for i in range(9):
        #     pred.append(self.z_motion_predictor_list[i](z_flatten))
        # pred = torch.cat(pred, 0)
        pred = torch.cat([pred0, pred1, pred2, pred3, pred4, pred5,
                          pred6, pred7, pred8], 0)

        f_expand = f.unsqueeze(1).expand(-1, self.frames, self.f_dim)
        zf = torch.cat((z_post, f_expand), dim=2)
        recon_x = self.decoder(zf)
        return f_mean, f_logvar, f, z_mean_post, z_logvar_post, z_post, z_mean_prior, z_logvar_prior, recon_x, pred

    def sample_z_prior_train(self, z_post, random_sampling=True):
        z_out = None  # This will ultimately store all z_s in the format [batch_size, frames, z_dim]
        z_means = None
        z_logvars = None
        batch_size = z_post.shape[0]

        z_t = torch.zeros(batch_size, self.z_dim).cuda()
        h_t_ly1 = torch.zeros(batch_size, self.hidden_dim).cuda()
        c_t_ly1 = torch.zeros(batch_size, self.hidden_dim).cuda()
        h_t_ly2 = torch.zeros(batch_size, self.hidden_dim).cuda()
        c_t_ly2 = torch.zeros(batch_size, self.hidden_dim).cuda()

        for i in range(self.frames):
            # two layer LSTM and two one-layer FC
            h_t_ly1, c_t_ly1 = self.z_prior_lstm_ly1(z_t, (h_t_ly1, c_t_ly1))
            h_t_ly2, c_t_ly2 = self.z_prior_lstm_ly2(h_t_ly1, (h_t_ly2, c_t_ly2))

            z_mean_t = self.z_prior_mean(h_t_ly2)
            z_logvar_t = self.z_prior_logvar(h_t_ly2)
            z_prior = self.reparameterize(z_mean_t, z_logvar_t, random_sampling)
            if z_out is None:
                # If z_out is none it means z_t is z_1, hence store it in the format [batch_size, 1, z_dim]
                z_out = z_prior.unsqueeze(1)
                z_means = z_mean_t.unsqueeze(1)
                z_logvars = z_logvar_t.unsqueeze(1)
            else:
                # If z_out is not none, z_t is not the initial z and hence append it to the previous z_ts collected in z_out
                z_out = torch.cat((z_out, z_prior.unsqueeze(1)), dim=1)
                z_means = torch.cat((z_means, z_mean_t.unsqueeze(1)), dim=1)
                z_logvars = torch.cat((z_logvars, z_logvar_t.unsqueeze(1)), dim=1)
            z_t = z_post[:,i,:]
        return z_means, z_logvars, z_out

    # If random sampling is true, reparametrization occurs else z_t is just set to the mean
    def sample_z(self, batch_size, random_sampling=True):
        z_out = None  # This will ultimately store all z_s in the format [batch_size, frames, z_dim]
        z_means = None
        z_logvars = None

        # All states are initially set to 0, especially z_0 = 0
        z_t = torch.zeros(batch_size, self.z_dim).cuda()
        # z_mean_t = torch.zeros(batch_size, self.z_dim)
        # z_logvar_t = torch.zeros(batch_size, self.z_dim)
        h_t_ly1 = torch.zeros(batch_size, self.hidden_dim).cuda()
        c_t_ly1 = torch.zeros(batch_size, self.hidden_dim).cuda()
        h_t_ly2 = torch.zeros(batch_size, self.hidden_dim).cuda()
        c_t_ly2 = torch.zeros(batch_size, self.hidden_dim).cuda()
        for _ in range(self.frames):
            # h_t, c_t = self.z_prior_lstm(z_t, (h_t, c_t))
            # two layer LSTM and two one-layer FC
            h_t_ly1, c_t_ly1 = self.z_prior_lstm_ly1(z_t, (h_t_ly1, c_t_ly1))
            h_t_ly2, c_t_ly2 = self.z_prior_lstm_ly2(h_t_ly1, (h_t_ly2, c_t_ly2))

            z_mean_t = self.z_prior_mean(h_t_ly2)
            z_logvar_t = self.z_prior_logvar(h_t_ly2)
            z_t = self.reparameterize(z_mean_t, z_logvar_t, random_sampling)
            if z_out is None:
                # If z_out is none it means z_t is z_1, hence store it in the format [batch_size, 1, z_dim]
                z_out = z_t.unsqueeze(1)
                z_means = z_mean_t.unsqueeze(1)
                z_logvars = z_logvar_t.unsqueeze(1)
            else:
                # If z_out is not none, z_t is not the initial z and hence append it to the previous z_ts collected in z_out
                z_out = torch.cat((z_out, z_t.unsqueeze(1)), dim=1)
                z_means = torch.cat((z_means, z_mean_t.unsqueeze(1)), dim=1)
                z_logvars = torch.cat((z_logvars, z_logvar_t.unsqueeze(1)), dim=1)
        return z_means, z_logvars, z_out

    # fixed content and sample motion for classification disagreement scores
    def forward_fixed_content_for_classification(self, x):
        z_mean_prior, z_logvar_prior, _ = self.sample_z(x.size(0), random_sampling=True)
        f_mean, f_logvar, f_post, z_mean_post, z_logvar_post, z_post = self.encode_and_sample_post(x)

        f_expand = f_mean.unsqueeze(1).expand(-1, self.frames, self.f_dim)

        zf = torch.cat((z_mean_prior, f_expand), dim=2)
        recon_x_sample = self.decoder(zf)


        zf = torch.cat((z_mean_post, f_expand), dim=2)
        recon_x = self.decoder(zf)

        return recon_x_sample, recon_x

    def forward_single(self, x):

        z_mean_prior, z_logvar_prior, _ = self.sample_z(x.size(0), random_sampling=self.training)
        conv_x = self.encoder_frame(x)
        f_mean, f_logvar, f = self.encode_f(conv_x)
        z_mean, z_logvar, z = self.encode_z(conv_x, f)

        f_expand = f.unsqueeze(1).expand(-1, self.frames, self.f_dim)
        zf = torch.cat((z, f_expand), dim=2)
        recon_x = self.decoder(zf)
        return f_mean, f_logvar, f, z_mean, z_logvar, z, z_mean_prior, z_logvar_prior, recon_x

    def forward_exchange(self, x):
        z_mean_prior, z_logvar_prior, _ = self.sample_z(x.size(0), random_sampling=self.training)
        conv_x = self.encoder_frame(x)
        f_mean, f_logvar, f = self.encode_f(conv_x)
        z_mean, z_logvar, z = self.encode_z(conv_x, f)

        # perm = torch.LongTensor(np.random.permutation(f.shape[0]))
        # f_mix = f[perm]

        a = f[np.arange(0, f.shape[0], 2)]
        b = f[np.arange(1, f.shape[0], 2)]
        f_mix = torch.stack((b, a), dim=1).view((-1, f.shape[1]))
        # mix = torch.stack((b[0], a[0], b[1], a[1], b[2], a[2], b[3], a[3], b[4], a[4]), dim=0)
        # f_mix = torch.cat((mix, a[5:], b[5:]))

        f_expand = f_mix.unsqueeze(1).expand(-1, self.frames, self.f_dim)

        zf = torch.cat((z, f_expand), dim=2)
        recon_x = self.decoder(zf)
        return f_mean, f_logvar, f, z_mean, z_logvar, z, z_mean_prior, z_logvar_prior, recon_x

    def forward_shuffle(self, x):
        z_mean_prior, z_logvar_prior, _ = self.sample_z(x.size(0), random_sampling=self.training)
        conv_x = self.encoder_frame(x)
        f_mean, f_logvar, f = self.encode_f(conv_x)
        z_mean, z_logvar, z = self.encode_z(conv_x, f)

        perm = torch.LongTensor(np.random.permutation(f.shape[0]))
        f_mix = f[perm]

        # a = f[np.arange(0, f.shape[0], 2)]
        # b = f[np.arange(1, f.shape[0], 2)]
        # f_mix = torch.stack((b, a), dim=1).view((-1, f.shape[1]))
        # mix = torch.stack((b[0], a[0], b[1], a[1], b[2], a[2], b[3], a[3], b[4], a[4]), dim=0)
        # f_mix = torch.cat((mix, a[5:], b[5:]))

        f_expand = f_mix.unsqueeze(1).expand(-1, self.frames, self.f_dim)

        zf = torch.cat((z, f_expand), dim=2)
        recon_x = self.decoder(zf)
        return f_mean, f_logvar, f, z_mean, z_logvar, z, z_mean_prior, z_logvar_prior, recon_x

    def forward_fixed_motion(self, x):
        z_mean_prior, z_logvar_prior, _ = self.sample_z(x.size(0), random_sampling=self.training)
        conv_x = self.encoder_frame(x)
        f_mean, f_logvar, f = self.encode_f(conv_x)
        z_mean, z_logvar, z = self.encode_z(conv_x, f)

        z_repeat = z[0].repeat(100, 1, 1)
        f_expand = f.unsqueeze(1).expand(-1, self.frames, self.f_dim)

        zf = torch.cat((z_repeat, f_expand), dim=2)
        recon_x = self.decoder(zf)
        return f_mean, f_logvar, f, z_mean, z_logvar, z, z_mean_prior, z_logvar_prior, recon_x


    def forward_fixed_content(self, x):
        z_mean_prior, z_logvar_prior, _ = self.sample_z(x.size(0), random_sampling=self.training)
        conv_x = self.encoder_frame(x)
        f_mean, f_logvar, f = self.encode_f(conv_x)
        z_mean, z_logvar, z = self.encode_z(conv_x, f)

        f_repeat = f[0].repeat(100, 1)
        f_expand = f_repeat.unsqueeze(1).expand(-1, self.frames, self.f_dim)

        zf = torch.cat((z, f_expand), dim=2)
        recon_x = self.decoder(zf)
        return f_mean, f_logvar, f, z_mean, z_logvar, z, z_mean_prior, z_logvar_prior, recon_x

    # Fixed the content and sample the motion vector.
    def forward_generating(self, x):
        z_mean_prior, z_logvar_prior, z_out = self.sample_z(x.size(0), random_sampling=True)
        conv_x = self.encoder_frame(x)
        f_mean, f_logvar, f = self.encode_f(conv_x)
        # z_mean, z_logvar, z = self.encode_z(conv_x, f)
        z_mean, z_logvar, z = z_mean_prior, z_logvar_prior, z_out
        f_repeat = f[0].repeat(100, 1)
        f_expand = f_repeat.unsqueeze(1).expand(-1, self.frames, self.f_dim)
        zf = torch.cat((z, f_expand), dim=2)
        recon_x = self.decoder(zf)
        return f_mean, f_logvar, f, z_mean, z_logvar, z, z_mean_prior, z_logvar_prior, recon_x

    # Fixed the motion vector and sample content prior.
    def forward_generating2(self, x):
        # z_mean_prior, z_logvar_prior, z_out = self.sample_z(x.size(0), random_sampling=self.training)
        conv_x = self.encoder_frame(x)
        f_mean, f_logvar, f = self.encode_f(conv_x)
        z_mean, z_logvar, z = self.encode_z(conv_x, f)
        # z_mean, z_logvar, z = z_mean_prior, z_logvar_prior, z_out
        # f_repeat = f[0].repeat(100, 1)
        z_repeat = z[0].repeat(100, 1, 1)
        f_sampled = self.reparameterize(torch.zeros(f_mean.shape).cuda(), torch.zeros(f_logvar.shape).cuda(), random_sampling=True)

        f_expand = f_sampled.unsqueeze(1).expand(-1, self.frames, self.f_dim)
        zf = torch.cat((z_repeat, f_expand), dim=2)
        recon_x = self.decoder(zf)
        return recon_x


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

    def encode_f(self, x):
        lstm_out, _ = self.f_lstm(x)
        # The features of the last timestep of the forward RNN is stored at the end of lstm_out in the first half, and the features
        # of the "first timestep" of the backward RNN is stored at the beginning of lstm_out in the second half
        # For a detailed explanation, check: https://gist.github.com/ceshine/bed2dadca48fe4fe4b4600ccce2fd6e1
        backward = lstm_out[:, 0, self.hidden_dim:2 * self.hidden_dim]
        frontal = lstm_out[:, self.frames - 1, 0:self.hidden_dim]
        lstm_out = torch.cat((frontal, backward), dim=1)
        mean = self.f_mean(lstm_out)
        logvar = self.f_logvar(lstm_out)
        return mean, logvar, self.reparameterize(mean, logvar, self.training)

    def encode_z(self, x, f):

        # The expansion is done to match the dimension of x and f, used for concatenating f to each x_t
        f_expand = f.unsqueeze(1).expand(-1, self.frames, self.f_dim)
        lstm_out, _ = self.z_lstm(torch.cat((x, f_expand), dim=2))
        features, _ = self.z_rnn(lstm_out)
        mean = self.z_mean(features)
        logvar = self.z_logvar(features)
        return mean, logvar, self.reparameterize(mean, logvar, self.training)

    def reparameterize(self, mean, logvar, random_sampling=True):
        # Reparametrization occurs only if random sampling is set to true, otherwise mean is returned
        if random_sampling is True:
            eps = torch.randn_like(logvar)
            std = torch.exp(0.5 * logvar)
            z = mean + eps * std
            return z
        else:
            return mean

"""
# Remove the network for f, only one vae, latent features are separated 
"""
class DisentangledVAE_ICLR(nn.Module):
    def __init__(self, endecoder_model, opt):
        super(DisentangledVAE_ICLR, self).__init__()
        self.f_dim = opt.f_dim  # content
        self.z_dim = opt.z_dim  # motion
        self.g_dim = opt.g_dim  # frame feature
        self.channels = opt.channels  # frame feature
        self.hidden_dim = opt.rnn_size
        self.f_rnn_layers = opt.f_rnn_layers
        self.frames = 15

        # Frame encoder and decoder
        self.encoder = endecoder_model.encoder(self.g_dim, self.channels)
        self.decoder = endecoder_model.decoder_woSkip(self.z_dim + self.f_dim, self.channels)

        # Prior of content is a uniform Gaussian and prior of the dynamics is an LSTM
        self.z_prior_lstm_ly1 = nn.LSTMCell(self.z_dim, self.hidden_dim)
        self.z_prior_lstm_ly2 = nn.LSTMCell(self.hidden_dim, self.hidden_dim)

        self.z_prior_mean = nn.Linear(self.hidden_dim, self.z_dim)
        self.z_prior_logvar = nn.Linear(self.hidden_dim, self.z_dim)

        # POSTERIOR DISTRIBUTION NETWORKS
        # -------------------------------
        # content and motion features share one lstm
        self.z_lstm = nn.LSTM(self.g_dim, self.hidden_dim, 1, bidirectional=True, batch_first=True)
        self.z_rnn = nn.RNN(self.hidden_dim * 2, self.hidden_dim, batch_first=True)
        # Each timestep is for each z so no reshaping and feature mixing
        self.z_mean = nn.Linear(self.hidden_dim, self.f_dim + self.z_dim )
        self.z_logvar = nn.Linear(self.hidden_dim, self.f_dim + self.z_dim)

        # nine block, label range (0,7)
        # self.z_motion_predictor = [nn.Sequential(OrderedDict([
        #     ('fc1', nn.Linear(self.z_dim, self.z_dim * 2)),
        #     ('relu', nn.LeakyReLU(0.2)),
        #     ('fc2', nn.Linear(self.z_dim * 2, 8)),
        # ])).cuda() for i in range(9)]

        self.z_motion_predictor0 = nn.Sequential(OrderedDict([
            ('fc1', nn.Linear(self.z_dim, self.z_dim * 2)),
            ('relu', nn.LeakyReLU(0.2)),
            ('fc2', nn.Linear(self.z_dim * 2, 8)),
            ]))
        self.z_motion_predictor1 = copy.deepcopy(self.z_motion_predictor0)
        self.z_motion_predictor2 = copy.deepcopy(self.z_motion_predictor0)
        self.z_motion_predictor3 = copy.deepcopy(self.z_motion_predictor0)
        self.z_motion_predictor4 = copy.deepcopy(self.z_motion_predictor0)
        self.z_motion_predictor5 = copy.deepcopy(self.z_motion_predictor0)
        self.z_motion_predictor6 = copy.deepcopy(self.z_motion_predictor0)
        self.z_motion_predictor7 = copy.deepcopy(self.z_motion_predictor0)
        self.z_motion_predictor8 = copy.deepcopy(self.z_motion_predictor0)
        # self.z_motion_predictor_list = [self.z_motion_predictor0, self.z_motion_predictor1, self.z_motion_predictor2,
        #                            self.z_motion_predictor3, self.z_motion_predictor4, self.z_motion_predictor5,
        #                            self.z_motion_predictor6, self.z_motion_predictor7, self.z_motion_predictor8]

    def forward(self, x):
        # z_mean_prior, z_logvar_prior, _ = self.sample_z(x.size(0), random_sampling=self.training)
        conv_x = self.encoder_frame(x)

        lstm_out, _ = self.z_lstm(conv_x)
        features, _ = self.z_rnn(lstm_out)
        fz_mean_post = self.z_mean(features)
        fz_logvar_post = self.z_logvar(features)
        fz_post =  self.reparameterize(fz_mean_post, fz_logvar_post, self.training)

        f_mean   = fz_mean_post[:,:,:self.f_dim]
        f_logvar = fz_logvar_post[:,:,:self.f_dim]
        f_post = fz_post[:, :, :self.f_dim]

        z_mean_post   = fz_mean_post[:,:,self.f_dim:]
        z_logvar_post = fz_logvar_post[:,:,self.f_dim:]
        z_post = fz_post[:,:,self.f_dim:]

        z_mean_prior, z_logvar_prior, z_prior = self.sample_z_prior_train(z_post, random_sampling=self.training)


        z_flatten = z_post.view(-1, z_post.shape[2])
        # pred = list()

        pred0 = self.z_motion_predictor0(z_flatten)
        pred1 = self.z_motion_predictor1(z_flatten)
        pred2 = self.z_motion_predictor2(z_flatten)
        pred3 = self.z_motion_predictor3(z_flatten)
        pred4 = self.z_motion_predictor4(z_flatten)
        pred5 = self.z_motion_predictor5(z_flatten)
        pred6 = self.z_motion_predictor6(z_flatten)
        pred7 = self.z_motion_predictor7(z_flatten)
        pred8 = self.z_motion_predictor8(z_flatten)

        # for i in range(9):
        #     pred.append(self.z_motion_predictor_list[i](z_flatten))
        # pred = torch.cat(pred, 0)
        pred = torch.cat([pred0, pred1, pred2, pred3, pred4, pred5,
                          pred6, pred7, pred8], 0)

        # f_expand = f.unsqueeze(1).expand(-1, self.frames, self.f_dim)
        # zf = torch.cat((z_post, f_expand), dim=2)
        recon_x = self.decoder(fz_post)
        return f_mean, f_logvar, f_post, z_mean_post, z_logvar_post, z_post, z_mean_prior, z_logvar_prior, recon_x, pred

    def sample_z_prior_train(self, z_post, random_sampling=True):
        z_out = None  # This will ultimately store all z_s in the format [batch_size, frames, z_dim]
        z_means = None
        z_logvars = None
        batch_size = z_post.shape[0]

        z_t = torch.zeros(batch_size, self.z_dim).cuda()
        h_t_ly1 = torch.zeros(batch_size, self.hidden_dim).cuda()
        c_t_ly1 = torch.zeros(batch_size, self.hidden_dim).cuda()
        h_t_ly2 = torch.zeros(batch_size, self.hidden_dim).cuda()
        c_t_ly2 = torch.zeros(batch_size, self.hidden_dim).cuda()

        for i in range(self.frames):
            # two layer LSTM and two one-layer FC
            h_t_ly1, c_t_ly1 = self.z_prior_lstm_ly1(z_t, (h_t_ly1, c_t_ly1))
            h_t_ly2, c_t_ly2 = self.z_prior_lstm_ly2(h_t_ly1, (h_t_ly2, c_t_ly2))

            z_mean_t = self.z_prior_mean(h_t_ly2)
            z_logvar_t = self.z_prior_logvar(h_t_ly2)
            z_prior = self.reparameterize(z_mean_t, z_logvar_t, random_sampling)
            if z_out is None:
                # If z_out is none it means z_t is z_1, hence store it in the format [batch_size, 1, z_dim]
                z_out = z_prior.unsqueeze(1)
                z_means = z_mean_t.unsqueeze(1)
                z_logvars = z_logvar_t.unsqueeze(1)
            else:
                # If z_out is not none, z_t is not the initial z and hence append it to the previous z_ts collected in z_out
                z_out = torch.cat((z_out, z_prior.unsqueeze(1)), dim=1)
                z_means = torch.cat((z_means, z_mean_t.unsqueeze(1)), dim=1)
                z_logvars = torch.cat((z_logvars, z_logvar_t.unsqueeze(1)), dim=1)
            z_t = z_post[:,i,:]
        return z_means, z_logvars, z_out

    # If random sampling is true, reparametrization occurs else z_t is just set to the mean
    def sample_z(self, batch_size, random_sampling=True):
        z_out = None  # This will ultimately store all z_s in the format [batch_size, frames, z_dim]
        z_means = None
        z_logvars = None

        # All states are initially set to 0, especially z_0 = 0
        z_t = torch.zeros(batch_size, self.z_dim).cuda()
        # z_mean_t = torch.zeros(batch_size, self.z_dim)
        # z_logvar_t = torch.zeros(batch_size, self.z_dim)
        h_t_ly1 = torch.zeros(batch_size, self.hidden_dim).cuda()
        c_t_ly1 = torch.zeros(batch_size, self.hidden_dim).cuda()
        h_t_ly2 = torch.zeros(batch_size, self.hidden_dim).cuda()
        c_t_ly2 = torch.zeros(batch_size, self.hidden_dim).cuda()
        for _ in range(self.frames):
            # h_t, c_t = self.z_prior_lstm(z_t, (h_t, c_t))
            # two layer LSTM and two one-layer FC
            h_t_ly1, c_t_ly1 = self.z_prior_lstm_ly1(z_t, (h_t_ly1, c_t_ly1))
            h_t_ly2, c_t_ly2 = self.z_prior_lstm_ly2(h_t_ly1, (h_t_ly2, c_t_ly2))

            z_mean_t = self.z_prior_mean(h_t_ly2)
            z_logvar_t = self.z_prior_logvar(h_t_ly2)
            z_t = self.reparameterize(z_mean_t, z_logvar_t, random_sampling)
            if z_out is None:
                # If z_out is none it means z_t is z_1, hence store it in the format [batch_size, 1, z_dim]
                z_out = z_t.unsqueeze(1)
                z_means = z_mean_t.unsqueeze(1)
                z_logvars = z_logvar_t.unsqueeze(1)
            else:
                # If z_out is not none, z_t is not the initial z and hence append it to the previous z_ts collected in z_out
                z_out = torch.cat((z_out, z_t.unsqueeze(1)), dim=1)
                z_means = torch.cat((z_means, z_mean_t.unsqueeze(1)), dim=1)
                z_logvars = torch.cat((z_logvars, z_logvar_t.unsqueeze(1)), dim=1)
        return z_means, z_logvars, z_out


    def forward_single(self, x):

        z_mean_prior, z_logvar_prior, _ = self.sample_z(x.size(0), random_sampling=self.training)

        conv_x = self.encoder_frame(x)
        lstm_out, _ = self.z_lstm(conv_x)
        features, _ = self.z_rnn(lstm_out)
        fz_mean_post = self.z_mean(features)
        fz_logvar_post = self.z_logvar(features)
        fz_post =  self.reparameterize(fz_mean_post, fz_logvar_post, self.training)

        f_mean   = fz_mean_post[:,:,:self.f_dim]
        f_logvar = fz_logvar_post[:,:,:self.f_dim]
        f_post = fz_post[:, :, :self.f_dim]

        z_mean_post   = fz_mean_post[:,:,self.f_dim:]
        z_logvar_post = fz_logvar_post[:,:,self.f_dim:]
        z_post = fz_post[:,:,self.f_dim:]

        # f_expand = f.unsqueeze(1).expand(-1, self.frames, self.f_dim)
        # zf = torch.cat((z, f_expand), dim=2)
        recon_x = self.decoder(fz_post)
        return f_mean, f_logvar, f_post, z_mean_post, z_logvar_post, z_post, z_mean_prior, z_logvar_prior, recon_x

    def forward_exchange(self, x):
        z_mean_prior, z_logvar_prior, _ = self.sample_z(x.size(0), random_sampling=self.training)

        conv_x = self.encoder_frame(x)
        lstm_out, _ = self.z_lstm(conv_x)
        features, _ = self.z_rnn(lstm_out)
        fz_mean_post = self.z_mean(features)
        fz_logvar_post = self.z_logvar(features)
        fz_post =  self.reparameterize(fz_mean_post, fz_logvar_post, self.training)

        f_mean   = fz_mean_post[:,:,:self.f_dim]
        f_logvar = fz_logvar_post[:,:,:self.f_dim]
        f = fz_post[:, :, :self.f_dim]

        z_mean   = fz_mean_post[:,:,self.f_dim:]
        z_logvar = fz_logvar_post[:,:,self.f_dim:]
        z = fz_post[:,:,self.f_dim:]

        # perm = torch.LongTensor(np.random.permutation(f.shape[0]))
        # f_mix = f[perm]

        a = f[np.arange(0, f.shape[0], 2)]
        b = f[np.arange(1, f.shape[0], 2)]
        f_mix = torch.stack((b, a), dim=1).view((-1, f.shape[-2], f.shape[-1]))
        # mix = torch.stack((b[0], a[0], b[1], a[1], b[2], a[2], b[3], a[3], b[4], a[4]), dim=0)
        # f_mix = torch.cat((mix, a[5:], b[5:]))

        # f_expand = f_mix.unsqueeze(1).expand(-1, self.frames, self.f_dim)

        fz = torch.cat((f_mix, z), dim=2)
        recon_x = self.decoder(fz)
        return f_mean, f_logvar, f, z_mean, z_logvar, z, z_mean_prior, z_logvar_prior, recon_x

    def forward_shuffle(self, x):
        z_mean_prior, z_logvar_prior, _ = self.sample_z(x.size(0), random_sampling=self.training)
        conv_x = self.encoder_frame(x)
        lstm_out, _ = self.z_lstm(conv_x)
        features, _ = self.z_rnn(lstm_out)
        fz_mean_post = self.z_mean(features)
        fz_logvar_post = self.z_logvar(features)
        fz_post =  self.reparameterize(fz_mean_post, fz_logvar_post, self.training)

        f_mean   = fz_mean_post[:,:,:self.f_dim]
        f_logvar = fz_logvar_post[:,:,:self.f_dim]
        f = fz_post[:, :, :self.f_dim]

        z_mean   = fz_mean_post[:,:,self.f_dim:]
        z_logvar = fz_logvar_post[:,:,self.f_dim:]
        z = fz_post[:,:,self.f_dim:]

        perm = torch.LongTensor(np.random.permutation(f.shape[0]))
        f_mix = f[perm]

        # a = f[np.arange(0, f.shape[0], 2)]
        # b = f[np.arange(1, f.shape[0], 2)]
        # f_mix = torch.stack((b, a), dim=1).view((-1, f.shape[1]))
        # mix = torch.stack((b[0], a[0], b[1], a[1], b[2], a[2], b[3], a[3], b[4], a[4]), dim=0)
        # f_mix = torch.cat((mix, a[5:], b[5:]))

        f_expand = f_mix.unsqueeze(1).expand(-1, self.frames, self.f_dim)

        zf = torch.cat((z, f_expand), dim=2)
        recon_x = self.decoder(zf)
        return f_mean, f_logvar, f, z_mean, z_logvar, z, z_mean_prior, z_logvar_prior, recon_x

    def forward_fixed_motion(self, x):
        z_mean_prior, z_logvar_prior, _ = self.sample_z(x.size(0), random_sampling=self.training)
        conv_x = self.encoder_frame(x)
        lstm_out, _ = self.z_lstm(conv_x)
        features, _ = self.z_rnn(lstm_out)
        fz_mean_post = self.z_mean(features)
        fz_logvar_post = self.z_logvar(features)
        fz_post =  self.reparameterize(fz_mean_post, fz_logvar_post, self.training)

        f_mean   = fz_mean_post[:,:,:self.f_dim]
        f_logvar = fz_logvar_post[:,:,:self.f_dim]
        f = fz_post[:, :, :self.f_dim]

        z_mean   = fz_mean_post[:,:,self.f_dim:]
        z_logvar = fz_logvar_post[:,:,self.f_dim:]
        z = fz_post[:,:,self.f_dim:]

        z_repeat = z[0].repeat(100, 1, 1)
        # f_expand = f.unsqueeze(1).expand(-1, self.frames, self.f_dim)

        fz = torch.cat((f, z_repeat), dim=2)
        recon_x = self.decoder(fz)
        return f_mean, f_logvar, f, z_mean, z_logvar, z, z_mean_prior, z_logvar_prior, recon_x


    def forward_fixed_content(self, x):
        z_mean_prior, z_logvar_prior, _ = self.sample_z(x.size(0), random_sampling=self.training)
        conv_x = self.encoder_frame(x)
        lstm_out, _ = self.z_lstm(conv_x)
        features, _ = self.z_rnn(lstm_out)
        fz_mean_post = self.z_mean(features)
        fz_logvar_post = self.z_logvar(features)
        fz_post =  self.reparameterize(fz_mean_post, fz_logvar_post, self.training)

        f_mean   = fz_mean_post[:,:,:self.f_dim]
        f_logvar = fz_logvar_post[:,:,:self.f_dim]
        f = fz_post[:, :, :self.f_dim]

        z_mean   = fz_mean_post[:,:,self.f_dim:]
        z_logvar = fz_logvar_post[:,:,self.f_dim:]
        z = fz_post[:,:,self.f_dim:]

        f_repeat = f[0].repeat(100, 1, 1)
        # f_expand = f_repeat.unsqueeze(1).expand(-1, self.frames, self.f_dim)

        fz = torch.cat((f_repeat, z), dim=2)
        recon_x = self.decoder(fz)
        return f_mean, f_logvar, f, z_mean, z_logvar, z, z_mean_prior, z_logvar_prior, recon_x

    # Fixed the content and sample the motion vector.
    def forward_generating(self, x):
        z_mean_prior, z_logvar_prior, z_out = self.sample_z(x.size(0), random_sampling=True)
        conv_x = self.encoder_frame(x)
        lstm_out, _ = self.z_lstm(conv_x)
        features, _ = self.z_rnn(lstm_out)
        fz_mean_post = self.z_mean(features)
        fz_logvar_post = self.z_logvar(features)
        fz_post = self.reparameterize(fz_mean_post, fz_logvar_post, self.training)

        f_mean = fz_mean_post[:, :, :self.f_dim]
        f_logvar = fz_logvar_post[:, :, :self.f_dim]
        f = fz_post[:, :, :self.f_dim]

        # z_mean, z_logvar, z = self.encode_z(conv_x, f)
        z_mean, z_logvar, z = z_mean_prior, z_logvar_prior, z_out
        f_repeat = f[0].repeat(100, 1, 1)
        # f_expand = f_repeat.unsqueeze(1).expand(-1, self.frames, self.f_dim)

        fz = torch.cat((f_repeat, z), dim=2)
        recon_x = self.decoder(fz)
        return f_mean, f_logvar, f, z_mean, z_logvar, z, z_mean_prior, z_logvar_prior, recon_x

    # Fixed the motion vector and sample content prior.
    def forward_generating2(self, x):
        # z_mean_prior, z_logvar_prior, z_out = self.sample_z(x.size(0), random_sampling=self.training)
        conv_x = self.encoder_frame(x)
        lstm_out, _ = self.z_lstm(conv_x)
        features, _ = self.z_rnn(lstm_out)
        fz_mean_post = self.z_mean(features)
        fz_logvar_post = self.z_logvar(features)
        fz_post =  self.reparameterize(fz_mean_post, fz_logvar_post, self.training)

        f_mean   = fz_mean_post[:,:,:self.f_dim]
        f_logvar = fz_logvar_post[:,:,:self.f_dim]
        # f = fz_post[:, :, :self.f_dim]

        z = fz_post[:,:,self.f_dim:]

        # z_mean, z_logvar, z = z_mean_prior, z_logvar_prior, z_out
        # f_repeat = f[0].repeat(100, 1)
        z_repeat = z[0].repeat(100, 1, 1)
        f_sampled = self.reparameterize(torch.zeros([f_mean.shape[0], f_mean.shape[2]]).cuda(),
                                        torch.zeros(f_logvar.shape[0], f_logvar.shape[2]).cuda(), random_sampling=True)

        f_expand = f_sampled.unsqueeze(1).expand(-1, self.frames, self.f_dim)
        fz = torch.cat((f_expand, z_repeat), dim=2)
        recon_x = self.decoder(fz)
        return recon_x


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

    def encode_f(self, x):
        lstm_out, _ = self.f_lstm(x)
        # The features of the last timestep of the forward RNN is stored at the end of lstm_out in the first half, and the features
        # of the "first timestep" of the backward RNN is stored at the beginning of lstm_out in the second half
        # For a detailed explanation, check: https://gist.github.com/ceshine/bed2dadca48fe4fe4b4600ccce2fd6e1
        backward = lstm_out[:, 0, self.hidden_dim:2 * self.hidden_dim]
        frontal = lstm_out[:, self.frames - 1, 0:self.hidden_dim]
        lstm_out = torch.cat((frontal, backward), dim=1)
        mean = self.f_mean(lstm_out)
        logvar = self.f_logvar(lstm_out)
        return mean, logvar, self.reparameterize(mean, logvar, self.training)

    # def encode_z(self, x):
    #
    #     # The expansion is done to match the dimension of x and f, used for concatenating f to each x_t
    #     f_expand = f.unsqueeze(1).expand(-1, self.frames, self.f_dim)
    #

    def reparameterize(self, mean, logvar, random_sampling=True):
        # Reparametrization occurs only if random sampling is set to true, otherwise mean is returned
        if random_sampling is True:
            eps = torch.randn_like(logvar)
            std = torch.exp(0.5 * logvar)
            z = mean + eps * std
            return z
        else:
            return mean


"""
# Remove the network for f, only one vae, bidirectional lstm to f and z', z' to rnn for z. 
"""
class DisentangledVAE_ICLR_V1(nn.Module):
    def __init__(self, endecoder_model, opt):
        super(DisentangledVAE_ICLR_V1, self).__init__()
        self.f_dim = opt.f_dim  # content
        self.z_dim = opt.z_dim  # motion
        self.g_dim = opt.g_dim  # frame feature
        self.channels = opt.channels  # frame feature
        self.hidden_dim = opt.rnn_size
        self.f_rnn_layers = opt.f_rnn_layers
        self.frames = opt.frames

        # Frame encoder and decoder
        self.encoder = endecoder_model.encoder(self.g_dim, self.channels)
        self.decoder = endecoder_model.decoder_woSkip(self.z_dim + self.f_dim, self.channels)

        # Prior of content is a uniform Gaussian and prior of the dynamics is an LSTM
        self.z_prior_lstm_ly1 = nn.LSTMCell(self.z_dim, self.hidden_dim)
        self.z_prior_lstm_ly2 = nn.LSTMCell(self.hidden_dim, self.hidden_dim)

        self.z_prior_mean = nn.Linear(self.hidden_dim, self.z_dim)
        self.z_prior_logvar = nn.Linear(self.hidden_dim, self.z_dim)

        # POSTERIOR DISTRIBUTION NETWORKS
        # -------------------------------
        # content and motion features share one lstm
        self.z_lstm = nn.LSTM(self.g_dim, self.hidden_dim, 1, bidirectional=True, batch_first=True)
        self.f_mean = LinearUnit(self.hidden_dim * 2, self.f_dim, False)
        self.f_logvar = LinearUnit(self.hidden_dim * 2, self.f_dim, False)

        self.z_rnn = nn.RNN(self.hidden_dim * 2, self.hidden_dim, batch_first=True)
        # Each timestep is for each z so no reshaping and feature mixing
        self.z_mean = nn.Linear(self.hidden_dim, self.z_dim)
        self.z_logvar = nn.Linear(self.hidden_dim, self.z_dim)

        # 9 area
        self.z_motion_predictor_area = nn.Sequential(OrderedDict([
            ('fc1', nn.Linear(self.z_dim, self.z_dim * 2)),
            ('relu', nn.LeakyReLU(0.2)),
            ('fc2', nn.Linear(self.z_dim * 2, 9)),
        ]))
        # 8 direction bins
        self.z_motion_predictor0 = nn.Sequential(OrderedDict([
            ('fc1', nn.Linear(self.z_dim, self.z_dim * 2)),
            ('relu', nn.LeakyReLU(0.2)),
            ('fc2', nn.Linear(self.z_dim * 2, 8)),
            ]))
        self.z_motion_predictor1 = copy.deepcopy(self.z_motion_predictor0)
        self.z_motion_predictor2 = copy.deepcopy(self.z_motion_predictor0)
        self.z_motion_predictor3 = copy.deepcopy(self.z_motion_predictor0)
        self.z_motion_predictor4 = copy.deepcopy(self.z_motion_predictor0)
        self.z_motion_predictor5 = copy.deepcopy(self.z_motion_predictor0)
        self.z_motion_predictor6 = copy.deepcopy(self.z_motion_predictor0)
        self.z_motion_predictor7 = copy.deepcopy(self.z_motion_predictor0)
        self.z_motion_predictor8 = copy.deepcopy(self.z_motion_predictor0)

    def encode_and_sample_post(self, x):
        if isinstance(x, list):
            conv_x = self.encoder_frame(x[0])
        else:
            conv_x = self.encoder_frame(x)
        
        # pass the bidirectional lstm
        lstm_out, _ = self.z_lstm(conv_x)

        # for i in range(len(x)):
        #     val = x[i]
        #     print("x"+str(i), val.size())
        # print("conv_x", conv_x.size())
        # print("lstm_out", lstm_out.size(), self.frames, self.hidden_dim)
        
        # get f:
        backward = lstm_out[:, 0, self.hidden_dim:2 * self.hidden_dim]
        frontal = lstm_out[:, self.frames - 1, 0:self.hidden_dim]
        # print("backward", backward.size())
        # print("frontal", frontal.size())

        lstm_out_f = torch.cat((frontal, backward), dim=1)
        # print("lstm_out_f", lstm_out_f.size())

        f_mean = self.f_mean(lstm_out_f)
        # print("f_mean", f_mean.size())
        f_logvar = self.f_logvar(lstm_out_f)
        # print("f_logvar", f_logvar.size())
        f_post = self.reparameterize(f_mean, f_logvar, random_sampling=True)
        # print("f_post", f_post.size())

        # pass to one direction rnn
        features, _ = self.z_rnn(lstm_out)
        # print("features", features.size())
        z_mean = self.z_mean(features)
        # print("z_mean", z_mean.size())
        z_logvar = self.z_logvar(features)
        # print("z_logvar", z_logvar.size())
        z_post = self.reparameterize(z_mean, z_logvar, random_sampling=True)
        # print("z_post" ,z_post.size())
        # print()

        if isinstance(x, list):
            f_mean_list = [f_mean]
            for _x in x[1:]:
                conv_x = self.encoder_frame(_x)
                lstm_out, _ = self.z_lstm(conv_x)
                # get f:
                backward = lstm_out[:, 0, self.hidden_dim:2 * self.hidden_dim]
                frontal = lstm_out[:, self.frames - 1, 0:self.hidden_dim]
                lstm_out_f = torch.cat((frontal, backward), dim=1)
                f_mean = self.f_mean(lstm_out_f)
                f_mean_list.append(f_mean)
            f_mean = f_mean_list

        # f_mean is list if triple else not
        return f_mean, f_logvar, f_post, z_mean, z_logvar, z_post

    def forward(self, x):
        f_mean, f_logvar, f_post, z_mean_post, z_logvar_post, z_post = self.encode_and_sample_post(x)
        
        # for i in range(len(f_mean)):
        #     val = f_mean[i]
        #     print("f_mean"+str(i), val.size())
        # print("f_logvar", f_logvar.size() )
        # print("f_post", f_post.size() )
        # print("z_mean_post", z_mean_post.size() )
        # print("z_logvar_post", z_logvar_post.size() )
        # print("z_post", z_post.size() )
        # print()

        z_mean_prior, z_logvar_prior, z_prior = self.sample_z_prior_train(z_post, random_sampling=self.training)
        # print("Prior; z_mean, z_logvar, z_sample", z_mean_prior.size(), z_logvar_prior.size(), z_prior.size())
        # print()

        z_flatten = z_post.view(-1, z_post.shape[2])
        # print("z_flatten", z_flatten.size())

        pred_area = self.z_motion_predictor_area(z_flatten)
        # print("pred_area", pred_area.size())
        pred0 = self.z_motion_predictor0(z_flatten)
        # print("pred0", pred0.size())
        pred1 = self.z_motion_predictor1(z_flatten)
        # print("pred1", pred1.size())
        pred2 = self.z_motion_predictor2(z_flatten)
        # print("pred2", pred2.size())
        pred3 = self.z_motion_predictor3(z_flatten)
        # print("pred3", pred3.size())
        pred4 = self.z_motion_predictor4(z_flatten)
        # print("pred4", pred4.size())
        pred5 = self.z_motion_predictor5(z_flatten)
        # print("pred5", pred5.size())
        pred6 = self.z_motion_predictor6(z_flatten)
        # print("pred6", pred6.size())
        pred7 = self.z_motion_predictor7(z_flatten)
        # print("pred7", pred7.size())
        pred8 = self.z_motion_predictor8(z_flatten)
        # print("pred8", pred8.size())
        # print()

        pred = torch.cat([pred0, pred1, pred2, pred3, pred4, pred5,
                          pred6, pred7, pred8], 0)
        # print("pred", pred.size())

        f_expand = f_post.unsqueeze(1).expand(-1, self.frames, self.f_dim)
        # print("f_expand", f_expand.size())
        zf = torch.cat((z_post, f_expand), dim=2)
        # print("zf", zf.size())
        recon_x = self.decoder(zf)
        # print("recon", recon_x.size())

        return f_mean, f_logvar, f_post, z_mean_post, z_logvar_post, z_post, z_mean_prior, z_logvar_prior, z_prior, \
               recon_x, pred, pred_area




    def forward_single(self, x):

        z_mean_prior, z_logvar_prior, _ = self.sample_z(x.size(0), random_sampling=self.training)
        f_mean, f_logvar, f_post, z_mean_post, z_logvar_post, z_post = self.encode_and_sample_post(x)

        f_expand = f_post.unsqueeze(1).expand(-1, self.frames, self.f_dim)
        zf = torch.cat((z_post, f_expand), dim=2)
        recon_x = self.decoder(zf)
        return f_mean, f_logvar, f_post, z_mean_post, z_logvar_post, z_post, z_mean_prior, z_logvar_prior, recon_x

    def forward_exchange(self, x):
        z_mean_prior, z_logvar_prior, _ = self.sample_z(x.size(0), random_sampling=self.training)
        f_mean, f_logvar, f_post, z_mean_post, z_logvar_post, z_post = self.encode_and_sample_post(x)

        a = f_post[np.arange(0, f_post.shape[0], 2)]
        b = f_post[np.arange(1, f_post.shape[0], 2)]
        # f_mix = torch.stack((b, a), dim=1).view((-1, f_post.shape[-2], f_post.shape[-1]))
        f_mix = torch.stack((b, a), dim=1).view((-1, f_post.shape[1]))

        f_expand = f_mix.unsqueeze(1).expand(-1, self.frames, self.f_dim)
        zf = torch.cat((z_post, f_expand), dim=2)
        recon_x = self.decoder(zf)
        return f_mean, f_logvar, f_post, z_mean_post, z_logvar_post, z_post, z_mean_prior, z_logvar_prior, recon_x

    def forward_shuffle(self, x):
        z_mean_prior, z_logvar_prior, _ = self.sample_z(x.size(0), random_sampling=self.training)
        f_mean, f_logvar, f_post, z_mean_post, z_logvar_post, z_post = self.encode_and_sample_post(x)

        perm = torch.LongTensor(np.random.permutation(f_post.shape[0]))
        f_mix = f_post[perm]

        f_expand = f_mix.unsqueeze(1).expand(-1, self.frames, self.f_dim)
        zf = torch.cat((z_post, f_expand), dim=2)
        recon_x = self.decoder(zf)
        return f_mean, f_logvar, f_post, z_mean_post, z_logvar_post, z_post, z_mean_prior, z_logvar_prior, recon_x

    def forward_fixed_motion(self, x):
        z_mean_prior, z_logvar_prior, _ = self.sample_z(x.size(0), random_sampling=self.training)
        f_mean, f_logvar, f_post, z_mean_post, z_logvar_post, z_post = self.encode_and_sample_post(x)

        z_repeat = z_post[0].repeat(z_post.shape[0], 1, 1)
        f_expand = f_post.unsqueeze(1).expand(-1, self.frames, self.f_dim)
        zf = torch.cat((z_repeat, f_expand), dim=2)
        recon_x = self.decoder(zf)
        return f_mean, f_logvar, f_post, z_mean_post, z_logvar_post, z_post, z_mean_prior, z_logvar_prior, recon_x


    def forward_fixed_content(self, x):
        z_mean_prior, z_logvar_prior, _ = self.sample_z(x.size(0), random_sampling=self.training)
        f_mean, f_logvar, f_post, z_mean_post, z_logvar_post, z_post = self.encode_and_sample_post(x)

        f_repeat = f_post[0].repeat(f_post.shape[0], 1)
        f_expand = f_repeat.unsqueeze(1).expand(-1, self.frames, self.f_dim)

        zf = torch.cat((z_post, f_expand), dim=2)
        recon_x = self.decoder(zf)
        return f_mean, f_logvar, f_post, z_mean_post, z_logvar_post, z_post, z_mean_prior, z_logvar_prior, recon_x


    # fixed content and sample motion for classification disagreement scores
    def forward_fixed_content_for_classification(self, x):
        z_mean_prior, z_logvar_prior, _ = self.sample_z(x.size(0), random_sampling=True)
        f_mean, f_logvar, f_post, z_mean_post, z_logvar_post, z_post = self.encode_and_sample_post(x)

        f_expand = f_mean.unsqueeze(1).expand(-1, self.frames, self.f_dim)

        zf = torch.cat((z_mean_prior, f_expand), dim=2)
        recon_x_sample = self.decoder(zf)


        zf = torch.cat((z_mean_post, f_expand), dim=2)
        recon_x = self.decoder(zf)

        return recon_x_sample, recon_x

    # sample content and fixed motion for classification disagreement scores
    def forward_fixed_action_for_classification(self, x):
        z_mean_prior, z_logvar_prior, _ = self.sample_z(x.size(0), random_sampling=True)
        f_mean, f_logvar, f_post, z_mean_post, z_logvar_post, z_post = self.encode_and_sample_post(x)

        # f_expand = f_mean.unsqueeze(1).expand(-1, self.frames, self.f_dim)

        f_prior = self.reparameterize(torch.zeros(f_mean.shape).cuda(), torch.zeros(f_logvar.shape).cuda(),
                                        random_sampling=True)
        f_expand = f_prior.unsqueeze(1).expand(-1, self.frames, self.f_dim)
        zf = torch.cat((z_mean_post, f_expand), dim=2)
        recon_x_sample = self.decoder(zf)

        f_expand = f_mean.unsqueeze(1).expand(-1, self.frames, self.f_dim)
        zf = torch.cat((z_mean_post, f_expand), dim=2)
        recon_x = self.decoder(zf)

        return recon_x_sample, recon_x

    # forward_generating and forward_generating2 are generating ONLY where the prior is used
    # Fixed the content and sample the motion vector.
    def forward_generating(self, x):
        z_mean_prior, z_logvar_prior, z_out = self.sample_z(x.size(0), random_sampling=True)
        f_mean, f_logvar, f_post, z_mean_post, z_logvar_post, z_post = self.encode_and_sample_post(x)

        # z_mean, z_logvar, z = self.encode_z(conv_x, f)
        z_mean, z_logvar, z = z_mean_prior, z_logvar_prior, z_out
        f_repeat = f_post[0].repeat(f_post.shape[0], 1)
        f_expand = f_repeat.unsqueeze(1).expand(-1, self.frames, self.f_dim)

        # z prior + f post
        zf = torch.cat((z_out, f_expand), dim=2)
        recon_x = self.decoder(zf)
        return f_mean, f_logvar, f_post, z_mean_post, z_logvar_post, z_post, z_mean_prior, z_logvar_prior, recon_x

    # Fixed the motion vector and sample content prior.
    def forward_generating2(self, x):
        f_mean, f_logvar, f_post, z_mean_post, z_logvar_post, z_post = self.encode_and_sample_post(x)

        # z_mean, z_logvar, z = z_mean_prior, z_logvar_prior, z_out
        # f_repeat = f[0].repeat(100, 1)
        z_repeat = z_post[0].repeat(z_post.shape[0], 1, 1)
        # f_prior = self.reparameterize(torch.zeros([f_mean.shape[0], f_mean.shape[2]]).cuda(),
        #                                 torch.zeros(f_logvar.shape[0], f_logvar.shape[2]).cuda(), random_sampling=True)
        f_prior = self.reparameterize(torch.zeros(f_mean.shape).cuda(), torch.zeros(f_logvar.shape).cuda(),
                                        random_sampling=True)
        f_expand = f_prior.unsqueeze(1).expand(-1, self.frames, self.f_dim)
        zf = torch.cat((z_repeat, f_expand), dim=2)
        recon_x = self.decoder(zf)
        return recon_x


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

    def encode_f(self, x):
        lstm_out, _ = self.f_lstm(x)
        # The features of the last timestep of the forward RNN is stored at the end of lstm_out in the first half, and the features
        # of the "first timestep" of the backward RNN is stored at the beginning of lstm_out in the second half
        # For a detailed explanation, check: https://gist.github.com/ceshine/bed2dadca48fe4fe4b4600ccce2fd6e1
        backward = lstm_out[:, 0, self.hidden_dim:2 * self.hidden_dim]
        frontal = lstm_out[:, self.frames - 1, 0:self.hidden_dim]
        lstm_out = torch.cat((frontal, backward), dim=1)
        mean = self.f_mean(lstm_out)
        logvar = self.f_logvar(lstm_out)
        return mean, logvar, self.reparameterize(mean, logvar, self.training)

    # def encode_z(self, x):
    #
    #     # The expansion is done to match the dimension of x and f, used for concatenating f to each x_t
    #     f_expand = f.unsqueeze(1).expand(-1, self.frames, self.f_dim)
    #

    def reparameterize(self, mean, logvar, random_sampling=True):
        # Reparametrization occurs only if random sampling is set to true, otherwise mean is returned
        if random_sampling is True:
            eps = torch.randn_like(logvar)
            std = torch.exp(0.5 * logvar)
            z = mean + eps * std
            return z
        else:
            return mean

    # testing with arbitrary frames
    def sample_z_prior_test(self, n_sample, n_frame, random_sampling=True):
        z_out = None  # This will ultimately store all z_s in the format [batch_size, frames, z_dim]
        z_means = None
        z_logvars = None
        batch_size = n_sample

        z_t = torch.zeros(batch_size, self.z_dim).cuda()
        h_t_ly1 = torch.zeros(batch_size, self.hidden_dim).cuda()
        c_t_ly1 = torch.zeros(batch_size, self.hidden_dim).cuda()
        h_t_ly2 = torch.zeros(batch_size, self.hidden_dim).cuda()
        c_t_ly2 = torch.zeros(batch_size, self.hidden_dim).cuda()

        for i in range(n_frame):
            # two layer LSTM and two one-layer FC
            h_t_ly1, c_t_ly1 = self.z_prior_lstm_ly1(z_t, (h_t_ly1, c_t_ly1))
            h_t_ly2, c_t_ly2 = self.z_prior_lstm_ly2(h_t_ly1, (h_t_ly2, c_t_ly2))

            z_mean_t = self.z_prior_mean(h_t_ly2)
            z_logvar_t = self.z_prior_logvar(h_t_ly2)
            z_prior = self.reparameterize(z_mean_t, z_logvar_t, random_sampling)
            if z_out is None:
                # If z_out is none it means z_t is z_1, hence store it in the format [batch_size, 1, z_dim]
                z_out = z_prior.unsqueeze(1)
                z_means = z_mean_t.unsqueeze(1)
                z_logvars = z_logvar_t.unsqueeze(1)
            else:
                # If z_out is not none, z_t is not the initial z and hence append it to the previous z_ts collected in z_out
                z_out = torch.cat((z_out, z_prior.unsqueeze(1)), dim=1)
                z_means = torch.cat((z_means, z_mean_t.unsqueeze(1)), dim=1)
                z_logvars = torch.cat((z_logvars, z_logvar_t.unsqueeze(1)), dim=1)
                # z_t = z_post[:,i,:]
            z_t = z_prior
        return z_means, z_logvars, z_out

    def sample_z_prior_train(self, z_post, random_sampling=True):
        z_out = None  # This will ultimately store all z_s in the format [batch_size, frames, z_dim]
        z_means = None
        z_logvars = None
        batch_size = z_post.shape[0]

        z_t = torch.zeros(batch_size, self.z_dim).cuda()
        h_t_ly1 = torch.zeros(batch_size, self.hidden_dim).cuda()
        c_t_ly1 = torch.zeros(batch_size, self.hidden_dim).cuda()
        h_t_ly2 = torch.zeros(batch_size, self.hidden_dim).cuda()
        c_t_ly2 = torch.zeros(batch_size, self.hidden_dim).cuda()

        for i in range(self.frames):
            # two layer LSTM and two one-layer FC
            h_t_ly1, c_t_ly1 = self.z_prior_lstm_ly1(z_t, (h_t_ly1, c_t_ly1))
            h_t_ly2, c_t_ly2 = self.z_prior_lstm_ly2(h_t_ly1, (h_t_ly2, c_t_ly2))

            z_mean_t = self.z_prior_mean(h_t_ly2)
            z_logvar_t = self.z_prior_logvar(h_t_ly2)
            z_prior = self.reparameterize(z_mean_t, z_logvar_t, random_sampling)

            if z_out is None:
                # If z_out is none it means z_t is z_1, hence store it in the format [batch_size, 1, z_dim]
                z_out = z_prior.unsqueeze(1)
                z_means = z_mean_t.unsqueeze(1)
                z_logvars = z_logvar_t.unsqueeze(1)
            else:
                # If z_out is not none, z_t is not the initial z and hence append it to the previous z_ts collected in z_out
                z_out = torch.cat((z_out, z_prior.unsqueeze(1)), dim=1)
                z_means = torch.cat((z_means, z_mean_t.unsqueeze(1)), dim=1)
                z_logvars = torch.cat((z_logvars, z_logvar_t.unsqueeze(1)), dim=1)
            z_t = z_post[:,i,:]

        return z_means, z_logvars, z_out

    # If random sampling is true, reparametrization occurs else z_t is just set to the mean
    def sample_z(self, batch_size, random_sampling=True):
        z_out = None  # This will ultimately store all z_s in the format [batch_size, frames, z_dim]
        z_means = None
        z_logvars = None

        # All states are initially set to 0, especially z_0 = 0
        z_t = torch.zeros(batch_size, self.z_dim).cuda()
        # z_mean_t = torch.zeros(batch_size, self.z_dim)
        # z_logvar_t = torch.zeros(batch_size, self.z_dim)
        h_t_ly1 = torch.zeros(batch_size, self.hidden_dim).cuda()
        c_t_ly1 = torch.zeros(batch_size, self.hidden_dim).cuda()
        h_t_ly2 = torch.zeros(batch_size, self.hidden_dim).cuda()
        c_t_ly2 = torch.zeros(batch_size, self.hidden_dim).cuda()
        for _ in range(self.frames):
            # h_t, c_t = self.z_prior_lstm(z_t, (h_t, c_t))
            # two layer LSTM and two one-layer FC
            h_t_ly1, c_t_ly1 = self.z_prior_lstm_ly1(z_t, (h_t_ly1, c_t_ly1))
            h_t_ly2, c_t_ly2 = self.z_prior_lstm_ly2(h_t_ly1, (h_t_ly2, c_t_ly2))

            z_mean_t = self.z_prior_mean(h_t_ly2)
            z_logvar_t = self.z_prior_logvar(h_t_ly2)
            z_t = self.reparameterize(z_mean_t, z_logvar_t, random_sampling)
            if z_out is None:
                # If z_out is none it means z_t is z_1, hence store it in the format [batch_size, 1, z_dim]
                z_out = z_t.unsqueeze(1)
                z_means = z_mean_t.unsqueeze(1)
                z_logvars = z_logvar_t.unsqueeze(1)
            else:
                # If z_out is not none, z_t is not the initial z and hence append it to the previous z_ts collected in z_out
                z_out = torch.cat((z_out, z_t.unsqueeze(1)), dim=1)
                z_means = torch.cat((z_means, z_mean_t.unsqueeze(1)), dim=1)
                z_logvars = torch.cat((z_logvars, z_logvar_t.unsqueeze(1)), dim=1)
        return z_means, z_logvars, z_out


"""
# Supervised with ID or attributes 
# flexiable definition 
"""
class Supervised_Classifier(nn.Module):
    def __init__(self, nin, nhidden, nout):
        super(Supervised_Classifier, self).__init__()
        self.cls_FCs = nn.Sequential(
            nn.Linear(nin, nhidden),
            nn.ReLU(True),
            nn.Linear(nhidden, nout)
        )
    def forward(self, x):
        return self.cls_FCs(x)

"""
# Supervised with ID or attributes 
# flexiable definition 
"""
class Supervised_Classifier_Sprite(nn.Module):
    def __init__(self, nz, n_hidden_lstm, nout, a_nin, a_nhidden, a_nout, frames):
        super(Supervised_Classifier_Sprite, self).__init__()
        self.frames =  frames
        self.n_hidden_lstm = n_hidden_lstm
        self.bilstm = nn.LSTM(nz, n_hidden_lstm, 1, bidirectional=True, batch_first=True)

        self.cls_action = nn.Sequential(
            nn.Linear(n_hidden_lstm*2, n_hidden_lstm*2),
            nn.ReLU(True),
            nn.Linear(n_hidden_lstm*2, nout)
        )
        self.cls_skin = nn.Sequential(
            nn.Linear(a_nin, a_nhidden),
            nn.ReLU(True),
            nn.Linear(a_nhidden, a_nout)
        )
        self.cls_pant = nn.Sequential(
            nn.Linear(a_nin, a_nhidden),
            nn.ReLU(True),
            nn.Linear(a_nhidden, a_nout)
        )
        self.cls_top = nn.Sequential(
            nn.Linear(a_nin, a_nhidden),
            nn.ReLU(True),
            nn.Linear(a_nhidden, a_nout)
        )
        self.cls_hair = nn.Sequential(
            nn.Linear(a_nin, a_nhidden),
            nn.ReLU(True),
            nn.Linear(a_nhidden, a_nout)
        )

    def forward(self, z, f):
        # pass the bidirectional lstm
        lstm_out, _ = self.bilstm(z)
        backward = lstm_out[:, 0, self.n_hidden_lstm:2 * self.n_hidden_lstm]
        frontal = lstm_out[:, self.frames - 1, 0:self.n_hidden_lstm]
        lstm_out_z = torch.cat((frontal, backward), dim=1)

        return self.cls_action(lstm_out_z), self.cls_skin(f), self.cls_pant(f), self.cls_top(f), self.cls_hair(f)

"""
# Supervised with action  
# flexiable definition 
"""
class Supervised_Classifier_MUG(nn.Module):
    def __init__(self, nz, n_hidden_lstm, nout, frames):
        super(Supervised_Classifier_MUG, self).__init__()
        self.frames =  frames
        self.n_hidden_lstm = n_hidden_lstm
        self.bilstm = nn.LSTM(nz, n_hidden_lstm, 1, bidirectional=True, batch_first=True)

        self.cls_action = nn.Sequential(
            nn.Linear(n_hidden_lstm*2, n_hidden_lstm*2),
            nn.ReLU(True),
            nn.Linear(n_hidden_lstm*2, nout)
        )

    def forward(self, z, f):
        # pass the bidirectional lstm
        lstm_out, _ = self.bilstm(z)
        backward = lstm_out[:, 0, self.n_hidden_lstm:2 * self.n_hidden_lstm]
        frontal = lstm_out[:, self.frames - 1, 0:self.n_hidden_lstm]
        lstm_out_z = torch.cat((frontal, backward), dim=1)

        return self.cls_action(lstm_out_z)

"""
# Based on DisentangledVAE_ICLR_V2, anxiliry task: predict the distance of eyes and mouth  
#
"""
class DisentangledVAE_ICLR_MUG(nn.Module):
    def __init__(self, opt):
        super(DisentangledVAE_ICLR_MUG, self).__init__()
        self.f_dim = opt.f_dim  # content
        self.z_dim = opt.z_dim  # motion
        self.g_dim = opt.g_dim  # frame feature
        self.channels = opt.channels  # frame feature
        self.hidden_dim = opt.rnn_size
        self.f_rnn_layers = opt.f_rnn_layers
        self.frames = opt.frames

        # Frame encoder and decoder
        if opt.image_width == 64:
            from models.dcgan_64 import encoder
            if opt.decoder == 'Conv':
                from models.dcgan_64 import decoder_conv as decoder
            elif opt.decoder == 'ConvT':
                from models.dcgan_64 import decoder_convT as decoder
            else:
                raise ValueError('no implementation of decoder {}'.format(opt.decoder))
        elif opt.image_width == 128:
            from models.dcgan_128 import decoder_conv as decoder
            from models.dcgan_128 import encoder
        self.encoder = encoder(self.g_dim, self.channels)
        self.decoder = decoder(self.z_dim + self.f_dim, self.channels)

        # Prior of content is a uniform Gaussian and prior of the dynamics is an LSTM
        self.z_prior_lstm_ly1 = nn.LSTMCell(self.z_dim, self.hidden_dim)
        self.z_prior_lstm_ly2 = nn.LSTMCell(self.hidden_dim, self.hidden_dim)
        #
        self.z_prior_mean = nn.Linear(self.hidden_dim, self.z_dim)
        self.z_prior_logvar = nn.Linear(self.hidden_dim, self.z_dim)
        # self.z_prior_transformer = Encoder_Transformer(d_model=self.hidden_dim, N=1, heads=8, dropout=0.1)

        # POSTERIOR DISTRIBUTION NETWORKS
        # -------------------------------
        # content and motion features share one lstm
        self.z_lstm = nn.LSTM(self.g_dim, self.hidden_dim, 1, bidirectional=True, batch_first=True)
        self.f_mean = LinearUnit(self.hidden_dim * 2, self.f_dim, False)
        self.f_logvar = LinearUnit(self.hidden_dim * 2, self.f_dim, False)

        self.z_rnn = nn.RNN(self.hidden_dim * 2, self.hidden_dim, batch_first=True)
        # Each timestep is for each z so no reshaping and feature mixing
        self.z_mean = nn.Linear(self.hidden_dim, self.z_dim)
        self.z_logvar = nn.Linear(self.hidden_dim, self.z_dim)

        # 8 direction bins
        self.z_motion_predictor = nn.Sequential(OrderedDict([
            ('fc1', nn.Linear(self.z_dim, self.z_dim * 2)),
            ('relu', nn.LeakyReLU(0.2)),
            ('fc2', nn.Linear(self.z_dim * 2, 3)),
            ]))


    def encode_and_sample_post(self, x):
        if isinstance(x, list):
            conv_x = self.encoder_frame(x[0])
        else:
            conv_x = self.encoder_frame(x)
        # pass the bidirectional lstm
        lstm_out, _ = self.z_lstm(conv_x)
        # get f:
        backward = lstm_out[:, 0, self.hidden_dim:2 * self.hidden_dim]
        frontal = lstm_out[:, self.frames - 1, 0:self.hidden_dim]
        lstm_out_f = torch.cat((frontal, backward), dim=1)
        f_mean = self.f_mean(lstm_out_f)
        f_logvar = self.f_logvar(lstm_out_f)
        f_post = self.reparameterize(f_mean, f_logvar, random_sampling=True)

        # pass to one direction rnn
        features, _ = self.z_rnn(lstm_out)
        z_mean = self.z_mean(features)
        z_logvar = self.z_logvar(features)
        z_post = self.reparameterize(z_mean, z_logvar, random_sampling=True)

        if isinstance(x, list):
            f_mean_list = [f_mean]
            for _x in x[1:]:
                conv_x = self.encoder_frame(_x)
                lstm_out, _ = self.z_lstm(conv_x)
                # get f:
                backward = lstm_out[:, 0, self.hidden_dim:2 * self.hidden_dim]
                frontal = lstm_out[:, self.frames - 1, 0:self.hidden_dim]
                lstm_out_f = torch.cat((frontal, backward), dim=1)
                f_mean = self.f_mean(lstm_out_f)
                f_mean_list.append(f_mean)
            f_mean = f_mean_list
        # f_mean is list if triple else not
        return f_mean, f_logvar, f_post, z_mean, z_logvar, z_post

    # If random sampling is true, reparametrization occurs else z_t is just set to the mean
    def sample_z(self, batch_size, random_sampling=True):
        z_out = None  # This will ultimately store all z_s in the format [batch_size, frames, z_dim]
        z_means = None
        z_logvars = None

        # All states are initially set to 0, especially z_0 = 0
        z_t = torch.zeros(batch_size, self.z_dim).cuda()
        # z_mean_t = torch.zeros(batch_size, self.z_dim)
        # z_logvar_t = torch.zeros(batch_size, self.z_dim)
        h_t_ly1 = torch.zeros(batch_size, self.hidden_dim).cuda()
        c_t_ly1 = torch.zeros(batch_size, self.hidden_dim).cuda()
        h_t_ly2 = torch.zeros(batch_size, self.hidden_dim).cuda()
        c_t_ly2 = torch.zeros(batch_size, self.hidden_dim).cuda()
        for _ in range(self.frames):
            # h_t, c_t = self.z_prior_lstm(z_t, (h_t, c_t))
            # two layer LSTM and two one-layer FC
            h_t_ly1, c_t_ly1 = self.z_prior_lstm_ly1(z_t, (h_t_ly1, c_t_ly1))
            h_t_ly2, c_t_ly2 = self.z_prior_lstm_ly2(h_t_ly1, (h_t_ly2, c_t_ly2))

            z_mean_t = self.z_prior_mean(h_t_ly2)
            z_logvar_t = self.z_prior_logvar(h_t_ly2)
            z_t = self.reparameterize(z_mean_t, z_logvar_t, random_sampling)
            if z_out is None:
                # If z_out is none it means z_t is z_1, hence store it in the format [batch_size, 1, z_dim]
                z_out = z_t.unsqueeze(1)
                z_means = z_mean_t.unsqueeze(1)
                z_logvars = z_logvar_t.unsqueeze(1)
            else:
                # If z_out is not none, z_t is not the initial z and hence append it to the previous z_ts collected in z_out
                z_out = torch.cat((z_out, z_t.unsqueeze(1)), dim=1)
                z_means = torch.cat((z_means, z_mean_t.unsqueeze(1)), dim=1)
                z_logvars = torch.cat((z_logvars, z_logvar_t.unsqueeze(1)), dim=1)
        return z_means, z_logvars, z_out

    def sample_z_prior_train(self, z_post, random_sampling=True):
        z_out = None  # This will ultimately store all z_s in the format [batch_size, frames, z_dim]
        z_means = None
        z_logvars = None
        batch_size = z_post.shape[0]

        z_t = torch.zeros(batch_size, self.z_dim).cuda()
        h_t_ly1 = torch.zeros(batch_size, self.hidden_dim).cuda()
        c_t_ly1 = torch.zeros(batch_size, self.hidden_dim).cuda()
        h_t_ly2 = torch.zeros(batch_size, self.hidden_dim).cuda()
        c_t_ly2 = torch.zeros(batch_size, self.hidden_dim).cuda()

        for i in range(self.frames):
            # two layer LSTM and two one-layer FC
            h_t_ly1, c_t_ly1 = self.z_prior_lstm_ly1(z_t, (h_t_ly1, c_t_ly1))
            h_t_ly2, c_t_ly2 = self.z_prior_lstm_ly2(h_t_ly1, (h_t_ly2, c_t_ly2))

            z_mean_t = self.z_prior_mean(h_t_ly2)
            z_logvar_t = self.z_prior_logvar(h_t_ly2)
            z_prior = self.reparameterize(z_mean_t, z_logvar_t, random_sampling)
            if z_out is None:
                # If z_out is none it means z_t is z_1, hence store it in the format [batch_size, 1, z_dim]
                z_out = z_prior.unsqueeze(1)
                z_means = z_mean_t.unsqueeze(1)
                z_logvars = z_logvar_t.unsqueeze(1)
            else:
                # If z_out is not none, z_t is not the initial z and hence append it to the previous z_ts collected in z_out
                z_out = torch.cat((z_out, z_prior.unsqueeze(1)), dim=1)
                z_means = torch.cat((z_means, z_mean_t.unsqueeze(1)), dim=1)
                z_logvars = torch.cat((z_logvars, z_logvar_t.unsqueeze(1)), dim=1)
            z_t = z_post[:,i,:]
        return z_means, z_logvars, z_out


    # testing with arbitrary frames
    def sample_z_prior_test(self, n_sample, n_frame, random_sampling=True):
        z_out = None  # This will ultimately store all z_s in the format [batch_size, frames, z_dim]
        z_means = None
        z_logvars = None
        batch_size = n_sample

        z_t = torch.zeros(batch_size, self.z_dim).cuda()
        h_t_ly1 = torch.zeros(batch_size, self.hidden_dim).cuda()
        c_t_ly1 = torch.zeros(batch_size, self.hidden_dim).cuda()
        h_t_ly2 = torch.zeros(batch_size, self.hidden_dim).cuda()
        c_t_ly2 = torch.zeros(batch_size, self.hidden_dim).cuda()

        for i in range(n_frame):
            # two layer LSTM and two one-layer FC
            h_t_ly1, c_t_ly1 = self.z_prior_lstm_ly1(z_t, (h_t_ly1, c_t_ly1))
            h_t_ly2, c_t_ly2 = self.z_prior_lstm_ly2(h_t_ly1, (h_t_ly2, c_t_ly2))

            z_mean_t = self.z_prior_mean(h_t_ly2)
            z_logvar_t = self.z_prior_logvar(h_t_ly2)
            z_prior = self.reparameterize(z_mean_t, z_logvar_t, random_sampling)
            if z_out is None:
                # If z_out is none it means z_t is z_1, hence store it in the format [batch_size, 1, z_dim]
                z_out = z_prior.unsqueeze(1)
                z_means = z_mean_t.unsqueeze(1)
                z_logvars = z_logvar_t.unsqueeze(1)
            else:
                # If z_out is not none, z_t is not the initial z and hence append it to the previous z_ts collected in z_out
                z_out = torch.cat((z_out, z_prior.unsqueeze(1)), dim=1)
                z_means = torch.cat((z_means, z_mean_t.unsqueeze(1)), dim=1)
                z_logvars = torch.cat((z_logvars, z_logvar_t.unsqueeze(1)), dim=1)
                # z_t = z_post[:,i,:]
            z_t = z_prior
        return z_means, z_logvars, z_out


    def forward(self, x):
        f_mean, f_logvar, f_post, z_mean_post, z_logvar_post, z_post = self.encode_and_sample_post(x)
        z_mean_prior, z_logvar_prior, z_prior = self.sample_z_prior_train(z_post, random_sampling=self.training)

        z_flatten = z_post.view(-1, z_post.shape[2])

        pred = self.z_motion_predictor(z_flatten)

        f_expand = f_post.unsqueeze(1).expand(-1, self.frames, self.f_dim)
        zf = torch.cat((z_post, f_expand), dim=2)
        recon_x = self.decoder(zf)
        return f_mean, f_logvar, f_post, z_mean_post, z_logvar_post, z_post, z_mean_prior, z_logvar_prior, z_prior, \
               recon_x, pred


    # sample content and fixed motion for classification disagreement scores
    def forward_fixed_action_for_classification(self, x):
        z_mean_prior, z_logvar_prior, _ = self.sample_z(x.size(0), random_sampling=True)
        f_mean, f_logvar, f_post, z_mean_post, z_logvar_post, z_post = self.encode_and_sample_post(x)

        # f_expand = f_mean.unsqueeze(1).expand(-1, self.frames, self.f_dim)

        f_prior = self.reparameterize(torch.zeros(f_mean.shape).cuda(), torch.zeros(f_logvar.shape).cuda(),
                                        random_sampling=True)
        f_expand = f_prior.unsqueeze(1).expand(-1, self.frames, self.f_dim)
        zf = torch.cat((z_mean_post, f_expand), dim=2)
        recon_x_sample = self.decoder(zf)

        f_expand = f_mean.unsqueeze(1).expand(-1, self.frames, self.f_dim)
        zf = torch.cat((z_mean_post, f_expand), dim=2)
        recon_x = self.decoder(zf)

        return recon_x_sample, recon_x

    def forward_augment_action_for_classification(self, x, label):
        f_mean, f_logvar, f_post, z_mean_post, z_logvar_post, z_post = self.encode_and_sample_post(x)
        f_mean_list, z_mean_post_list, label_list = list(), list(), list()
        for i in range(6):
            flag = label == i
            if flag.sum() == 0:
                continue
            f_mean_tmp = f_mean[flag]
            z_mean_post_tmp = z_mean_post[flag]
            r = torch.randperm(f_mean_tmp.shape[0])
            f_mean_tmp =f_mean_tmp[r]
            z_mean_post_tmp = z_mean_post_tmp[r]
            f_mean_list.append(f_mean_tmp)
            z_mean_post_list.append(z_mean_post_tmp)
            label_list.append(label[flag])
        f_mean_list = torch.cat(f_mean_list, 0)
        z_mean_post_list = torch.cat(z_mean_post_list, 0)
        label_list = torch.cat(label_list, 0)


        f_expand = f_mean_list.unsqueeze(1).expand(-1, self.frames, self.f_dim)
        zf = torch.cat((z_mean_post_list, f_expand), dim=2)
        recon_x = self.decoder(zf)
        return recon_x, label_list

    # sample content and fixed motion for classification disagreement scores
    def forward_fixed_action_mix_other_Content_for_classification(self, x):
        z_mean_prior, z_logvar_prior, _ = self.sample_z(x.size(0), random_sampling=True)
        f_mean, f_logvar, f_post, z_mean_post, z_logvar_post, z_post = self.encode_and_sample_post(x)
        r = torch.randperm(f_mean.shape[0])
        # f_mean = f_mean[r]
        # f_expand = f_mean.unsqueeze(1).expand(-1, self.frames, self.f_dim)

        # f_prior = self.reparameterize(torch.zeros(f_mean.shape).cuda(), torch.zeros(f_logvar.shape).cuda(),
        #                                 random_sampling=True)
        f_expand = f_mean[r].unsqueeze(1).expand(-1, self.frames, self.f_dim)
        zf = torch.cat((z_mean_post, f_expand), dim=2)
        recon_x_sample = self.decoder(zf)

        f_expand = f_mean.unsqueeze(1).expand(-1, self.frames, self.f_dim)
        zf = torch.cat((z_mean_post, f_expand), dim=2)
        recon_x = self.decoder(zf)

        return recon_x_sample, recon_x

    def forward_single(self, x):
        f_mean, f_logvar, f_post, z_mean_post, z_logvar_post, z_post = self.encode_and_sample_post(x)
        z_mean_prior, z_logvar_prior, z_prior = self.sample_z_prior_train(z_post, random_sampling=True)

        f_expand = f_post.unsqueeze(1).expand(-1, self.frames, self.f_dim)
        zf = torch.cat((z_post, f_expand), dim=2)
        recon_x = self.decoder(zf)
        return f_mean, f_logvar, f_post, z_mean_post, z_logvar_post, z_post, z_mean_prior, z_logvar_prior, recon_x

    def forward_exchange(self, x):

        f_mean, f_logvar, f_post, z_mean_post, z_logvar_post, z_post = self.encode_and_sample_post(x)
        z_mean_prior, z_logvar_prior, z_prior = self.sample_z_prior_train(z_post, random_sampling=True)

        a = f_post[np.arange(0, f_post.shape[0], 2)]
        b = f_post[np.arange(1, f_post.shape[0], 2)]
        # f_mix = torch.stack((b, a), dim=1).view((-1, f_post.shape[-2], f_post.shape[-1]))
        f_mix = torch.stack((b, a), dim=1).view((-1, f_post.shape[1]))

        f_expand = f_mix.unsqueeze(1).expand(-1, self.frames, self.f_dim)
        zf = torch.cat((z_post, f_expand), dim=2)
        recon_x = self.decoder(zf)
        return f_mean, f_logvar, f_post, z_mean_post, z_logvar_post, z_post, z_mean_prior, z_logvar_prior, recon_x

    def forward_transient(self, x):
        f_mean, f_logvar, f_post, z_mean_post, z_logvar_post, z_post = self.encode_and_sample_post(x)

        z_expand = z_post.view(1, -1, self.z_dim)
        z_mean_prior, z_logvar_prior, z_prior = self.sample_z_prior_transient(z_expand, random_sampling=True)
        # f_repeat = f_post[0].repeat(f_post.shape[0], 1)
        f_expand = f_post[0].unsqueeze(0).unsqueeze(0).expand(-1, self.frames * f_post.shape[0], self.f_dim)
        #z_expand = z_post.view(1, -1, self.z_dim)
        z_expand = z_prior # z_post.view(1, -1, self.z_dim)

        zf = torch.cat((z_expand, f_expand), dim=2)
        recon_x = self.decoder(zf)
        return f_mean, f_logvar, f_post, z_mean_post, z_logvar_post, z_post, z_mean_prior, z_logvar_prior, recon_x

    def sample_z_prior_transient(self, z_post, random_sampling=True):
        z_out = None  # This will ultimately store all z_s in the format [batch_size, frames, z_dim]
        z_means = None
        z_logvars = None
        batch_size = z_post.shape[0]

        z_t = torch.zeros(batch_size, self.z_dim).cuda()
        h_t_ly1 = torch.zeros(batch_size, self.hidden_dim).cuda()
        c_t_ly1 = torch.zeros(batch_size, self.hidden_dim).cuda()
        h_t_ly2 = torch.zeros(batch_size, self.hidden_dim).cuda()
        c_t_ly2 = torch.zeros(batch_size, self.hidden_dim).cuda()

        for i in range(z_post.shape[1]):
            # two layer LSTM and two one-layer FC
            h_t_ly1, c_t_ly1 = self.z_prior_lstm_ly1(z_t, (h_t_ly1, c_t_ly1))
            h_t_ly2, c_t_ly2 = self.z_prior_lstm_ly2(h_t_ly1, (h_t_ly2, c_t_ly2))

            z_mean_t = self.z_prior_mean(h_t_ly2)
            z_logvar_t = self.z_prior_logvar(h_t_ly2)
            z_prior = self.reparameterize(z_mean_t, z_logvar_t, random_sampling)
            if z_out is None:
                # If z_out is none it means z_t is z_1, hence store it in the format [batch_size, 1, z_dim]
                z_out = z_prior.unsqueeze(1)
                z_means = z_mean_t.unsqueeze(1)
                z_logvars = z_logvar_t.unsqueeze(1)
            else:
                # If z_out is not none, z_t is not the initial z and hence append it to the previous z_ts collected in z_out
                z_out = torch.cat((z_out, z_prior.unsqueeze(1)), dim=1)
                z_means = torch.cat((z_means, z_mean_t.unsqueeze(1)), dim=1)
                z_logvars = torch.cat((z_logvars, z_logvar_t.unsqueeze(1)), dim=1)
            z_t = (z_post[:,i,:] + z_prior)/2
        return z_means, z_logvars, z_out

    def forward_shuffle(self, x):
        f_mean, f_logvar, f_post, z_mean_post, z_logvar_post, z_post = self.encode_and_sample_post(x)
        z_mean_prior, z_logvar_prior, z_prior = self.self.sample_z_prior_train(z_post, random_sampling=True)


        perm = torch.LongTensor(np.random.permutation(f_post.shape[0]))
        f_mix = f_post[perm]

        f_expand = f_mix.unsqueeze(1).expand(-1, self.frames, self.f_dim)
        zf = torch.cat((z_post, f_expand), dim=2)
        recon_x = self.decoder(zf)
        return f_mean, f_logvar, f_post, z_mean_post, z_logvar_post, z_post, z_mean_prior, z_logvar_prior, recon_x

    def forward_fixed_motion(self, x):

        f_mean, f_logvar, f_post, z_mean_post, z_logvar_post, z_post = self.encode_and_sample_post(x)
        z_mean_prior, z_logvar_prior, z_prior = self.sample_z_prior_train(z_post, random_sampling=True)
        z_repeat = z_post[0].repeat(z_post.shape[0], 1, 1)
        f_expand = f_post.unsqueeze(1).expand(-1, self.frames, self.f_dim)
        zf = torch.cat((z_repeat, f_expand), dim=2)
        recon_x = self.decoder(zf)
        return f_mean, f_logvar, f_post, z_mean_post, z_logvar_post, z_post, z_mean_prior, z_logvar_prior, recon_x


    def forward_fixed_content(self, x):

        f_mean, f_logvar, f_post, z_mean_post, z_logvar_post, z_post = self.encode_and_sample_post(x)
        z_mean_prior, z_logvar_prior, z_prior = self.sample_z_prior_train(z_post, random_sampling=True)
        f_repeat = f_post[0].repeat(f_post.shape[0], 1)
        f_expand = f_repeat.unsqueeze(1).expand(-1, self.frames, self.f_dim)

        zf = torch.cat((z_post, f_expand), dim=2)
        recon_x = self.decoder(zf)
        return f_mean, f_logvar, f_post, z_mean_post, z_logvar_post, z_post, z_mean_prior, z_logvar_prior, recon_x

    # forward_generating and forward_generating2 are generating ONLY where the prior is used
    # Fixed the content and sample the motion vector.
    def forward_generating(self, x):
        f_mean, f_logvar, f_post, z_mean_post, z_logvar_post, z_post = self.encode_and_sample_post(x)
        z_mean_prior, z_logvar_prior, z_out = self.sample_z_prior_train(z_post, random_sampling=True)

        # z_mean, z_logvar, z = self.encode_z(conv_x, f)
        z_mean, z_logvar, z = z_mean_prior, z_logvar_prior, z_out
        f_repeat = f_post[0].repeat(f_post.shape[0], 1)
        f_expand = f_repeat.unsqueeze(1).expand(-1, 20, self.f_dim) #self.frames

        # z prior + f post
        zf = torch.cat((z_out, f_expand), dim=2)
        recon_x = self.decoder(zf)
        return f_mean, f_logvar, f_post, z_mean_post, z_logvar_post, z_post, z_mean_prior, z_logvar_prior, recon_x

    def forward_generating_nframe(self, x, n_frame):
        f_mean, f_logvar, f_post, z_mean_post, z_logvar_post, z_post = self.encode_and_sample_post(x)
        z_mean_prior, z_logvar_prior, z_out = self.sample_z_prior_test(z_post, n_frame, random_sampling=True)

        # z_mean, z_logvar, z = self.encode_z(conv_x, f)
        z_mean, z_logvar, z = z_mean_prior, z_logvar_prior, z_out
        f_repeat = f_post[0].repeat(f_post.shape[0], 1)
        f_expand = f_repeat.unsqueeze(1).expand(-1, z_out.shape[1], self.f_dim) #self.frames

        # z prior + f postｚ
        zf = torch.cat((z_out, f_expand), dim=2)
        recon_x = self.decoder(zf)
        return f_mean, f_logvar, f_post, z_mean_post, z_logvar_post, z_post, z_mean_prior, z_logvar_prior, recon_x

    # Sample both content prior and motion prior.
    def forward_generating3_nframe(self, x, n_frame):
        f_mean, f_logvar, f_post, z_mean_post, z_logvar_post, z_post = self.encode_and_sample_post(x)
        z_mean_prior, z_logvar_prior, z_out = self.sample_z_prior_test(z_post, n_frame, random_sampling=True)

        # z_mean, z_logvar, z = self.encode_z(conv_x, f)
        z_mean, z_logvar, z = z_mean_prior, z_logvar_prior, z_out
        # f_repeat = f_post[0].repeat(f_post.shape[0], 1)

        f_prior = self.reparameterize(torch.zeros(f_mean.shape).cuda(), torch.zeros(f_logvar.shape).cuda(),
                                                                                            random_sampling=True)

        f_expand = f_prior.unsqueeze(1).expand(-1, z_out.shape[1], self.f_dim) #self.frames

        # z prior + f postｚ
        zf = torch.cat((z_out, f_expand), dim=2)
        recon_x = self.decoder(zf)
        return f_mean, f_logvar, f_post, z_mean_post, z_logvar_post, z_post, z_mean_prior, z_logvar_prior, recon_x


    # Fixed the motion vector and sample content prior.
    def forward_generating2(self, x):
        f_mean, f_logvar, f_post, z_mean_post, z_logvar_post, z_post = self.encode_and_sample_post(x)

        # z_mean, z_logvar, z = z_mean_prior, z_logvar_prior, z_out
        # f_repeat = f[0].repeat(100, 1)
        z_repeat = z_post[0].repeat(z_post.shape[0], 1, 1)
        # f_prior = self.reparameterize(torch.zeros([f_mean.shape[0], f_mean.shape[2]]).cuda(),
        #                                 torch.zeros(f_logvar.shape[0], f_logvar.shape[2]).cuda(), random_sampling=True)
        f_prior = self.reparameterize(torch.zeros(f_mean.shape).cuda(), torch.zeros(f_logvar.shape).cuda(),
                                                                                            random_sampling=True)
        f_expand = f_prior.unsqueeze(1).expand(-1, self.frames, self.f_dim)
        zf = torch.cat((z_repeat, f_expand), dim=2)
        recon_x = self.decoder(zf)
        return recon_x


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

    def encode_f(self, x):
        lstm_out, _ = self.f_lstm(x)
        # The features of the last timestep of the forward RNN is stored at the end of lstm_out in the first half, and the features
        # of the "first timestep" of the backward RNN is stored at the beginning of lstm_out in the second half
        # For a detailed explanation, check: https://gist.github.com/ceshine/bed2dadca48fe4fe4b4600ccce2fd6e1
        backward = lstm_out[:, 0, self.hidden_dim:2 * self.hidden_dim]
        frontal = lstm_out[:, self.frames - 1, 0:self.hidden_dim]
        lstm_out = torch.cat((frontal, backward), dim=1)
        mean = self.f_mean(lstm_out)
        logvar = self.f_logvar(lstm_out)
        return mean, logvar, self.reparameterize(mean, logvar, self.training)

    # def encode_z(self, x):
    #
    #     # The expansion is done to match the dimension of x and f, used for concatenating f to each x_t
    #     f_expand = f.unsqueeze(1).expand(-1, self.frames, self.f_dim)
    #

    def reparameterize(self, mean, logvar, random_sampling=True):
        # Reparametrization occurs only if random sampling is set to true, otherwise mean is returned
        if random_sampling is True:
            eps = torch.randn_like(logvar)
            std = torch.exp(0.5 * logvar)
            z = mean + eps * std
            return z
        else:
            return mean
    # def sample_z_transformer_prior_train(self, batch_size, random_sampling=True):
    #     # This will ultimately store all z_s in the format [batch_size, frames, z_dim]
    #     # batch_size = z_post.shape[0]
    #     input_zero = torch.zeros(batch_size, self.frames, self.hidden_dim).cuda()
    #     hidden_vector = self.z_prior_transformer(input_zero)
    #     z_mean = self.z_prior_mean(hidden_vector)
    #     z_logvar = self.z_prior_logvar(hidden_vector)
    #     z_prior = self.reparameterize(z_mean, z_logvar, random_sampling)
    #
    #     return z_mean, z_logvar, z_prior




# concatenate Z and F and feed into LSTM to get H
class DisentangledVAE_New_fixed_Z_notDependonF_PlusLSTM(nn.Module):
    def __init__(self, endecoder_model, opt):
        super(DisentangledVAE_New_fixed_Z_notDependonF_PlusLSTM, self).__init__()
        self.f_dim = opt.f_dim  # content
        self.z_dim = opt.z_dim  # motion
        self.g_dim = opt.g_dim  # frame feature
        self.channels = opt.channels  # frame feature
        self.hidden_dim = opt.rnn_size
        self.f_rnn_layers = opt.f_rnn_layers
        self.frames = 15

        # Frame encoder and decoder
        self.encoder = endecoder_model.encoder(self.g_dim, self.channels)
        self.decoder = endecoder_model.decoder_woSkip(self.z_dim + self.f_dim, self.channels)

        # Prior of content is a uniform Gaussian and prior of the dynamics is an LSTM
        self.z_prior_lstm = nn.LSTMCell(self.z_dim, self.hidden_dim)
        self.z_prior_mean = nn.Linear(self.hidden_dim, self.z_dim)
        self.z_prior_logvar = nn.Linear(self.hidden_dim, self.z_dim)

        # POSTERIOR DISTRIBUTION NETWORKS
        # -------------------------------
        # content lstm
        self.f_lstm = nn.LSTM(self.g_dim, self.hidden_dim, self.f_rnn_layers,
                              bidirectional=True, batch_first=True)
        self.f_mean = LinearUnit(self.hidden_dim * 2, self.f_dim, False)
        self.f_logvar = LinearUnit(self.hidden_dim * 2, self.f_dim, False)

        # motion lstm
        self.z_lstm = nn.LSTM(self.g_dim, self.hidden_dim, 1, bidirectional=True, batch_first=True)
        self.z_rnn = nn.RNN(self.hidden_dim * 2, self.hidden_dim, batch_first=True)
        # Each timestep is for each z so no reshaping and feature mixing
        self.z_mean = nn.Linear(self.hidden_dim, self.z_dim)
        self.z_logvar = nn.Linear(self.hidden_dim, self.z_dim)

        # merge content and motion
        self.zf_lstm = nn.LSTM(self.f_dim+self.z_dim, self.f_dim+self.z_dim, 1 , batch_first=True)
        # nine block, label range (0,7)
        # self.z_motion_predictor = [nn.Sequential(OrderedDict([
        #     ('fc1', nn.Linear(self.z_dim, self.z_dim * 2)),
        #     ('relu', nn.LeakyReLU(0.2)),
        #     ('fc2', nn.Linear(self.z_dim * 2, 8)),
        # ])).cuda() for i in range(9)]

        self.z_motion_predictor0 = nn.Sequential(OrderedDict([
            ('fc1', nn.Linear(self.z_dim, self.z_dim * 2)),
            ('relu', nn.LeakyReLU(0.2)),
            ('fc2', nn.Linear(self.z_dim * 2, 8)),
            ]))
        self.z_motion_predictor1 = copy.deepcopy(self.z_motion_predictor0)
        self.z_motion_predictor2 = copy.deepcopy(self.z_motion_predictor0)
        self.z_motion_predictor3 = copy.deepcopy(self.z_motion_predictor0)
        self.z_motion_predictor4 = copy.deepcopy(self.z_motion_predictor0)
        self.z_motion_predictor5 = copy.deepcopy(self.z_motion_predictor0)
        self.z_motion_predictor6 = copy.deepcopy(self.z_motion_predictor0)
        self.z_motion_predictor7 = copy.deepcopy(self.z_motion_predictor0)
        self.z_motion_predictor8 = copy.deepcopy(self.z_motion_predictor0)
        # self.z_motion_predictor_list = [self.z_motion_predictor0, self.z_motion_predictor1, self.z_motion_predictor2,
        #                            self.z_motion_predictor3, self.z_motion_predictor4, self.z_motion_predictor5,
        #                            self.z_motion_predictor6, self.z_motion_predictor7, self.z_motion_predictor8]

    def forward(self, x):
        z_mean_prior, z_logvar_prior, _ = self.sample_z(x.size(0), random_sampling=self.training)
        conv_x = self.encoder_frame(x)
        f_mean, f_logvar, f = self.encode_f(conv_x)
        z_mean, z_logvar, z = self.encode_z(conv_x)
        z_flatten = z.view(-1, z.shape[2])
        # pred = list()

        # pred0 = self.z_motion_predictor0(z_flatten)
        # pred1 = self.z_motion_predictor1(z_flatten)
        # pred2 = self.z_motion_predictor2(z_flatten)
        # pred3 = self.z_motion_predictor3(z_flatten)
        # pred4 = self.z_motion_predictor4(z_flatten)
        # pred5 = self.z_motion_predictor5(z_flatten)
        # pred6 = self.z_motion_predictor6(z_flatten)
        # pred7 = self.z_motion_predictor7(z_flatten)
        # pred8 = self.z_motion_predictor8(z_flatten)

        # for i in range(9):
        #     pred.append(self.z_motion_predictor_list[i](z_flatten))
        # pred = torch.cat(pred, 0)

        # pred = torch.cat([pred0, pred1, pred2, pred3, pred4, pred5,
        #                   pred6, pred7, pred8], 0)

        f_expand = f.unsqueeze(1).expand(-1, self.frames, self.f_dim)
        zf = torch.cat((z, f_expand), dim=2)
        zf_lstm_out, _ =  self.zf_lstm(zf)
        recon_x = self.decoder(zf_lstm_out.contiguous())
        return f_mean, f_logvar, f, z_mean, z_logvar, z, z_mean_prior, z_logvar_prior, recon_x, 1 #pred

    def forward_single(self, x):
        z_mean_prior, z_logvar_prior, _ = self.sample_z(x.size(0), random_sampling=self.training)
        conv_x = self.encoder_frame(x)
        f_mean, f_logvar, f = self.encode_f(conv_x)
        z_mean, z_logvar, z = self.encode_z(conv_x)

        f_expand = f.unsqueeze(1).expand(-1, self.frames, self.f_dim)
        zf = torch.cat((z, f_expand), dim=2)
        recon_x = self.decoder(zf)
        return f_mean, f_logvar, f, z_mean, z_logvar, z, z_mean_prior, z_logvar_prior, recon_x

    def forward_exchange(self, x):
        z_mean_prior, z_logvar_prior, _ = self.sample_z(x.size(0), random_sampling=self.training)
        conv_x = self.encoder_frame(x)
        f_mean, f_logvar, f = self.encode_f(conv_x)
        z_mean, z_logvar, z = self.encode_z(conv_x)

        # perm = torch.LongTensor(np.random.permutation(f.shape[0]))
        # f_mix = f[perm]

        a = f[np.arange(0, f.shape[0], 2)]
        b = f[np.arange(1, f.shape[0], 2)]
        f_mix = torch.stack((b, a), dim=1).view((-1, f.shape[1]))
        # mix = torch.stack((b[0], a[0], b[1], a[1], b[2], a[2], b[3], a[3], b[4], a[4]), dim=0)
        # f_mix = torch.cat((mix, a[5:], b[5:]))

        f_expand = f_mix.unsqueeze(1).expand(-1, self.frames, self.f_dim)

        zf = torch.cat((z, f_expand), dim=2)
        recon_x = self.decoder(zf)
        return f_mean, f_logvar, f, z_mean, z_logvar, z, z_mean_prior, z_logvar_prior, recon_x

    def forward_shuffle(self, x):
        z_mean_prior, z_logvar_prior, _ = self.sample_z(x.size(0), random_sampling=self.training)
        conv_x = self.encoder_frame(x)
        f_mean, f_logvar, f = self.encode_f(conv_x)
        z_mean, z_logvar, z = self.encode_z(conv_x)

        perm = torch.LongTensor(np.random.permutation(f.shape[0]))
        f_mix = f[perm]

        # a = f[np.arange(0, f.shape[0], 2)]
        # b = f[np.arange(1, f.shape[0], 2)]
        # f_mix = torch.stack((b, a), dim=1).view((-1, f.shape[1]))
        # mix = torch.stack((b[0], a[0], b[1], a[1], b[2], a[2], b[3], a[3], b[4], a[4]), dim=0)
        # f_mix = torch.cat((mix, a[5:], b[5:]))

        f_expand = f_mix.unsqueeze(1).expand(-1, self.frames, self.f_dim)

        zf = torch.cat((z, f_expand), dim=2)
        recon_x = self.decoder(zf)
        return f_mean, f_logvar, f, z_mean, z_logvar, z, z_mean_prior, z_logvar_prior, recon_x

    def forward_fixed_motion(self, x):
        z_mean_prior, z_logvar_prior, _ = self.sample_z(x.size(0), random_sampling=self.training)
        conv_x = self.encoder_frame(x)
        f_mean, f_logvar, f = self.encode_f(conv_x)
        z_mean, z_logvar, z = self.encode_z(conv_x)

        z_repeat = z[0].repeat(100, 1, 1)
        f_expand = f.unsqueeze(1).expand(-1, self.frames, self.f_dim)

        zf = torch.cat((z_repeat, f_expand), dim=2)
        recon_x = self.decoder(zf)
        return f_mean, f_logvar, f, z_mean, z_logvar, z, z_mean_prior, z_logvar_prior, recon_x


    def forward_fixed_content(self, x):
        z_mean_prior, z_logvar_prior, _ = self.sample_z(x.size(0), random_sampling=self.training)
        conv_x = self.encoder_frame(x)
        f_mean, f_logvar, f = self.encode_f(conv_x)
        z_mean, z_logvar, z = self.encode_z(conv_x)

        f_repeat = f[0].repeat(100, 1)
        f_expand = f_repeat.unsqueeze(1).expand(-1, self.frames, self.f_dim)

        zf = torch.cat((z, f_expand), dim=2)
        recon_x = self.decoder(zf)
        return f_mean, f_logvar, f, z_mean, z_logvar, z, z_mean_prior, z_logvar_prior, recon_x

    # Fixed the content and sample the motion vector.
    def forward_generating(self, x):
        z_mean_prior, z_logvar_prior, z_out = self.sample_z(x.size(0), random_sampling=True)
        conv_x = self.encoder_frame(x)
        f_mean, f_logvar, f = self.encode_f(conv_x)
        # z_mean, z_logvar, z = self.encode_z(conv_x, f)
        z_mean, z_logvar, z = z_mean_prior, z_logvar_prior, z_out
        f_repeat = f[0].repeat(100, 1)
        f_expand = f_repeat.unsqueeze(1).expand(-1, self.frames, self.f_dim)
        zf = torch.cat((z, f_expand), dim=2)
        recon_x = self.decoder(zf)
        return f_mean, f_logvar, f, z_mean, z_logvar, z, z_mean_prior, z_logvar_prior, recon_x

    # Fixed the motion vector and sample content prior.
    def forward_generating2(self, x):
        # z_mean_prior, z_logvar_prior, z_out = self.sample_z(x.size(0), random_sampling=self.training)
        conv_x = self.encoder_frame(x)
        f_mean, f_logvar, f = self.encode_f(conv_x)
        z_mean, z_logvar, z = self.encode_z(conv_x)
        # z_mean, z_logvar, z = z_mean_prior, z_logvar_prior, z_out
        # f_repeat = f[0].repeat(100, 1)
        z_repeat = z[0].repeat(100, 1, 1)
        f_sampled = self.reparameterize(torch.zeros(f_mean.shape).cuda(), torch.zeros(f_logvar.shape).cuda(), random_sampling=True)

        f_expand = f_sampled.unsqueeze(1).expand(-1, self.frames, self.f_dim)
        zf = torch.cat((z_repeat, f_expand), dim=2)
        recon_x = self.decoder(zf)
        return recon_x


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

    def encode_f(self, x):
        lstm_out, _ = self.f_lstm(x)
        # The features of the last timestep of the forward RNN is stored at the end of lstm_out in the first half, and the features
        # of the "first timestep" of the backward RNN is stored at the beginning of lstm_out in the second half
        # For a detailed explanation, check: https://gist.github.com/ceshine/bed2dadca48fe4fe4b4600ccce2fd6e1
        backward = lstm_out[:, 0, self.hidden_dim:2 * self.hidden_dim]
        frontal = lstm_out[:, self.frames - 1, 0:self.hidden_dim]
        lstm_out = torch.cat((frontal, backward), dim=1)
        mean = self.f_mean(lstm_out)
        logvar = self.f_logvar(lstm_out)
        return mean, logvar, self.reparameterize(mean, logvar, self.training)

    def encode_z(self, x):

        # The expansion is done to match the dimension of x and f, used for concatenating f to each x_t
        # f_expand = f.unsqueeze(1).expand(-1, self.frames, self.f_dim)
        # lstm_out, _ = self.z_lstm(torch.cat((x, f_expand), dim=2))
        lstm_out, _ = self.z_lstm(x)
        features, _ = self.z_rnn(lstm_out)
        mean = self.z_mean(features)
        logvar = self.z_logvar(features)
        return mean, logvar, self.reparameterize(mean, logvar, self.training)

    def reparameterize(self, mean, logvar, random_sampling=True):
        # Reparametrization occurs only if random sampling is set to true, otherwise mean is returned
        if random_sampling is True:
            eps = torch.randn_like(logvar)
            std = torch.exp(0.5 * logvar)
            z = mean + eps * std
            return z
        else:
            return mean

    # If random sampling is true, reparametrization occurs else z_t is just set to the mean
    def sample_z(self, batch_size, random_sampling=True):
        z_out = None  # This will ultimately store all z_s in the format [batch_size, frames, z_dim]
        z_means = None
        z_logvars = None

        # All states are initially set to 0, especially z_0 = 0
        z_t = torch.zeros(batch_size, self.z_dim).cuda()
        # z_mean_t = torch.zeros(batch_size, self.z_dim)
        # z_logvar_t = torch.zeros(batch_size, self.z_dim)
        h_t = torch.zeros(batch_size, self.hidden_dim).cuda()
        c_t = torch.zeros(batch_size, self.hidden_dim).cuda()

        for _ in range(self.frames):
            h_t, c_t = self.z_prior_lstm(z_t, (h_t, c_t))
            z_mean_t = self.z_prior_mean(h_t)
            z_logvar_t = self.z_prior_logvar(h_t)
            z_t = self.reparameterize(z_mean_t, z_logvar_t, random_sampling)
            if z_out is None:
                # If z_out is none it means z_t is z_1, hence store it in the format [batch_size, 1, z_dim]
                z_out = z_t.unsqueeze(1)
                z_means = z_mean_t.unsqueeze(1)
                z_logvars = z_logvar_t.unsqueeze(1)
            else:
                # If z_out is not none, z_t is not the initial z and hence append it to the previous z_ts collected in z_out
                z_out = torch.cat((z_out, z_t.unsqueeze(1)), dim=1)
                z_means = torch.cat((z_means, z_mean_t.unsqueeze(1)), dim=1)
                z_logvars = torch.cat((z_logvars, z_logvar_t.unsqueeze(1)), dim=1)

        return z_means, z_logvars, z_out



class DisentangledVAE_New_Triplet(nn.Module):
    def __init__(self, endecoder_model, opt):
        super(DisentangledVAE_New_Triplet, self).__init__()
        self.f_dim = opt.f_dim  # content
        self.z_dim = opt.z_dim  # motion
        self.g_dim = opt.g_dim  # frame feature
        self.channels = opt.channels  # frame feature
        self.hidden_dim = opt.rnn_size
        self.f_rnn_layers = opt.f_rnn_layers
        self.frames = 15

        # Frame encoder and decoder
        self.encoder = endecoder_model.encoder(self.g_dim, self.channels)
        self.decoder = endecoder_model.decoder_woSkip(self.z_dim + self.f_dim, self.channels)

        # Prior of content is a uniform Gaussian and prior of the dynamics is an LSTM
        self.z_prior_lstm = nn.LSTMCell(self.z_dim, self.hidden_dim)
        self.z_prior_mean = nn.Linear(self.hidden_dim, self.z_dim)
        self.z_prior_logvar = nn.Linear(self.hidden_dim, self.z_dim)

        # POSTERIOR DISTRIBUTION NETWORKS
        # -------------------------------
        # content lstm
        self.f_lstm = nn.LSTM(self.g_dim, self.hidden_dim, self.f_rnn_layers,
                              bidirectional=True, batch_first=True)
        self.f_mean = LinearUnit(self.hidden_dim * 2, self.f_dim, False)
        self.f_logvar = LinearUnit(self.hidden_dim * 2, self.f_dim, False)

        # motion lstm
        self.z_lstm = nn.LSTM(self.g_dim + self.f_dim, self.hidden_dim, 1, bidirectional=True, batch_first=True)
        self.z_rnn = nn.RNN(self.hidden_dim * 2, self.hidden_dim, batch_first=True)
        # Each timestep is for each z so no reshaping and feature mixing
        self.z_mean = nn.Linear(self.hidden_dim, self.z_dim)
        self.z_logvar = nn.Linear(self.hidden_dim, self.z_dim)

    def forward(self, x, x_pos, x_neg):
        z_mean_prior, z_logvar_prior, _ = self.sample_z(x.size(0), random_sampling=self.training)
        conv_x = self.encoder_frame(x)
        f_mean, f_logvar, f = self.encode_f(conv_x)
        # content triplet part
        conv_x_pos = self.encoder_frame(x_pos)
        _, _, f_pos = self.encode_f(conv_x_pos)
        conv_x_neg = self.encoder_frame(x_neg)
        _, _, f_neg = self.encode_f(conv_x_neg)

        z_mean, z_logvar, z = self.encode_z(conv_x, f)

        f_expand = f.unsqueeze(1).expand(-1, self.frames, self.f_dim)
        zf = torch.cat((z, f_expand), dim=2)
        recon_x = self.decoder(zf)
        return f_mean, f_logvar, f, z_mean, z_logvar, z, z_mean_prior, z_logvar_prior, recon_x, f_pos, f_neg

    def forward_single(self, x):
        z_mean_prior, z_logvar_prior, _ = self.sample_z(x.size(0), random_sampling=self.training)
        conv_x = self.encoder_frame(x)
        f_mean, f_logvar, f = self.encode_f(conv_x)

        z_mean, z_logvar, z = self.encode_z(conv_x, f)

        f_expand = f.unsqueeze(1).expand(-1, self.frames, self.f_dim)
        zf = torch.cat((z, f_expand), dim=2)
        recon_x = self.decoder(zf)
        return f_mean, f_logvar, f, z_mean, z_logvar, z, z_mean_prior, z_logvar_prior, recon_x

    def forward_exchange(self, x):
        z_mean_prior, z_logvar_prior, _ = self.sample_z(x.size(0), random_sampling=self.training)
        conv_x = self.encoder_frame(x)
        f_mean, f_logvar, f = self.encode_f(conv_x)
        z_mean, z_logvar, z = self.encode_z(conv_x, f)

        # perm = torch.LongTensor(np.random.permutation(f.shape[0]))
        # f_mix = f[perm]

        a = f[np.arange(0, f.shape[0], 2)]
        b = f[np.arange(1, f.shape[0], 2)]
        f_mix = torch.stack((b, a), dim=1).view((-1, f.shape[1]))
        # mix = torch.stack((b[0], a[0], b[1], a[1], b[2], a[2], b[3], a[3], b[4], a[4]), dim=0)
        # f_mix = torch.cat((mix, a[5:], b[5:]))

        f_expand = f_mix.unsqueeze(1).expand(-1, self.frames, self.f_dim)

        zf = torch.cat((z, f_expand), dim=2)
        recon_x = self.decoder(zf)
        return f_mean, f_logvar, f, z_mean, z_logvar, z, z_mean_prior, z_logvar_prior, recon_x

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

    def encode_f(self, x):
        lstm_out, _ = self.f_lstm(x)
        # The features of the last timestep of the forward RNN is stored at the end of lstm_out in the first half, and the features
        # of the "first timestep" of the backward RNN is stored at the beginning of lstm_out in the second half
        # For a detailed explanation, check: https://gist.github.com/ceshine/bed2dadca48fe4fe4b4600ccce2fd6e1
        backward = lstm_out[:, 0, self.hidden_dim:2 * self.hidden_dim]
        frontal = lstm_out[:, self.frames - 1, 0:self.hidden_dim]
        lstm_out = torch.cat((frontal, backward), dim=1)
        mean = self.f_mean(lstm_out)
        logvar = self.f_logvar(lstm_out)
        return mean, logvar, self.reparameterize(mean, logvar, self.training)

    def encode_z(self, x, f):

        # The expansion is done to match the dimension of x and f, used for concatenating f to each x_t
        f_expand = f.unsqueeze(1).expand(-1, self.frames, self.f_dim)
        lstm_out, _ = self.z_lstm(torch.cat((x, f_expand), dim=2))
        features, _ = self.z_rnn(lstm_out)
        mean = self.z_mean(features)
        logvar = self.z_logvar(features)
        return mean, logvar, self.reparameterize(mean, logvar, self.training)

    def reparameterize(self, mean, logvar, random_sampling=True):
        # Reparametrization occurs only if random sampling is set to true, otherwise mean is returned
        if random_sampling is True:
            eps = torch.randn_like(logvar)
            std = torch.exp(0.5 * logvar)
            z = mean + eps * std
            return z
        else:
            return mean

    # If random sampling is true, reparametrization occurs else z_t is just set to the mean
    def sample_z(self, batch_size, random_sampling=True):
        z_out = None  # This will ultimately store all z_s in the format [batch_size, frames, z_dim]
        z_means = None
        z_logvars = None

        # All states are initially set to 0, especially z_0 = 0
        z_t = torch.zeros(batch_size, self.z_dim).cuda()
        # z_mean_t = torch.zeros(batch_size, self.z_dim)
        # z_logvar_t = torch.zeros(batch_size, self.z_dim)
        h_t = torch.zeros(batch_size, self.hidden_dim).cuda()
        c_t = torch.zeros(batch_size, self.hidden_dim).cuda()

        for _ in range(self.frames):
            h_t, c_t = self.z_prior_lstm(z_t, (h_t, c_t))
            z_mean_t = self.z_prior_mean(h_t)
            z_logvar_t = self.z_prior_logvar(h_t)
            z_t = self.reparameterize(z_mean_t, z_logvar_t, random_sampling)
            if z_out is None:
                # If z_out is none it means z_t is z_1, hence store it in the format [batch_size, 1, z_dim]
                z_out = z_t.unsqueeze(1)
                z_means = z_mean_t.unsqueeze(1)
                z_logvars = z_logvar_t.unsqueeze(1)
            else:
                # If z_out is not none, z_t is not the initial z and hence append it to the previous z_ts collected in z_out
                z_out = torch.cat((z_out, z_t.unsqueeze(1)), dim=1)
                z_means = torch.cat((z_means, z_mean_t.unsqueeze(1)), dim=1)
                z_logvars = torch.cat((z_logvars, z_logvar_t.unsqueeze(1)), dim=1)

        return z_means, z_logvars, z_out




"""
# Based on DisentangledVAE_ICLR_V2, anxiliry task: predict the distance of eyes and mouth  
#
"""
class DisentangledVAE_ICLR_TIMIT(nn.Module):
    def __init__(self, opt):
        super(DisentangledVAE_ICLR_TIMIT, self).__init__()
        self.x_dim = opt.x_dim  # data dimension
        self.g_dim = opt.g_dim  # frame feature
        self.f_dim = opt.f_dim  # content
        self.z_dim = opt.z_dim  # motion

        self.hidden_dim = opt.rnn_size
        self.f_rnn_layers = opt.f_rnn_layers
        self.frames = opt.frames


        self.encoder = nn.Linear(self.x_dim, self.g_dim)

        self.decoder = nn.Sequential(nn.Linear(self.z_dim + self.f_dim, self.hidden_dim * 2),
                                     nn.LeakyReLU(0.2),
                                     nn.Linear(self.hidden_dim * 2 , self.hidden_dim),
                                     nn.LeakyReLU(0.2),
                                     nn.Linear(self.hidden_dim, self.x_dim)
                                     )

        # Prior of content is a uniform Gaussian and prior of the dynamics is an LSTM
        self.z_prior_lstm_ly1 = nn.LSTMCell(self.z_dim, self.hidden_dim )
        self.z_prior_lstm_ly2 = nn.LSTMCell(self.hidden_dim, self.hidden_dim)
        #
        self.z_prior_mean = nn.Linear(self.hidden_dim, self.z_dim)
        self.z_prior_logvar = nn.Linear(self.hidden_dim, self.z_dim)
        # self.z_prior_transformer = Encoder_Transformer(d_model=self.hidden_dim, N=1, heads=8, dropout=0.1)

        # POSTERIOR DISTRIBUTION NETWORKS
        # -------------------------------
        # content and motion features share one lstm
        self.z_lstm = nn.LSTM(self.g_dim, self.hidden_dim, 1, bidirectional=True, batch_first=True)
        self.f_mean = nn.Linear(self.hidden_dim * 2, self.f_dim) #LinearUnit(self.hidden_dim * 2, self.f_dim, False)
        self.f_logvar = nn.Linear(self.hidden_dim * 2, self.f_dim) #LinearUnit(self.hidden_dim * 2, self.f_dim, False)

        self.z_rnn = nn.LSTM(self.hidden_dim * 2, self.hidden_dim, 1, batch_first=True)
        # Each timestep is for each z so no reshaping and feature mixing
        self.z_mean = nn.Linear(self.hidden_dim, self.z_dim)
        self.z_logvar = nn.Linear(self.hidden_dim, self.z_dim)

        # binary silient or not
        self.z_motion_predictor = nn.Sequential(OrderedDict([
            ('fc1', nn.Linear(self.z_dim, self.z_dim * 2)),
            ('relu', nn.LeakyReLU(0.2)),
            ('fc2', nn.Linear(self.z_dim * 2, 1)),
            ]))


    def encode_and_sample_post(self, x):
        if isinstance(x, list):
            conv_x = self.encoder_frame(x[0])
        else:
            conv_x = self.encoder_frame(x)
        # pass the bidirectional lstm
        lstm_out, _ = self.z_lstm(conv_x)
        # get f:
        backward = lstm_out[:, 0, self.hidden_dim:2 * self.hidden_dim]
        frontal = lstm_out[:, self.frames - 1, 0:self.hidden_dim]
        lstm_out_f = torch.cat((frontal, backward), dim=1)
        f_mean = self.f_mean(lstm_out_f)
        f_logvar = self.f_logvar(lstm_out_f)
        f_post = self.reparameterize(f_mean, f_logvar, random_sampling=True)

        # pass to one direction rnn
        features, _ = self.z_rnn(lstm_out)
        z_mean = self.z_mean(features)
        z_logvar = self.z_logvar(features)
        z_post = self.reparameterize(z_mean, z_logvar, random_sampling=True)

        if isinstance(x, list):
            f_mean_list = [f_mean]
            for _x in x[1:]:
                conv_x = self.encoder_frame(_x)
                lstm_out, _ = self.z_lstm(conv_x)
                # get f:
                backward = lstm_out[:, 0, self.hidden_dim:2 * self.hidden_dim]
                frontal = lstm_out[:, self.frames - 1, 0:self.hidden_dim]
                lstm_out_f = torch.cat((frontal, backward), dim=1)
                f_mean = self.f_mean(lstm_out_f)
                f_mean_list.append(f_mean)
            f_mean = f_mean_list
        # f_mean is list if triple else not
        return f_mean, f_logvar, f_post, z_mean, z_logvar, z_post

    def sample_z_prior_train(self, z_post, random_sampling=True):
        z_out = None  # This will ultimately store all z_s in the format [batch_size, frames, z_dim]
        z_means = None
        z_logvars = None
        batch_size = z_post.shape[0]

        z_t = torch.zeros(batch_size, self.z_dim).cuda()
        h_t_ly1 = torch.zeros(batch_size, self.hidden_dim).cuda()
        c_t_ly1 = torch.zeros(batch_size, self.hidden_dim).cuda()
        h_t_ly2 = torch.zeros(batch_size, self.hidden_dim).cuda()
        c_t_ly2 = torch.zeros(batch_size, self.hidden_dim).cuda()

        for i in range(self.frames):
            # two layer LSTM and two one-layer FC
            h_t_ly1, c_t_ly1 = self.z_prior_lstm_ly1(z_t, (h_t_ly1, c_t_ly1))
            h_t_ly2, c_t_ly2 = self.z_prior_lstm_ly2(h_t_ly1, (h_t_ly2, c_t_ly2))

            z_mean_t = self.z_prior_mean(h_t_ly2)
            z_logvar_t = self.z_prior_logvar(h_t_ly2)
            z_prior = self.reparameterize(z_mean_t, z_logvar_t, random_sampling)
            if z_out is None:
                # If z_out is none it means z_t is z_1, hence store it in the format [batch_size, 1, z_dim]
                z_out = z_prior.unsqueeze(1)
                z_means = z_mean_t.unsqueeze(1)
                z_logvars = z_logvar_t.unsqueeze(1)
            else:
                # If z_out is not none, z_t is not the initial z and hence append it to the previous z_ts collected in z_out
                z_out = torch.cat((z_out, z_prior.unsqueeze(1)), dim=1)
                z_means = torch.cat((z_means, z_mean_t.unsqueeze(1)), dim=1)
                z_logvars = torch.cat((z_logvars, z_logvar_t.unsqueeze(1)), dim=1)
            z_t = z_post[:,i,:]
        return z_means, z_logvars, z_out

    def forward(self, x):
        f_mean, f_logvar, f_post, z_mean_post, z_logvar_post, z_post = self.encode_and_sample_post(x)
        z_mean_prior, z_logvar_prior, z_prior = self.sample_z_prior_train(z_post, random_sampling=self.training)

        z_flatten = z_post.view(-1, z_post.shape[2])
        pred = self.z_motion_predictor(z_flatten)

        f_expand = f_post.unsqueeze(1).expand(-1, self.frames, self.f_dim)
        zf = torch.cat((z_post, f_expand), dim=2)
        recon_x = self.decoder(zf)
        return f_mean, f_logvar, f_post, z_mean_post, z_logvar_post, z_post, z_mean_prior, z_logvar_prior, z_prior, \
               recon_x, pred



    def forward_single(self, x):
        f_mean, f_logvar, f_post, z_mean_post, z_logvar_post, z_post = self.encode_and_sample_post(x)
        z_mean_prior, z_logvar_prior, z_prior = self.sample_z_prior_train(z_post, random_sampling=True)

        f_expand = f_post.unsqueeze(1).expand(-1, self.frames, self.f_dim)
        zf = torch.cat((z_post, f_expand), dim=2)
        recon_x = self.decoder(zf)
        return f_mean, f_logvar, f_post, z_mean_post, z_logvar_post, z_post, z_mean_prior, z_logvar_prior, recon_x

    def forward_exchange(self, x):

        f_mean, f_logvar, f_post, z_mean_post, z_logvar_post, z_post = self.encode_and_sample_post(x)
        z_mean_prior, z_logvar_prior, z_prior = self.sample_z_prior_train(z_post, random_sampling=True)

        a = f_post[np.arange(0, f_post.shape[0], 2)]
        b = f_post[np.arange(1, f_post.shape[0], 2)]
        # f_mix = torch.stack((b, a), dim=1).view((-1, f_post.shape[-2], f_post.shape[-1]))
        f_mix = torch.stack((b, a), dim=1).view((-1, f_post.shape[1]))

        f_expand = f_mix.unsqueeze(1).expand(-1, self.frames, self.f_dim)
        zf = torch.cat((z_post, f_expand), dim=2)
        recon_x = self.decoder(zf)
        return f_mean, f_logvar, f_post, z_mean_post, z_logvar_post, z_post, z_mean_prior, z_logvar_prior, recon_x

    def forward_shuffle(self, x):
        f_mean, f_logvar, f_post, z_mean_post, z_logvar_post, z_post = self.encode_and_sample_post(x)
        z_mean_prior, z_logvar_prior, z_prior = self.self.sample_z_prior_train(z_post, random_sampling=True)


        perm = torch.LongTensor(np.random.permutation(f_post.shape[0]))
        f_mix = f_post[perm]

        f_expand = f_mix.unsqueeze(1).expand(-1, self.frames, self.f_dim)
        zf = torch.cat((z_post, f_expand), dim=2)
        recon_x = self.decoder(zf)
        return f_mean, f_logvar, f_post, z_mean_post, z_logvar_post, z_post, z_mean_prior, z_logvar_prior, recon_x


    def forward_exchange_matrix(self, x):

        f_mean, f_logvar, f_post, z_mean_post, z_logvar_post, z_post = self.encode_and_sample_post(x)

        result_matrix = list()
        for i in range(z_post.shape[0]):
            z_repeat = z_post[i].repeat(z_post.shape[0], 1, 1)
            f_expand = f_post.unsqueeze(1).expand(-1, self.frames, self.f_dim)
            zf = torch.cat((z_repeat, f_expand), dim=2)
            recon_x = self.decoder(zf)
            result_matrix.append(recon_x)
        return torch.stack(result_matrix, dim=0)

    def forward_exchange_flexible(self, x, zf_idx, zt_idx):

        f_mean, f_logvar, f_post, z_mean_post, z_logvar_post, z_post = self.encode_and_sample_post(x)

        result_matrix = list()
        f_num = len(zf_idx)
        for i in zt_idx:
            z_repeat = z_post[i].repeat(f_num, 1, 1)
            f_expand = (f_post[np.array(zf_idx)]).unsqueeze(1).expand(-1, self.frames, self.f_dim)
            zf = torch.cat((z_repeat, f_expand), dim=2)
            recon_x = self.decoder(zf)
            result_matrix.append(recon_x)
        return torch.stack(result_matrix, dim=0)

    def forward_fixed_motion(self, x):

        f_mean, f_logvar, f_post, z_mean_post, z_logvar_post, z_post = self.encode_and_sample_post(x)
        z_mean_prior, z_logvar_prior, z_prior = self.sample_z_prior_train(z_post, random_sampling=True)
        z_repeat = z_post[0].repeat(z_post.shape[0], 1, 1)
        f_expand = f_post.unsqueeze(1).expand(-1, self.frames, self.f_dim)
        zf = torch.cat((z_repeat, f_expand), dim=2)
        recon_x = self.decoder(zf)
        return f_mean, f_logvar, f_post, z_mean_post, z_logvar_post, z_post, z_mean_prior, z_logvar_prior, recon_x


    def forward_fixed_content(self, x):

        f_mean, f_logvar, f_post, z_mean_post, z_logvar_post, z_post = self.encode_and_sample_post(x)
        z_mean_prior, z_logvar_prior, z_prior = self.sample_z_prior_train(z_post, random_sampling=True)
        f_repeat = f_post[0].repeat(f_post.shape[0], 1)
        f_expand = f_repeat.unsqueeze(1).expand(-1, self.frames, self.f_dim)

        zf = torch.cat((z_post, f_expand), dim=2)
        recon_x = self.decoder(zf)
        return f_mean, f_logvar, f_post, z_mean_post, z_logvar_post, z_post, z_mean_prior, z_logvar_prior, recon_x

    # forward_generating and forward_generating2 are generating ONLY where the prior is used
    # Fixed the content and sample the motion vector.
    def forward_generating(self, x):
        f_mean, f_logvar, f_post, z_mean_post, z_logvar_post, z_post = self.encode_and_sample_post(x)
        z_mean_prior, z_logvar_prior, z_out = self.sample_z_prior_train(z_post, random_sampling=True)

        # z_mean, z_logvar, z = self.encode_z(conv_x, f)
        z_mean, z_logvar, z = z_mean_prior, z_logvar_prior, z_out
        f_repeat = f_post[0].repeat(f_post.shape[0], 1)
        f_expand = f_repeat.unsqueeze(1).expand(-1, self.frames, self.f_dim)

        # z prior + f post
        zf = torch.cat((z_out, f_expand), dim=2)
        recon_x = self.decoder(zf)
        return f_mean, f_logvar, f_post, z_mean_post, z_logvar_post, z_post, z_mean_prior, z_logvar_prior, recon_x

    # Fixed the motion vector and sample content prior.
    def forward_generating2(self, x):
        f_mean, f_logvar, f_post, z_mean_post, z_logvar_post, z_post = self.encode_and_sample_post(x)

        # z_mean, z_logvar, z = z_mean_prior, z_logvar_prior, z_out
        # f_repeat = f[0].repeat(100, 1)
        z_repeat = z_post[0].repeat(z_post.shape[0], 1, 1)
        # f_prior = self.reparameterize(torch.zeros([f_mean.shape[0], f_mean.shape[2]]).cuda(),
        #                                 torch.zeros(f_logvar.shape[0], f_logvar.shape[2]).cuda(), random_sampling=True)
        f_prior = self.reparameterize(torch.zeros(f_mean.shape).cuda(), torch.zeros(f_logvar.shape).cuda(),
                                                                                            random_sampling=True)
        f_expand = f_prior.unsqueeze(1).expand(-1, self.frames, self.f_dim)
        zf = torch.cat((z_repeat, f_expand), dim=2)
        recon_x = self.decoder(zf)
        return recon_x


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

    def encode_f(self, x):
        lstm_out, _ = self.f_lstm(x)
        # The features of the last timestep of the forward RNN is stored at the end of lstm_out in the first half, and the features
        # of the "first timestep" of the backward RNN is stored at the beginning of lstm_out in the second half
        # For a detailed explanation, check: https://gist.github.com/ceshine/bed2dadca48fe4fe4b4600ccce2fd6e1
        backward = lstm_out[:, 0, self.hidden_dim:2 * self.hidden_dim]
        frontal = lstm_out[:, self.frames - 1, 0:self.hidden_dim]
        lstm_out = torch.cat((frontal, backward), dim=1)
        mean = self.f_mean(lstm_out)
        logvar = self.f_logvar(lstm_out)
        return mean, logvar, self.reparameterize(mean, logvar, self.training)

    # def encode_z(self, x):
    #
    #     # The expansion is done to match the dimension of x and f, used for concatenating f to each x_t
    #     f_expand = f.unsqueeze(1).expand(-1, self.frames, self.f_dim)
    #

    def reparameterize(self, mean, logvar, random_sampling=True):
        # Reparametrization occurs only if random sampling is set to true, otherwise mean is returned
        if random_sampling is True:
            eps = torch.randn_like(logvar)
            std = torch.exp(0.5 * logvar)
            z = mean + eps * std
            return z
        else:
            return mean
    # def sample_z_transformer_prior_train(self, batch_size, random_sampling=True):
    #     # This will ultimately store all z_s in the format [batch_size, frames, z_dim]
    #     # batch_size = z_post.shape[0]
    #     input_zero = torch.zeros(batch_size, self.frames, self.hidden_dim).cuda()
    #     hidden_vector = self.z_prior_transformer(input_zero)
    #     z_mean = self.z_prior_mean(hidden_vector)
    #     z_logvar = self.z_prior_logvar(hidden_vector)
    #     z_prior = self.reparameterize(z_mean, z_logvar, random_sampling)
    #
    #     return z_mean, z_logvar, z_prior
