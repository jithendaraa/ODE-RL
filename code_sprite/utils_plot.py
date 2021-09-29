import numpy as np
import utils

# # --------- plotting funtions ------------------------------------
def plot(x, epoch):
    nsample = 20
    gen_seq = [[] for i in range(nsample)]
    gt_seq = [x[i] for i in range(len(x))]

    for s in range(nsample):
        frame_predictor.hidden = frame_predictor.init_hidden()
        posterior.hidden = posterior.init_hidden()
        prior.hidden = prior.init_hidden()
        gen_seq[s].append(x[0])
        x_in = x[0]
        for i in range(1, opt.n_eval):
            h = ds_vae.encoder(x_in)
            if opt.last_frame_skip or i < opt.n_past:
                h, skip = h
            else:
                h, _ = h
            h = h.detach()
            if i < opt.n_past:
                h_target = ds_vae.encoder(x[i])
                h_target = h_target[0].detach()
                z_t, _, _ = posterior(h_target)
                prior(h)
                frame_predictor(torch.cat([h, z_t], 1))
                x_in = x[i]
                gen_seq[s].append(x_in)
            else:
                z_t, _, _ = prior(h)
                h = frame_predictor(torch.cat([h, z_t], 1)).detach()
                x_in = ds_vae.decoder([h, skip]).detach()
                gen_seq[s].append(x_in)

    to_plot = []
    gifs = [[] for t in range(opt.n_eval)]
    nrow = min(opt.batch_size, 10)
    for i in range(nrow):
        # ground truth sequence
        row = []
        for t in range(opt.n_eval):
            row.append(gt_seq[t][i])
        to_plot.append(row)

        # best sequence
        min_mse = 1e7
        for s in range(nsample):
            mse = 0
            for t in range(opt.n_eval):
                mse += torch.sum((gt_seq[t][i].data.cpu() - gen_seq[s][t][i].data.cpu()) ** 2)
            if mse < min_mse:
                min_mse = mse
                min_idx = s

        s_list = [min_idx,
                  np.random.randint(nsample),
                  np.random.randint(nsample),
                  np.random.randint(nsample),
                  np.random.randint(nsample)]
        for ss in range(len(s_list)):
            s = s_list[ss]
            row = []
            for t in range(opt.n_eval):
                row.append(gen_seq[s][t][i])
            to_plot.append(row)
        for t in range(opt.n_eval):
            row = []
            row.append(gt_seq[t][i])
            for ss in range(len(s_list)):
                s = s_list[ss]
                row.append(gen_seq[s][t][i])
            gifs[t].append(row)

    fname = '%s/gen/sample_%d.png' % (opt.log_dir, epoch)
    utils.save_tensors_image(fname, to_plot)

    fname = '%s/gen/sample_%d.gif' % (opt.log_dir, epoch)
    utils.save_gif(fname, gifs)

#
def plot_rec(x, epoch):
    frame_predictor.hidden = frame_predictor.init_hidden()
    posterior.hidden = posterior.init_hidden()
    gen_seq = []
    gen_seq.append(x[0])
    x_in = x[0]
    for i in range(1, opt.n_past + opt.n_future):
        h = ds_vae.encoder(x[i - 1])
        h_target = ds_vae.encoder(x[i])
        if opt.last_frame_skip or i < opt.n_past:
            h, skip = h
        else:
            h, _ = h
        h_target, _ = h_target
        h = h.detach()
        h_target = h_target.detach()
        z_t, _, _ = posterior(h_target)
        if i < opt.n_past:
            frame_predictor(torch.cat([h, z_t], 1))
            gen_seq.append(x[i])
        else:
            h_pred = frame_predictor(torch.cat([h, z_t], 1))
            x_pred = ds_vae.decoder([h_pred, skip]).detach()
            gen_seq.append(x_pred)

    to_plot = []
    nrow = min(opt.batch_size, 10)
    for i in range(nrow):
        row = []
        for t in range(opt.n_past + opt.n_future):
            row.append(gen_seq[t][i])
        to_plot.append(row)
    fname = '%s/gen/rec_%d.png' % (opt.log_dir, epoch)
    utils.save_tensors_image(fname, to_plot)

import torch
# plot reconstruction
def plot_rec_new(x, epoch, opt, model):
    _, _, _, _, _, _, _, _, recon_x = model.forward_single(x)
    to_plot = []
    nrow = min(opt.batch_size, 5)
    for i in range(nrow):
        row = []
        for t in range(opt.frames):
            row.append(x[i][t])
        to_plot.append(row)

        row = []
        for t in range(opt.frames):
            row.append(recon_x[i][t])
        to_plot.append(row)
    fname = '%s/gen/rec_%d.png' % (opt.log_dir, epoch)
    utils.save_tensors_image(fname, to_plot)


# plot reconstruction
def plot_rec_new_single(x, epoch, opt, model):
    _, _, _, _, _, _, _, _, recon_x = model.forward_single(x)
    to_plot = []
    nrow = min(opt.batch_size, 5)
    for i in range(nrow):
        row = []
        for t in range(opt.frames):
            row.append(x[i][t])
        to_plot.append(row)

        row = []
        for t in range(opt.frames):
            row.append(recon_x[i][t])
        to_plot.append(row)
    fname = '%s/gen/rec_%d.png' % (opt.log_dir, epoch)
    utils.save_tensors_image(fname, to_plot)


# plot reconstruction
def plot_rec_exchange(x, epoch, opt, model):
    _, _, _, _, _, _, _, _, recon_x = model.forward_exchange(x)
    to_plot = []
    nrow = min(x.shape[0], 6)
    for i in range(nrow):
        row = []
        for t in range(opt.frames):
            row.append(x[i][t])
        to_plot.append(row)
        row = []
        for t in range(opt.frames):
            row.append(recon_x[i][t])
        to_plot.append(row)
    fname = '{}/gen/{}_mix.png'.format(opt.log_dir, epoch)
    utils.save_tensors_image(fname, to_plot)


# plot reconstruction
def plot_rec_exchange_paper(x, epoch, opt, model, index):
    _, _, _, _, _, _, _, _, recon_x = model.forward_exchange(x)
    to_plot = []
    nrow = min(opt.batch_size, 6)

    i = index
    row = []
    for t in range(opt.frames):
        to_plot = [[x[i][t]]]
        fname = '%s/gen/org_%d_%d.png' % (opt.log_dir, i, t)
        utils.save_tensors_image(fname, to_plot)
    for t in range(opt.frames):
        to_plot = [[recon_x[i][t]]]
        fname = '%s/gen/rec_%d_%d.png' % (opt.log_dir, i, t)
        utils.save_tensors_image(fname, to_plot)

    i = index+1
    row = []
    for t in range(opt.frames):
        to_plot = [[x[i][t]]]
        fname = '%s/gen/org_%d_%d.png' % (opt.log_dir, i, t)
        utils.save_tensors_image(fname, to_plot)
    for t in range(opt.frames):
        to_plot = [[recon_x[i][t]]]
        fname = '%s/gen/rec_%d_%d.png' % (opt.log_dir, i, t)
        utils.save_tensors_image(fname, to_plot)


# plot reconstruction
def plot_rec_fixed_motion(x, epoch, opt, model):
    _, _, _, _, _, _, _, _, recon_x = model.forward_fixed_motion(x)
    to_plot = []
    nrow = min(opt.batch_size, 10)
    row = []
    row.append(x[0][0])
    for t in range(opt.frames):
        row.append(x[0][t])
    to_plot.append(row)

    for i in range(nrow):
        row = []
        row.append(x[i][0])
        for t in range(opt.frames):
            row.append(recon_x[i][t])
        to_plot.append(row)
    fname = '%s/gen/rec_%d_fixed_motion.png' % (opt.log_dir, epoch)
    utils.save_tensors_image(fname, to_plot)

# plot reconstruction
def plot_rec_fixed_content(x, epoch, opt, model):
    _, _, _, _, _, _, _, _, recon_x = model.forward_fixed_content(x)
    to_plot = []
    nrow = min(opt.batch_size, 6)
    row = []
    for i in range(nrow):
        row = []
        for t in range(opt.frames):
            row.append(x[i][t])
        to_plot.append(row)
        row = []
        for t in range(opt.frames):
            row.append(recon_x[i][t])
        to_plot.append(row)
    fname = '%s/gen/rec_%d_fixed_content.png' % (opt.log_dir, epoch)
    utils.save_tensors_image(fname, to_plot)

# plot fixed content and sample motion
def plot_rec_generating(x, epoch, opt, model):
    #_, _, _, _, _, _, _, _, recon_x = model.forward_generating_nframe(x, n_frame)
    _, _, _, _, _, _, _, _, recon_x = model.forward_generating(x)

    to_plot = []
    nrow = min(opt.batch_size, 10)
    for i in range(nrow):
        row = []
        for t in range(opt.frames):
            row.append(recon_x[i][t])
        to_plot.append(row)
    fname = '%s/gen/rec_%d_generation.png' % (opt.log_dir, epoch)
    utils.save_tensors_image(fname, to_plot)

# plot fixed content and sample motion
def plot_rec_generating_paper(x, epoch,n_frame, idx, opt, model):
    _, _, _, _, _, _, _, _, recon_x = model.forward_generating_nframe(x, n_frame)

    nrow = opt.batch_size #min(opt.batch_size, 10)
    for i in [idx]: #range(nrow):
        for t in range(opt.frames):
            to_plot = [[recon_x[idx][t]]]
            fname = '%s/gen/rec_%d_generation_%d_%d.png' % (opt.log_dir,epoch, i, t)
            utils.save_tensors_image(fname, to_plot)


# plot fixed motion and sample content
def plot_rec_generating2(x, epoch, opt, model):
    recon_x = model.forward_generating2(x)

    to_plot = []
    nrow = min(opt.batch_size, 10)
    for i in range(nrow):
        row = []
        for t in range(opt.frames):
            row.append(recon_x[i][t])
        to_plot.append(row)
    fname = '%s/gen/rec_%d_generation2.png' % (opt.log_dir, epoch)
    utils.save_tensors_image(fname, to_plot)


# plot fixed motion and sample content
def plot_rec_generating2_paper(x, epoch, opt, model):
    recon_x = model.forward_generating2(x)

    to_plot = []
    # nrow = min(opt.batch_size, 10)
    nrow = opt.batch_size# min(opt.batch_size, 10)
    for i in range(nrow):
        for t in range(opt.frames):
            to_plot = [[recon_x[i][t]]]
            fname = '%s/gen/rec_%d_generation2_%d_%d.png' % (opt.log_dir,epoch, i, t)
            utils.save_tensors_image(fname, to_plot)



# plot # Sample both content prior and motion prior.
def plot_rec_generating3(x, epoch, n_frame, opt, model):
    _, _, _, _, _, _, _, _, recon_x = model.forward_generating3_nframe(x, n_frame)
    # _, _, _, _, _, _, _, _, recon_x = model.forward_generating(x)

    to_plot = []
    nrow = opt.batch_size #min(opt.batch_size, 10)
    for i in range(nrow):
        row = []
        for t in range(n_frame): #opt.frames):
            row.append(recon_x[i][t])
        to_plot.append(row)
    fname = '%s/gen/rec_%d_generation3.png' % (opt.log_dir, epoch)
    utils.save_tensors_image(fname, to_plot)

# plot # Sample both content prior and motion prior.
def plot_rec_generating3_paper(x, epoch,n_frame, idx, opt, model):
    _, _, _, _, _, _, _, _, recon_x = model.forward_generating3_nframe(x, n_frame)
    nrow = opt.batch_size #min(opt.batch_size, 10)
    for i in [idx]: #range(nrow):
        for t in range(n_frame):
            to_plot = [[recon_x[idx][t]]]
            fname = '%s/gen/rec_%d_generation_%d_%d.png' % (opt.log_dir,epoch, i, t)
            utils.save_tensors_image(fname, to_plot)


# plot transient
def plot_rec_transient(x, epoch, opt, model):
    _, _, _, _, _, _, _, _, recon_x = model.forward_transient(x)
    to_plot = []
    nrow = min(x.shape[0], 6)
    x = x.view(1, -1, 3, 64, 64)

    row = []
    for t in range(x.shape[1]):
        row.append(x[0][t])
    to_plot.append(row)
    row = []
    for t in range(x.shape[1]):
        row.append(recon_x[0][t])
    to_plot.append(row)
    fname = '{}/gen/{}_mix.png'.format(opt.log_dir, epoch)
    utils.save_tensors_image(fname, to_plot)

# plot transient
def plot_rec_transient_paper(x, epoch, opt, model):
    _, _, _, _, _, _, _, _, recon_x = model.forward_transient(x)
    to_plot = []
    nrow = min(x.shape[0], 6)
    x = x.view(1, -1, 3, 64, 64)


    for i in range(recon_x.shape[0]): #range(nrow):
        for t in range(recon_x.shape[1]):
            to_plot = [[recon_x[i][t]]]
            fname = '%s/gen/%s_transient_%d_%d.png' % (opt.log_dir, epoch, i, t)
            utils.save_tensors_image(fname, to_plot)


