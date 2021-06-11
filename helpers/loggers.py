import os
import numpy as np
import json

def print_exp_details(opt, n_batches):
    print()
    print("Exp ID: ", opt.id)
    print(f"Logging to {opt.logdir}")
    if opt.phase == 'train':
        print("Training the", opt.model, "model on", opt.dataset, 'for', n_batches, 'batches of batch size', opt.batch_size)
        print("Input frames:", opt.train_in_seq)
        print("Output frames:", opt.train_out_seq)

    else:
        print("Training the", opt.model, "model on", opt.dataset, 'for', n_batches, 'batches of batch size', opt.batch_size)
        print("Input frames:", opt.test_in_seq)
        print("Output frames:", opt.test_out_seq)
    print()

def log_after_epoch(epoch_num, loss, metrics, step, start_time, total_steps, opt=None):
    
    if (epoch_num % opt.log_freq) == 0:
        metrics['train_losses_x'].append(step)
        metrics['train_losses'].append(loss)
        et = time.time() - start_time
        et = str(datetime.timedelta(seconds=et))[:-7]
        log = f"Elapsed [{et}] Epoch [{epoch_num:03d}/{opt.epochs:03d}]\t"\
                        f"Iterations [{(step):7d}/{(total_steps):7d}] \t"\
                        f"Mse [{loss:.10f}]\t"
        print(log)
        # TODO: Output train loss to tensorboard

    return metrics

def log_test_loss(opt, step, loss):
    if (step % opt.test_log_freq) == 0:
        print(f"Test loss for step {step}: {loss}")

def plot_metrics_vs_n_frames(avg_psnrs, avg_mses, avg_ssims, exp_id, metrics_logdir='metrics'):
    
    print()
    if os.path.isdir('plots') is False: os.mkdir('plots')
    if os.path.isdir(metrics_logdir) is False: os.mkdir(metrics_logdir)
    png_save_path = os.path.join('plots', id+'-metric_plots.png')
    json_save_path = os.path.join(metrics_logdir, id+'-metrics.pickle')

    title = 'MSE, PSNR, and SSIM vs. predicted frame length(' + str(exp_id) + ')'

    x = np.arange(len(avg_psnrs))
    plt.plot(x, avg_psnrs, color='red', label='PSNR (Using Avg MSE across batches, batch size, and t)')
    plt.plot(x, avg_mses, color='blue', label='MSE (Avg across batches, batch size, and t)')
    plt.plot(x, avg_ssims, color='black', label='SSIM (Avg across batches, batch size, and t)')
    plt.xlabel('predicted frame length')
    plt.legend(loc='upper right', frameon=False)
    plt.title(title)
    plt.savefig(png_save_path)
    print(f"Saved metrics plots for testing id {exp_id} at {png_save_path}")

    metrics = {
        'MSE':  avg_mses,
        'PSNR': avg_psnrs,
        'SSIM': avg_ssims,
        'id':   exp_id,
        'opt':  opt
    }

    with open(json_save_path, 'wb') as handle:
        json.dump(metrics, handle, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"Saved metrics.json for testing id {exp_id} at {json_save_path}")


def log_final_test_metrics(test_loss, avg_mse, avg_psnr, avg_ssim, id):
    print()
    print("NOTE: All metrics are normalized across test_batches, batch_size, and predicted frame length")
    print(f"Metrics after testing id: {id}")
    print("Final Test loss on entire dataset after evaluation:", test_loss)
    print("MSE:", avg_mse)
    print("PSNR:", avg_psnr)
    print("SSIM:", avg_ssim)

def log_metrics_to_tb(avg_mses, avg_psnrs, avg_ssims, tb):
    assert len(avg_mses) == len(avg_psnrs)
    assert len(avg_psnrs) == len(avg_ssims)

    for i in range(len(avg_mses)):
        tb.add_scalar('MSE (avgd across batches, batch_size, and timesteps)', avg_mses[i], i)  # MSE vs. Number of Predicted frames
        tb.add_scalar('SSIM (avgd across batches, batch_size, and timesteps)', avg_ssims[i], i)  # SSIM vs. Number of Predicted frames
        tb.add_scalar('PSNR (using MSE avgd across batches, batch_size, and timesteps)', avg_psnrs[i], i)  # PSNR vs. Number of Predicted frames

    print()
    print("Logged metrics (MSE, PSNR, SSIM) to tensorboard")
    print()