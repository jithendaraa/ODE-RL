def plot_metrics_vs_n_frames(avg_psnrs, avg_mses, avg_ssims, id):
    if os.path.isdir('plots') is False: os.mkdir('plots')
    save_path = os.path.join('plots', id+'-metrics.png')
    title = 'MSE, PSNR, and SSIM vs. predicted frame length(' + str(id) + ')'

    x = np.arange(len(avg_psnrs))
    plt.plot(x, avg_psnrs, color='red', label='PSNR (Using Avg MSE across batches, batch size, and t)')
    plt.plot(x, avg_mses, color='blue', label='MSE (Avg across batches, batch size, and t)')
    plt.plot(x, avg_ssims, color='black', label='SSIM (Avg across batches, batch size, and t)')
    plt.xlabel('predicted frame length')
    plt.legend(loc='upper right', frameon=False)
    plt.title(title)
    plt.savefig(save_path)

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
