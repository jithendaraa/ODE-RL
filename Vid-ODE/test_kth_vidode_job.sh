#!/bin/bash
#SBATCH --mail-user=jithen.subra@gmail.com
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
#SBATCH --mail-type=REQUEUE
#SBATCH --mail-type=ALL
#SBATCH --time=1:00:00
#SBATCH --gres=gpu:v100l
#SBATCH --mem=32G
#SBATCH --output=VidODE_cwvae_epoch_500_test-%j.out
python main.py --phase test_met --test_dir storage/logs/0511/datasetkth_extrapTrue_irregularFalse_runBackTrue_vid_ode_nruFalse_nru2False_epoch500_batch4_unequalTrue_36_50_86 --dataset kth --extrap
