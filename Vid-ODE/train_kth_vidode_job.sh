#!/bin/bash
#SBATCH --mail-user=jithen.subra@gmail.com
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
#SBATCH --mail-type=REQUEUE
#SBATCH --mail-type=ALL
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:v100l
#SBATCH --mem=32G
#SBATCH --output=VidODENRU2_cwvae_epoch_100_batch4_train_64frames-%j.out
python main.py --phase train --epoch 100 --dataset kth --nru2 --extrap --input_sequence 36 --output_sequence 50 -b 4 --ckpt_save_freq 5000 --image_print_freq 10 --unequal
