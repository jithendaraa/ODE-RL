#!/bin/bash
#SBATCH --time=2:00:00
#SBATCH --account=def-ebrahimi
#SBATCH --mem=32G
#SBATCH --gres=gpu:v100l:1
#SBATCH --job-name=MovingMNIST
#SBATCH --output=out/RSSM_eval-%j.out
#SBATCH --cpus-per-task=6
#SBATCH --mail-user=jithen.subra@gmail.com
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
#SBATCH --mail-type=REQUEUE
#SBATCH --mail-type=ALL
python evaluation.py --logdir logs --id original_dreamer_bs_4_steps_50000 --configs defaults moving_mnist --use_wandb False --steps 5e4 --batch_size 4