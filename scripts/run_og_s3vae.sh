#!/bin/bash
#SBATCH --time=27:00:00
#SBATCH --account=def-ebrahimi
#SBATCH --mem=32G
#SBATCH --gres=gpu:v100:1
#SBATCH --job-name=MovingMNIST
#SBATCH --cpus-per-task=6
#SBATCH --mail-user=jithen.subra@gmail.com
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
#SBATCH --mail-type=REQUEUE
#SBATCH --mail-type=ALL

dataset=$1
train=$2
cd code_sprite

if [ $2 == 'train' ]
then
    python train_DS_VAE_sprite.py --dataset ${dataset}
elif [ $2 == 'test' ]
then
    python test_DS_VAE_Sprite_Cls_disagree.py --dataset ${dataset}
else
    echo "Wrong command"
fi

