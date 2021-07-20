#!/bin/bash
#SBATCH --time=5:30:00
#SBATCH --account=def-ebrahimi
#SBATCH --mem=32G
#SBATCH --gres=gpu:v100l:1
#SBATCH --job-name=MovingMNIST
#SBATCH --output=out/RSSM-%j.out
#SBATCH --cpus-per-task=6
#SBATCH --mail-user=jithen.subra@gmail.com
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
#SBATCH --mail-type=REQUEUE
#SBATCH --mail-type=ALL
batch_size=4
steps=50000
dataset='moving_mnist'
python_file=$1 # usually points to dreamerv2/dreamer.py
ID=original_dreamer_bs_${batch_size}_steps_${steps}

act_env

echo "Running RSSM.."
echo "Batch size: ""$batch_size"
echo "Steps: ""$steps"
echo "Dataset: ""$dataset"
echo "Python File: ""$python_file"
echo "ID: ""$ID"

echo "Starting run at: `date`"
command="python ${python_file} --logdir logs --id ${ID} --configs defaults ${dataset} --use_wandb False --steps ${steps} --batch_size ${batch_size}"
echo "$command"
python ${python_file} --logdir logs --id ${ID} --configs defaults ${dataset} --use_wandb False --steps ${steps} --batch_size ${batch_size}
echo "Ending run at: `date`"