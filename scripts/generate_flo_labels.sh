#!/bin/bash
#SBATCH --time=0:30:00
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

vid=$1
jump=$2
dataset=$3
start=`date +%s`
echo "Starting run at: `date`"
python get_labels_from_pred_flow.py --vid ${vid} --jump ${jump} --dataset ${dataset}
echo "Ending run at: `date`"
end=`date +%s`
runtime=$((end-start))
echo "Program time: $runtime"
