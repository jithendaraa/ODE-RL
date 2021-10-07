#!/bin/bash
#SBATCH --partition=main
#SBATCH --time=32:00:00
#SBATCH --mem=48G
#SBATCH --gres=gpu:v100:1
#SBATCH --job-name=MovingMNIST
#SBATCH --cpus-per-task=6
#SBATCH --mail-user=jithen.subra@gmail.com
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
#SBATCH --mail-type=REQUEUE
#SBATCH --mail-type=ALL

id=$1
start=`date +%s`
echo $start

python main.py --config defaults ${id} mila

echo $end
end=`date +%s`
runtime=$((end-start))
echo "Program time: $runtime"