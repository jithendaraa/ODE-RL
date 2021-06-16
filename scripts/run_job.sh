#!/bin/bash
#SBATCH --time=18:00:00
#SBATCH --account=def-ebrahimi
#SBATCH --mem=32G
#SBATCH --gres=gpu:v100l:1
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
echo "Starting run at: `date`"
python main.py --config defaults ${id}
echo "Ending run at: `date`"
end=`date +%s`
runtime=$((end-start))
echo "Program time: $runtime"
