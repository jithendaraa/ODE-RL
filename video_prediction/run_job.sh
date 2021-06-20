#!/bin/bash
#SBATCH --time=4:00:00
#SBATCH --account=def-ebrahimi
#SBATCH --mem=32G
#SBATCH --gres=gpu:v100:1
#SBATCH --job-name=MovingMNIST
#SBATCH --cpus-per-task=6
#SBATCH --mail-user=pranav2109@hotmail.com
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
#SBATCH --mail-type=REQUEUE
#SBATCH --mail-type=ALL

source ~/tf_env/bin/activate
start=`date +%s`
echo "Starting run at: `date`"
echo " Running python ConvGRU.py"
python ConvGRU.py
echo "Ending run at: `date`"
end=`date +%s`
runtime=$((end-start))
echo "Program time: $runtime"

