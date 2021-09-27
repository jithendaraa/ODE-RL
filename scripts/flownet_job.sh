#!/bin/bash
#SBATCH --time=0:40:00
#SBATCH --account=def-ebrahimi
#SBATCH --mem=32G
#SBATCH --gres=gpu:v100:1
#SBATCH --job-name=FlowNet_S3VAE
#SBATCH --cpus-per-task=6
#SBATCH --mail-user=jithen.subra@gmail.com
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
#SBATCH --mail-type=REQUEUE
#SBATCH --mail-type=ALL
#SBATCH --output=out/FlowNet/FlowNet-%j.out

id=$1
start=`date +%s`
echo "Starting run at: `date`"
cd ~/projects/rrg-ebrahimi/jithen/ODE-RL/flownet2-pytorch && source flownet_env/bin/activate
python main.py --inference --model FlowNet2 --save_flow --inference_dataset MMNIST --inference_dataset_root ~/scratch/datasets/MovingMNIST_video --inference_visualize --vid ${id}
echo "Ending run at: `date`"
end=`date +%s`
runtime=$((end-start))
echo "Program time: $runtime"
