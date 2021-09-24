#!/bin/bash
#SBATCH --time=12:00:00
#SBATCH --account=def-ebrahimi
#SBATCH --mem=32G
#SBATCH --gres=gpu:v100:1
#SBATCH --job-name=OG_S3VAE_Sprite
#SBATCH --cpus-per-task=6
#SBATCH --mail-user=jithen.subra@gmail.com
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
#SBATCH --mail-type=REQUEUE
#SBATCH --mail-type=ALL
#SBATCH --output=out/OG_S3VAE/OG_s3vae-%j.out

start=`date +%s`
echo "Starting run at: `date`"
source ~/ENV/bin/activate
cd code_sprite && python train_DS_VAE_sprite.py
echo "Ending run at: `date`"
end=`date +%s`
runtime=$((end-start))
echo "Program time: $runtime"
