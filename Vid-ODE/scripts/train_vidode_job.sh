#!/bin/bash
#SBATCH --time=29:00:00
#SBATCH --account=def-ebrahimi
#SBATCH --mail-user=jithen.subra@gmail.com
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
#SBATCH --mail-type=REQUEUE
#SBATCH --mail-type=ALL
#SBATCH --gres=gpu:v100l:1
#SBATCH --mem=32G
BATCH_SIZE=$1
EPOCHS=$2
frame_dims=64
is=$3
os=$4
ckpt=5000
img=5000
nru=$5
nru2=$6
dataset=$7

start=`date +%s`
echo "python main.py --nru ${nru} -nl $nl --nru2 ${nru2} -b ${BATCH_SIZE} -e ${EPOCHS} -fd ${frame_dims} -is ${is} -os ${os} --ckpt_save_freq ${ckpt} --image_print_freq ${img} --unequal -d ${dataset} --extrap -p train"
echo "Starting run at: `date`"
python main.py --nru ${nru} --nru2 ${nru2} -b ${BATCH_SIZE} -e ${EPOCHS} -fd ${frame_dims} -is ${is} -os ${os} --ckpt_save_freq ${ckpt} --image_print_freq ${img} --unequal -d ${dataset} --extrap -p train 
echo "Ending run at: `date`"
end=`date +%s`
runtime=$((end-start))
echo "Program time: $runtime"