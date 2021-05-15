#!/bin/bash
#SBATCH --time=48:00:00
#SBATCH --account=def-ebrahimi
#SBATCH --mail-user=jithen.subra@gmail.com
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
#SBATCH --mail-type=REQUEUE
#SBATCH --mail-type=ALL
#SBATCH --gres=gpu:v100:1
#SBATCH --mem=32G
EPOCHS=200
BATCH_SIZE=4
frame_dims=64
is=36
os=200
a=10
b=2
ckpt=$((a * EPOCHS))
img=$((b * EPOCHS))
#SBATCH --output=VidODE_epoch_${EPOCHS}_batch${BATCH_SIZE}_train${frame_dims}-%j.out
echo "python main.py -b ${BATCH_SIZE} -e ${EPOCHS} -fd ${frame_dims} -is ${is} -os ${os} --ckpt_save_freq 10*${EPOCHS} --image_print_freq 2*${EPOCHS} --unequal -d minerl --extrap -p train"
python main.py -b ${BATCH_SIZE} -e ${EPOCHS} -fd ${frame_dims} -is ${is} -os ${os} --ckpt_save_freq ${ckpt} --image_print_freq ${img} --unequal -d minerl --extrap -p train