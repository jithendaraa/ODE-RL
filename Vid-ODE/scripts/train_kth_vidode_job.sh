#!/bin/bash
#SBATCH --time=24:00:00
#SBATCH --account=def-ebrahimi
#SBATCH --mail-user=jithen.subra@gmail.com
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
#SBATCH --mail-type=REQUEUE
#SBATCH --mail-type=ALL
#SBATCH --gres=gpu:v100:1
#SBATCH --mem=32G
BATCH_SIZE=$1
EPOCHS=$2
frame_dims=$3
is=$4
os=$5
ckpt=$6
img=$7
nru=$8
nru2=$9

start=`date +%s`
echo "python main.py --nru ${nru} --nru2 ${nru2} -b ${BATCH_SIZE} -e ${EPOCHS} -fd ${frame_dims} -is ${is} -os ${os} --ckpt_save_freq ${ckpt} --image_print_freq ${img} --unequal -d kth --extrap -p train"
echo "Starting run at: `date`"
python main.py --nru ${nru} --nru2 ${nru2} -b ${BATCH_SIZE} -e ${EPOCHS} -fd ${frame_dims} -is ${is} -os ${os} --ckpt_save_freq ${ckpt} --image_print_freq ${img} --unequal -d kth --extrap -p train
echo "Ending run at: `date`"
end=`date +%s`
runtime=$((end-start))
echo "Program time: $runtime"