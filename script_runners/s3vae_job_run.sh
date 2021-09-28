#!/bin/bash
echo ""

# 1. Train S3VAE on Moving MNIST
id=$1
output_file="out/S3VAE/""$id""-%j.out"
RES=$(sbatch --job-name ${id} --output ${output_file} --time 23:00:00 scripts/run_job.sh ${id})
train_job_id=${RES##* }
echo "train_job_id"" ""${train_job_id}"" -> ""${id}" >> out/job_logs.txt
