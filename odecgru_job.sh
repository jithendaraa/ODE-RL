#!/bin/bash
echo ""

# 1. Train ODEConv on Moving MNIST
id=$1
job_script='run_job.sh'
output_file="out/ODEConv/""$id""-%j.out"
RES=$(sbatch --job-name ${id} --output ${output_file} --time 25:00:00 scripts/${job_script} ${id})
train_job_id=${RES##* }
echo "train_job_id"" ""${train_job_id}"" -> ""${id}" >> out/job_logs.txt