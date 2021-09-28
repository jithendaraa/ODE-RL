#!/bin/bash
echo ""

id=$1
output_file="out/FlowNet/""$id""-%j.out"
RES=$(sbatch --job-name ${id} --output ${output_file} --time 0:40:00 scripts/flownet_job.sh ${id})
train_job_id=${RES##* }
echo "train_job_id"" ""${train_job_id}"" -> ""${id}" >> out/job_logs.txt
