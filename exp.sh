#!/bin/bash
id="train_mmnist_cgru"
output_file="out/ConvGRU/""$id""-%j.out"
RES=$(sbatch --job-name ${id} --output ${output_file} scripts/convgru_job.sh ${id})
train_job_id=${RES##* }
echo "$train_job_id"" ""${train_job_id}" >> out/job_logs.txt

id="test_mmnist_cgru"
output_file="out/ConvGRU/""$id""-%j.out"
RES=$(sbatch --job-name ${id} --output ${output_file} --dependency=afterok:${train_job_id} scripts/convgru_job.sh ${id})
test_job_id=${RES##* }
echo "$test_job_id"" ""${train_job_id}" >> out/job_logs.txt