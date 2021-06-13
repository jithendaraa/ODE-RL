#!/bin/bash
id="train_mmnist_odecgru"
output_file="out/ODEConv/""$id""-%j.out"
RES=$(sbatch --job-name ${id} --output ${output_file} --time 25:00:00 scripts/odeconv_job.sh ${id})
train_job_id=${RES##* }
echo "train_job_id"" ""${train_job_id}"" -> ""${id}" >> out/job_logs.txt

id="test_mmnist_odecgru"
output_file="out/ODEConv/""$id""-%j.out"
RES=$(sbatch --job-name ${id} --output ${output_file} --time 3:00:00 scripts/odeconv_job.sh ${id})
test_job_id=${RES##* }
echo "test_job_id"" ""${test_job_id}"" -> ""${id}" >> out/job_logs.txt

# id="test_mmnist_cgru"
# output_file="out/ConvGRU/""$id""-%j.out"
# RES=$(sbatch --job-name ${id} --output ${output_file} --time 3:00:00 scripts/convgru_job.sh ${id})
# test_job_id=${RES##* }
# echo "test_job_id"" ""${test_job_id}"" -> ""${id}" >> out/job_logs.txt

