#!/bin/bash
echo ""

# 1. Train and test ConvGRU on Moving MNIST
id=$1
job_script='run_job.sh'
output_file="out/ConvGRU/""$id""-%j.out"
RES=$(sbatch --job-name ${id} --output ${output_file} --time 10:00:00 scripts/${job_script} ${id})
train_job_id=${RES##* }
echo "train_job_id"" ""${train_job_id}"" -> ""${id}" >> out/job_logs.txt

# id="test_mmnist_cgru"
# job_script='convgru_job.sh'
# output_file="out/ConvGRU/""$id""-%j.out"
# RES=$(sbatch --job-name ${id} --output ${output_file} --time 2:00:00 scripts/${job_script} ${id})
# test_job_id=${RES##* }
# echo "test_job_id"" ""${test_job_id}"" -> ""${id}" >> out/job_logs.txt

# 2. Train and test ODEConv on Moving MNIST
# id="train_mmnist_odecgru"
# job_script='odeconv_job.sh'
# output_file="out/ODEConv/""$id""-%j.out"
# RES=$(sbatch --job-name ${id} --output ${output_file} --time 25:00:00 scripts/${job_script} ${id})
# train_job_id=${RES##* }
# echo "train_job_id"" ""${train_job_id}"" -> ""${id}" >> out/job_logs.txt

# id="test_mmnist_odecgru"
# job_script='odeconv_job.sh'
# output_file="out/ODEConv/""$id""-%j.out"
# RES=$(sbatch --job-name ${id} --dependency=afterok:${train_job_id} --output ${output_file} --time 6:00:00 scripts/${job_script} ${id})
# test_job_id=${RES##* }
# echo "test_job_id"" ""${test_job_id}"" -> ""${id}" >> out/job_logs.txt
