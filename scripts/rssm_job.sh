#!/bin/bash
dataset=$1
batch_size=$2
steps=$3
output_file=$4
jobname="RSSM-""$dataset"




sbatch --output ${output_file} --job-name ${jobname} ${job_script} ${batch_size} ${steps} ${dataset} ${python_file}