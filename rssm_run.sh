#!/bin/bash

job_script="dreamerv2/scripts/exp.sh"

# dataset supports 'moving_mnist', 'cater'
python_file="dreamerv2/dreamer.py"
dataset="moving_mnist"
batch_size=50
steps=1e5
exp_name="RSSM_""$dataset"
output_file="out/RSSM/""$exp_name""-%j.out"

sbatch --output $output_file --job-name $exp_name $job_script $batch_size $steps $dataset $python_file