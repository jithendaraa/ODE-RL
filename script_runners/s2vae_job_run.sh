#!/bin/bash
dataset=$1
train=$2
model=$3
time=$4
mode=${5:-'extrap'}
start=`date +%s`

config=${train}'_'${dataset}'_'${model}
echo ${config}
echo ""

if [ ${train} == 'train' ]
then
    if [ ${dataset} == 'mmnist' ]
    then
        if [ ${model} == 's2vae' ]
        then
            # Train S2VAE on MMNIST
            output_file="out/S2VAE/""$config""-%j.out"
            echo "Train S2VAE MMNIST: ${config}"
        elif [ ${model} == 'cs2vae' ]
        then
            # Train CS2VAE on MMNIST
            output_file="out/CS2VAE/""$config""-%j.out"
            echo "Train CS2VAE MMNIST: ${config}"
        fi
    fi
fi

command="sbatch --job-name ${config} --output ${output_file} --time ${time} scripts/run_s2vae.sh ${config}"      
echo ${command}
echo ""

RES=$(${command})
job_id=${RES##* }
echo "Job ID"" ""${job_id}"" -> ""${config}" >> out/job_logs.txt
echo "Job ID"" ""${job_id}"" -> ""${config}" 
echo ""

end=`date +%s`
runtime=$((end-start))
echo "Program time: $runtime"