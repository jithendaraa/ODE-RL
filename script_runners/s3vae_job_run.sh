#!/bin/bash

# 1. Train S3VAE on Moving MNIST
dataset=$1
train=$2
model=$3
time=$4

start=`date +%s`

if [ ${train} == 'train' ]
then
    if [ ${dataset} == 'mmnist' ]
    then
        if [ ${model} == 's3vae' ]
        then
            # Train S3VAE on MMNIST
            config='train_mmnist_recon_s3vae_def'
            output_file="out/S3VAE/""$config""-%j.out"
            echo "Train S3VAE MMNIST: ${config}"
        elif [ ${model} == 'cs3vae' ]
        then
            # Train Conv. S3VAE on MMNIST
            config='train_mmnist_recon_s3vaecgru'
            output_file="out/ConvS3VAE/""$config""-%j.out"
            echo "Train Conv S3VAE MMNIST: ${config}"
        elif [ ${model} == 's4vae' ]
        then
            # Train S4VAE on MMNIST
            config='train_mmnist_recon_s3vae_def_sa'
            output_file="out/S4VAE/""$config""-%j.out"
            echo "Train S4VAE MMNIST: ${config}"
        elif [ ${model} == 'cs4vae' ]
        then
            # Train Conv S4VAE on MMNIST
            config='train_mmnist_recon_s3vaecgru_sa'
            output_file="out/ConvS4VAE/""$config""-%j.out"
            echo "Train Conv S4VAE MMNIST: ${config}"
        else    
            echo "Not implemented model ${model}" 
        fi
    else
        echo "Not implemented dataset ${dataset}" 
    fi

elif [ ${train} == 'test' ] 
then
    if [ ${dataset} == 'mmnist' ]
    then
        if [ ${model} == 's3vae' ]
        then
            # Test S3VAE on MMNIST
            config='test_mmnist_recon_s3vae_def'
            output_file="out/S3VAE/""$config""-%j.out"
            echo "Test S3VAE MMNIST: ${config}"
        elif [ ${model} == 'cs3vae' ]
        then
            # Test Conv. S3VAE on MMNIST
            config='test_mmnist_recon_s3vaecgru'
            output_file="out/ConvS3VAE/""$config""-%j.out"
            echo "Test Conv S3VAE MMNIST: ${config}"
        elif [ ${model} == 's4vae' ]
        then
            # Test S4VAE on MMNIST
            config='test_mmnist_recon_s3vae_def_sa'
            output_file="out/S4VAE/""$config""-%j.out"
            echo "Test S4VAE MMNIST: ${config}"
        elif [ ${model} == 'cs4vae' ]
        then
            # Test Conv S4VAE on MMNIST
            config='test_mmnist_recon_s3vaecgru_sa'
            output_file="out/ConvS4VAE/""$config""-%j.out"
            echo "Test Conv S4VAE MMNIST: ${config}"
        else    
            echo "Not implemented model ${model}" 
        fi
    else
        echo "Not implemented dataset ${dataset}" 
    fi
    
else
    echo "Wrong command!"
fi

echo "sbatch --job-name ${config} --output ${output_file} --time ${time} scripts/run_s3vae.sh ${config}"      
echo ""

RES=$(sbatch --job-name ${config} --output ${output_file} --time ${time} scripts/run_job.sh ${config})
job_id=${RES##* }
echo "Job ID"" ""${job_id}"" -> ""${config}" >> out/job_logs.txt
echo "Job ID"" ""${job_id}"" -> ""${config}" 
echo ""

end=`date +%s`
runtime=$((end-start))
echo "Program time: $runtime"

