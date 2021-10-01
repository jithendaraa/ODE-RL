#!/bin/bash
exp_id=$1   # 0 for OG S3VAE, 1 for S3VAE, 2 for Conv. S3VAE
dataset=$2  # ['mmnist']
train=${3-'train'}
time=$4'23:00:00'

if [ ${exp_id} == 0 ]    
then
    # Run OG S3VAE
    sh script_runners/og_s3vae_job_run.sh ${dataset} ${train} ${time}
elif [ ${exp_id} == 1 ]
then 
    # Run my S3VAE
    sh script_runners/s3vae_job_run.sh ${dataset} ${train} 's3vae' ${time}
elif [ ${exp_id} == 2 ]
then 
    # Run Conv S3VAE
    sh script_runners/s3vae_job_run.sh ${dataset} ${train} 'cs3vae' ${time}
elif [ ${exp_id} == 3 ]
then 
    # Run S4VAE
    sh script_runners/s3vae_job_run.sh ${dataset} ${train} 's4vae' ${time}
elif [ ${exp_id} == 4 ]
then 
    # Run Conv S4VAE
    sh script_runners/s3vae_job_run.sh ${dataset} ${train} 'cs4vae' ${time}
fi
