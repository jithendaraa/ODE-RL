#!/bin/bash
exp_id=$1   # 0 for OG S3VAE, 1 for S3VAE, 2 for Conv. S3VAE
dataset=${2:-'mmnist'}  # ['mmnist']
train=${3:-'train'}
def_time='23:00:00'
time=${4:-$def_time}

if [ ${exp_id} == '0' ]    
then
    # Run OG S3VAE
    bash script_runners/og_s3vae_job_run.sh ${dataset} ${train} ${time}
elif [ ${exp_id} == '1' ]
then 
    # Run my S3VAE (recon)
    bash script_runners/s3vae_job_run.sh ${dataset} ${train} 's3vae' ${time}
elif [ ${exp_id} == '1e' ]
then 
    # Run my S3VAE (extrap)
    bash script_runners/s3vae_job_run.sh ${dataset} ${train} 's3vae' ${time} 'extrap'
elif [ ${exp_id} == '2' ]
then 
    # Run Conv S3VAE (recon)
    bash script_runners/s3vae_job_run.sh ${dataset} ${train} 'cs3vae' ${time}
elif [ ${exp_id} == '2e' ]
then 
    # Run Conv S3VAE (extrap)
    bash script_runners/s3vae_job_run.sh ${dataset} ${train} 'cs3vae' ${time} 'extrap'
elif [ ${exp_id} == '3' ]
then 
    # Run S4VAE (recon)
    bash script_runners/s3vae_job_run.sh ${dataset} ${train} 's4vae' ${time}
elif [ ${exp_id} == '3e' ]
then 
    # Run S4VAE (extrap)
    bash script_runners/s3vae_job_run.sh ${dataset} ${train} 's4vae' ${time} 'extrap'
elif [ ${exp_id} == '4' ]
then 
    # Run Conv S4VAE (recon)
    bash script_runners/s3vae_job_run.sh ${dataset} ${train} 'cs4vae' ${time}
elif [ ${exp_id} == '4e' ]
then 
    # Run Conv S4VAE (extrap)
    bash script_runners/s3vae_job_run.sh ${dataset} ${train} 'cs4vae' ${time} 'extrap'
elif [ ${exp_id} == '5' ]
then 
    # Run S2VAE (extrap)
    bash script_runners/s2vae_job_run.sh ${dataset} ${train} 's2vae' ${time} 'extrap'
elif [ ${exp_id} == '6' ]
then 
    # Run CS2VAE (extrap)
    bash script_runners/s2vae_job_run.sh ${dataset} ${train} 'cs2vae' ${time} 'extrap'
fi


