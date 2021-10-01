#!/bin/bash
#SBATCH --time=12:00:00
#SBATCH --account=def-ebrahimi
#SBATCH --mem=32G
#SBATCH --gres=gpu:v100:1
#SBATCH --job-name=OG_S3VAE_Sprite
#SBATCH --cpus-per-task=6
#SBATCH --mail-user=jithen.subra@gmail.com
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
#SBATCH --mail-type=REQUEUE
#SBATCH --mail-type=ALL
#SBATCH --output=out/OG_S3VAE/OG_s3vae-%j.out

dataset=$1
train=$2
time=$3
start=`date +%s`

# echo "Starting run at: `date`"
source ~/ENV/bin/activate
output_file='out/OG_S3VAE/'$2'_og_s3vae_'$dataset'-%j.out'
config=${train}'_og_s3vae'

command='sbatch --job-name '${train}'_og_s3vae --output '${output_file}' --time '${time}' scripts/run_og_s3vae.sh '${dataset}' '${train}
echo $command

RES=$(${command})
job_id=${RES##* }
echo ""
echo "Job ID"" ""${job_id}"" -> ""${config}" >> out/job_logs.txt
echo "Job ID"" ""${job_id}"" -> ""${config}" 

echo ""
# echo "Ending run at: `date`"
end=`date +%s`
runtime=$((end-start))
echo "Program time: $runtime"