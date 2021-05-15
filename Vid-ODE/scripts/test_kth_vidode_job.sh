#!/bin/bash
#SBATCH --account=def-ebrahimi
#SBATCH --mail-user=jithen.subra@gmail.com
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
#SBATCH --mail-type=REQUEUE
#SBATCH --mail-type=ALL
#SBATCH --time=1:00:00
#SBATCH --gres=gpu:v100:1
#SBATCH --mem=32G

start=`date +%s`
test_dir=$1
echo "Starting run at: `date`"
echo "python main.py --test_dir ${test_dir} -d kth --extrap -p test_met"
python main.py --test_dir $test_dir -d kth --extrap -p test_met
echo "Ending run at: `date`"
end=`date +%s`
runtime=$((end-start))
echo "Program time: $runtime"