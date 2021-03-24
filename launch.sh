#!/bin/sh

# Takes the gpu with the most memory available and set it as the only visible device
# Probably too simple, but it is really difficult to define a better strategy in a multi-user environment without exclusivity on the GPUs

gpu=`nvidia-smi --query-gpu=memory.free,index --format=csv,noheader,nounits | sort -nr | sed "s/^[0-9]+,[ \t]*//" -r | head -1`
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=$gpu
echo $gpu

export MKL_NUM_THREADS=1 # For Anaconda MKL thing
"$@"
