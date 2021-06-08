#!/bin/bash
mkdir -p ~/scratch/datasets/MovingMNIST
cd ~/scratch/datasets/MovingMNIST
wget http://www.cs.toronto.edu/~nitish/unsupervised_video/mnist_test_seq.npy
mkdir train
mkdir test
cd ~/projects/rrg-ebrahimi/jithen/ODE-RL/data_gen_scripts
python train_test_split_moving_mnist.py