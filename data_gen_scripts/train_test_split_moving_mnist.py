"""
    Splits the (20, 10000, 64, 64) np array at ~/scratch/datasets/MovingMNIST/mnist_test_seq.npy into
    (16, 10000, 64, 64) and (4, 10000, 64, 64) and saves as ~/scratch/datasets/MovingMNIST/train/mmnist_train_seq.npy and ~/scratch/datasets/MovingMNIST/train/mmnist_test_seq.npy
"""

import numpy as np
import os

train_split = 0.8
sequences = 20
mmnist_data = np.load("/home/jithen/scratch/datasets/MovingMNIST/mnist_test_seq.npy")
print(mmnist_data.shape)
seq_idx = np.arange(0, sequences)
np.random.shuffle(seq_idx)

train_idx = int(train_split * sequences)
train_seq_idx = seq_idx[:train_idx]
test_seq_idx = seq_idx[train_idx:]

train_seq, test_seq = [], []

for train_idx in train_seq_idx:
    train_seq.append(mmnist_data[train_idx])

for test_idx in test_seq_idx:
    test_seq.append(mmnist_data[test_idx])

train_seq = np.array(train_seq)
test_seq = np.array(test_seq)

np.save("/home/jithen/scratch/datasets/MovingMNIST/train/mmnist_train_seq.npy", train_seq)
np.save("/home/jithen/scratch/datasets/MovingMNIST/test/mmnist_test_seq.npy", test_seq)



