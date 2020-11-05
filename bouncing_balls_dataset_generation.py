"""
Taken from https://github.com/AlbertoCenzato/dnn_bouncing_balls/blob/master/dataset_generation.py
This script comes from the RTRBM code by Ilya Sutskever from 
http://www.cs.utoronto.ca/~ilya/code/2008/RTRBM.tar
Modified by Alberto Cenzato, 2019
"""

import os
import argparse

import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


# redefine numpy functions as function_std
shape_std = np.shape
size_std  = np.size


# --------------------- constants ----------------------------
SIZE = 10            # size of bounding box: SIZE X SIZE.
logdir = './sample'  # make sure you have this folder


def shape(A):
    if isinstance(A, np.ndarray):
        return shape_std(A)
    else:
        return A.shape()


def size(A):
    if isinstance(A, np.ndarray):
        return size_std(A)
    else:
        return A.size()


def new_speeds(m1, m2, v1, v2):
    new_v2 = (2*m1*v1 + v2*(m2-m1))/(m1+m2)
    new_v1 = new_v2 + (v2 - v1)
    return new_v1, new_v2
    

def norm(x):
    return np.sqrt((x**2).sum())


def sigmoid(x):
    return 1. / (1. + np.exp(-x))


def bounce_n(T=128, n=2, r=None, m=None):
    if r is None:
        r = np.array([1.2]*n)
    if m is None:
        m = np.array([1]*n)
    
    # r is to be rather small.
    X = np.zeros((T, n, 2), dtype='float')
    v = np.random.randn(n, 2)
    v = v / norm(v)*.5
    good_config = False
    while not good_config:
        x = 2 + np.random.rand(n, 2) * 8
        good_config = True
        for i in range(n):
            for z in range(2):
                if x[i][z] - r[i] < 0:    good_config = False
                if x[i][z] + r[i] > SIZE: good_config = False

        # that's the main part.
        for i in range(n):
            for j in range(i):
                if norm(x[i] - x[j]) < r[i] + r[j]:
                    good_config = False
    
    eps = .5
    for t in range(T):
        # for how long do we show small simulation

        for i in range(n):
            X[t, i] = x[i]
            
        for mu in range(int(1/eps)):
            for i in range(n):
                x[i] += eps*v[i]

            for i in range(n):
                for z in range(2):
                    if x[i][z] - r[i] < 0:    v[i][z] =  abs(v[i][z])  # want positive
                    if x[i][z] + r[i] > SIZE: v[i][z] = -abs(v[i][z])  # want negative

            for i in range(n):
                for j in range(i):
                    if norm(x[i] - x[j]) < r[i] + r[j]:
                        # the bouncing off part:
                        w = x[i] - x[j]
                        w = w / norm(w)

                        v_i = np.dot(w.transpose(), v[i])
                        v_j = np.dot(w.transpose(), v[j])

                        new_v_i, new_v_j = new_speeds(m[i], m[j], v_i, v_j)
                        
                        v[i] += w*(new_v_i - v_i)
                        v[j] += w*(new_v_j - v_j)

    return X


def ar(x, y, z):
    return z / 2 + np.arange(x, y, z, dtype='float')


def matricize(X, res, r=None):
    T, n = shape(X)[0:2]
    if r is None: r = np.array([1.2]*n)

    A = np.zeros((T, res, res), dtype='float')
    
    [I, J] = np.meshgrid(ar(0, 1, 1./res)*SIZE, ar(0, 1, 1./res)*SIZE)

    for t in range(T):
        for i in range(n):
            A[t] += np.exp(-( ((I - X[t, i, 0])**2 + (J - X[t, i, 1])**2) / (r[i]**2))**4)
            
        A[t][A[t] > 1] = 1
        
    return A


def bounce_mat(res, n=2, T=128, r=None):
    if r is None: 
        r = np.array([1.2]*n)
        
    x = bounce_n(T, n, r)
    A = matricize(x, res, r)
    
    return A


def bounce_vec(res, n=2, T=128, r=None, m=None):
    if r is None: 
        r = np.array([1.2]*n)
        
    x = bounce_n(T, n, r, m)
    V = matricize(x, res, r)
    
    return V.reshape(T, res**2)


def show_sample(V):
    T   = len(V)
    res = int(np.sqrt(shape(V)[1]))
    for t in range(T):
        plt.imshow(V[t].reshape(res, res), cmap=matplotlib.cm.Greys_r) 
        # Save it
        fname = logdir+'/'+str(t)+'.png'
        plt.savefig(fname)      


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Trains and/or tests bouncing ball trajectory prediction models.')
    parser.add_argument('--res',     default=60,   type=int, help='image resolution (only square aspect ratios allowed)')
    parser.add_argument('--frames',  default=40,   type=int, help='sequence length')
    parser.add_argument('--samples', default=6000, type=int, help='number of sequences to generate')

    args = parser.parse_args()

    res = args.res
    T   = args.frames
    N   = args.samples

    digits = len(str(N))
    file_name_template = 'sequence_{:0' + str(digits) + 'd}.npy'

    data_path = './data'
    training_path   = os.path.join(data_path, 'training')
    validation_path = os.path.join(data_path, 'validation')
    testing_path    = os.path.join(data_path, 'testing')

    if not os.path.exists(data_path):       os.mkdir(data_path)
    if not os.path.exists(training_path):   os.mkdir(training_path)
    if not os.path.exists(validation_path): os.mkdir(validation_path)
    if not os.path.exists(testing_path):    os.mkdir(testing_path)

    # training set generation
    for i in range(N):
        if i % 200 == 0:
            print('Training set sample {}'.format(i))
        sequence = bounce_vec(res=res, n=3, T=T)
        file_path = os.path.join(training_path, file_name_template.format(i))
        np.save(file_path, sequence.astype(np.float32))

    # validation set generation
    for i in range(int(0.2*N)):
        if i % 200 == 0:
            print('Validation set sample {}'.format(i))
        sequence = bounce_vec(res=res, n=3, T=T)
        file_path = os.path.join(validation_path, file_name_template.format(i))
        np.save(file_path, sequence.astype(np.float32))

    # testing set generation
    for i in range(int(0.2*N)):
        if i % 200 == 0:
            print('Testing set sample {}'.format(i))
        sequence = bounce_vec(res=res, n=3, T=T)
        file_path = os.path.join(testing_path, file_name_template.format(i))
        np.save(file_path, sequence.astype(np.float32))