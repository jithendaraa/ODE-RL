import os
import numpy as np
import json
import matplotlib.pyplot as plt
import time
import datetime

def print_exp_details(opt, n_batches):
    print()
    print("Exp ID: ", opt.id)
    print(f"Logging to {opt.logdir}")
    
    if opt.phase == 'train':
        print("Training the", opt.model, "model on", opt.dataset, 'for', n_batches, 'batches of batch size', opt.batch_size)
        print("Input frames:", opt.train_in_seq)
        print("Output frames:", opt.train_out_seq)
        print("Training batches:", n_batches)

    else:
        print("Training the", opt.model, "model on", opt.dataset, 'for', n_batches, 'batches of batch size', opt.batch_size)
        print("Input frames:", opt.test_in_seq)
        print("Output frames:", opt.test_out_seq)
        print("Training batches:", n_batches)
    
    print()


