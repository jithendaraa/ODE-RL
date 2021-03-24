import phyre
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import os
import math 
import random
import argparse
import cv2
import torch

parser = argparse.ArgumentParser()
parser.add_argument("-r", "--rollouts", help="get n rollouts of PHYRE plays", default=1600, type=int)
args = parser.parse_args()

get_n_rollouts = args.rollouts
eval_setup = 'ball_cross_template'
fold_id = 0
rollouts = np.array([])
rollout_results = []
rollout_num = 1
total_frames = 17
height, width = 64, 64 # Phyre is 256, 256; we reduce it to 64, 64
channels = 3

train_tasks, dev_tasks, test_tasks = phyre.get_fold(eval_setup, fold_id)
action_tier = phyre.eval_setup_to_action_tier(eval_setup)

simulator = phyre.initialize_simulator(train_tasks, action_tier)
actions = simulator.build_discrete_action_space(max_actions=200)
tasks = train_tasks
print("Getting frames for", get_n_rollouts, "rollouts...", len(train_tasks))

for _ in range(get_n_rollouts):
    path = os.getcwd() + '/rollout_data/rollout_'+str(rollout_num)
    isdir = os.path.isdir(path)  
    if isdir == False:  os.mkdir('rollout_data/rollout_'+str(rollout_num))
    task_index = np.random.choice(len(train_tasks))# The simulator takes an index into simulator.task_ids.
    action = random.choice(actions)

    simulation = simulator.simulate_action(task_index, action, need_images=True, need_featurized_objects=True)
    while simulation.images is None:
        action = random.choice(actions)
        simulation = simulator.simulate_action(task_index, action, need_images=True, need_featurized_objects=True)
    sequence = np.array([])

    for i, image in enumerate(simulation.images):
        img = phyre.observations_to_float_rgb(image)
        img = torch.tensor(cv2.resize(img, (height, width), interpolation=cv2.INTER_LINEAR).transpose(2, 0, 1), dtype=torch.float32).permute(1, 2, 0).numpy()
        if len(sequence) == 0: sequence = img.reshape(1, height, width, channels)
        else: sequence = np.append(sequence, img.reshape(1, height, width, channels), axis=0)
        sequence_num = len(sequence)
        filename = 'rollout_data/rollout_' + str(rollout_num) + '/frame_' + str(sequence_num) + '.jpg'
        matplotlib.image.imsave(filename, img)
    
    if len(sequence) == 0: print("Error: Sequence is 0 frames long!")
    while sequence.shape[0] < total_frames:
        sequence = np.append(sequence, sequence[len(sequence)-1].reshape(1, height, width, channels), axis=0)
        sequence_num = len(sequence)
        filename = 'rollout_data/rollout_' + str(rollout_num) + '/frame_' + str(sequence_num) + '.jpg'
        matplotlib.image.imsave(filename, img)

    if len(sequence) != 17: print("Error", len(sequence))
    if len(rollouts) == 0:  rollouts = sequence.reshape(1, total_frames, height, width, channels)
    else:                   rollouts = np.append(rollouts, sequence.reshape(1, total_frames, height, width, channels), axis=0)
    
    rollout_result = simulation.status.is_solved()
    rollout_results.append(rollout_result)
    rollout_num += 1

    if _ % 10 == 0:
        print(_, "rollouts generated...")

np.save('rollout_results.npy', rollout_results)
