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
parser.add_argument("-r", "--rollouts", help="#Train rollouts for PHYRE", default=100, type=int)
parser.add_argument("-tr", "--test_rollouts", help="#Test rollouts for PHYRE", default=50, type=int)
parser.add_argument("-f", "--folder_name", help="Folder to store phyre data in", default='dataset/phyre/train', type=str)
parser.add_argument('-st', "--single_template", action='store_true', default=False)
args = parser.parse_args()

folder_name = args.folder_name
get_n_rollouts = args.rollouts
eval_setup = 'ball_cross_template'
fold_id = 0
rollouts = np.array([])
rollout_results = []
rollout_num = 1
total_frames = 17
height, width = 64, 64 # Phyre is 256, 256; we reduce it to 64, 64
channels = 3
test_rollouts = args.test_rollouts

train_tasks, dev_tasks, test_tasks = phyre.get_fold(eval_setup, fold_id)
action_tier = phyre.eval_setup_to_action_tier(eval_setup)

simulator = phyre.initialize_simulator(train_tasks, action_tier)
actions = simulator.build_discrete_action_space(max_actions=200)
tasks = train_tasks
print("Getting frames for", get_n_rollouts, "rollouts...", len(train_tasks))

for _ in range(get_n_rollouts+test_rollouts):
    path = os.getcwd() + '/' + folder_name + '/rollout_'+str(rollout_num)
    if args.single_template:
        task_index = 3
    else:
        task_index = np.random.choice(len(train_tasks))# The simulator takes an index into simulator.task_ids.
    action = random.choice(actions)

    simulation = simulator.simulate_action(task_index, action, need_images=True, need_featurized_objects=True)
    while simulation.images is None:
        action = random.choice(actions)
        simulation = simulator.simulate_action(task_index, action, need_images=True, need_featurized_objects=True)
    sequence = np.array([])

    for i, image in enumerate(simulation.images):
        img = phyre.observations_to_float_rgb(image)
        img = torch.tensor(cv2.resize(img, (height, width), interpolation=cv2.INTER_LINEAR).transpose(2, 0, 1), dtype=torch.float32).permute(1, 2, 0).numpy()*255.0
        if len(sequence) == 0: sequence = img.reshape(1, height, width, channels)
        else: sequence = np.append(sequence, img.reshape(1, height, width, channels), axis=0)
        sequence_num = len(sequence)
    
    if len(sequence) == 0: print("Error: Sequence is 0 frames long!")
    while sequence.shape[0] < total_frames:
        sequence = np.append(sequence, sequence[len(sequence)-1].reshape(1, height, width, channels), axis=0)
        sequence_num = len(sequence)

    if len(sequence) != 17: print("Error", len(sequence))
    
    rollout_result = simulation.status.is_solved()
    rollout_results.append(rollout_result)
    if rollout_num < get_n_rollouts:
        np.save('dataset/phyre/train/rollout_'+str(rollout_num)+'.npy', sequence)
    else: 
        np.save('dataset/phyre/test/rollout_'+str(1+rollout_num-get_n_rollouts)+'.npy', sequence)

    if _ % 10 == 0:
        print(_+10, "rollouts generated...")
    rollout_num += 1

np.save('rollout_results.npy', rollout_results)
