import phyre
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import os
import math 
import random

get_n_rollouts = 6
eval_setup = 'ball_cross_template'
fold_id = 0
total_frames = 17
rollouts = np.array([])
rollout_results = []
rollout_num = 1

train_tasks, dev_tasks, test_tasks = phyre.get_fold(eval_setup, fold_id)
action_tier = phyre.eval_setup_to_action_tier(eval_setup)

simulator = phyre.initialize_simulator(train_tasks, action_tier)
actions = simulator.build_discrete_action_space(max_actions=100)
tasks = train_tasks
print("Getting frames for", get_n_rollouts, "rollouts...")

for _ in range(get_n_rollouts):
    path = os.getcwd() + '/rollout_data/rollout_'+str(rollout_num)
    isdir = os.path.isdir(path)  
    if isdir == False:  os.mkdir('rollout_data/rollout_'+str(rollout_num))
    task_index =  np.random.choice(100)# The simulator takes an index into simulator.task_ids.
    action = random.choice(actions)

    simulation = simulator.simulate_action(task_index, action, need_images=True, need_featurized_objects=True)
    while simulation.images is None:
        action = random.choice(actions)
        simulation = simulator.simulate_action(task_index, action, need_images=True, need_featurized_objects=True)
    sequence = np.array([])

    # We can visualize the simulation at each timestep.
    for i, image in enumerate(simulation.images):

        img = phyre.observations_to_float_rgb(image)
        if len(sequence) == 0: sequence = img.reshape(1, 256, 256, 3)
        else: sequence = np.append(sequence, img.reshape(1, 256, 256, 3), axis=0)
        sequence_num = len(sequence)
        filename = 'rollout_data/rollout_' + str(rollout_num) + '/frame_' + str(sequence_num) + '.png'
        matplotlib.image.imsave(filename, img)
    
    if len(rollouts) == 0: 
        rollouts = sequence.reshape(1, 17, 256, 256, 3)
    else:
        rollouts = np.append(rollouts, sequence.reshape(1, 17, 256, 256, 3), axis=0)
    
    rollout_result = simulation.status.is_solved()
    rollout_results.append(rollout_result)
    rollout_num += 1