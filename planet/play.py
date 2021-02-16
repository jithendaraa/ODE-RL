import os
from math import inf
import numpy as np
import argparse
from tqdm import tqdm
from env import CONTROL_SUITE_ENVS, Env, GYM_ENVS, PHYRE_ENVS, EnvBatcher
from memory import ExperienceReplay
from models import bottle, Encoder, ObservationModel, RewardModel, TransitionModel
from planner import MPCPlanner

import torch
from torch import nn, optim
from torch.distributions import Normal
from torch.distributions.kl import kl_divergence
from torch.nn import functional as F

def arg_parser():
  # Hyperparameters
  parser = argparse.ArgumentParser(description='PlaNet')
  parser.add_argument('--id', type=str, default='default', help='Experiment ID')
  parser.add_argument('--gpu', type=int, default=os.system('./launch.sh'), help='GPU number')
  parser.add_argument('--seed', type=int, default=1, metavar='S', help='Random seed')
  parser.add_argument('--disable-cuda', action='store_true', help='Disable CUDA')
  parser.add_argument('--env', type=str, default='Pendulum-v0', choices=GYM_ENVS + CONTROL_SUITE_ENVS + PHYRE_ENVS, help='Gym/Control Suite environment')
  parser.add_argument('--symbolic-env', action='store_true', help='Symbolic features')
  parser.add_argument('--max-episode-length', type=int, default=17, metavar='T', help='Max episode length')
  parser.add_argument('--experience-size', type=int, default=1000000, metavar='D', help='Experience replay size')  # Original implementation has an unlimited buffer size, but 1 million is the max experience collected anyway
  parser.add_argument('--activation-function', type=str, default='relu', choices=dir(F), help='Model activation function')
  parser.add_argument('--embedding-size', type=int, default=1024, metavar='E', help='Observation embedding size')  # Note that the default encoder for visual observations outputs a 1024D vector; for other embedding sizes an additional fully-connected layer is used
  parser.add_argument('--hidden-size', type=int, default=200, metavar='H', help='Hidden size')
  parser.add_argument('--belief-size', type=int, default=200, metavar='H', help='Belief/hidden size')
  parser.add_argument('--state-size', type=int, default=30, metavar='Z', help='State/latent size')
  parser.add_argument('--action-repeat', type=int, default=1, metavar='R', help='Action repeat')
  parser.add_argument('--action-noise', type=float, default=0.3, metavar='ε', help='Action noise')
  parser.add_argument('--episodes', type=int, default=2000, metavar='E', help='Total number of episodes')
  parser.add_argument('--seed-episodes', type=int, default=100, metavar='S', help='Seed episodes')
  parser.add_argument('--collect-interval', type=int, default=1000, metavar='C', help='Collect interval')
  parser.add_argument('--batch-size', type=int, default=25, metavar='B', help='Batch size')
  parser.add_argument('--chunk-size', type=int, default=17, metavar='L', help='Chunk size')
  parser.add_argument('--overshooting-distance', type=int, default=50, metavar='D', help='Latent overshooting distance/latent overshooting weight for t = 1')
  parser.add_argument('--overshooting-kl-beta', type=float, default=0, metavar='β>1', help='Latent overshooting KL weight for t > 1 (0 to disable)')
  parser.add_argument('--overshooting-reward-scale', type=float, default=0, metavar='R>1', help='Latent overshooting reward prediction weight for t > 1 (0 to disable)')
  parser.add_argument('--global-kl-beta', type=float, default=0, metavar='βg', help='Global KL weight (0 to disable)')
  parser.add_argument('--free-nats', type=float, default=3, metavar='F', help='Free nats')
  parser.add_argument('--bit-depth', type=int, default=5, metavar='B', help='Image bit depth (quantisation)')
  parser.add_argument('--learning-rate', type=float, default=3e-4, metavar='α', help='Learning rate') 
  parser.add_argument('--learning-rate-schedule', type=int, default=0, metavar='αS', help='Linear learning rate schedule (optimisation steps from 0 to final learning rate; 0 to disable)') 
  parser.add_argument('--adam-epsilon', type=float, default=1e-4, metavar='ε', help='Adam optimiser epsilon value') 
  # Note that original has a linear learning rate decay, but it seems unlikely that this makes a significant difference
  parser.add_argument('--grad-clip-norm', type=float, default=1000, metavar='C', help='Gradient clipping norm')
  parser.add_argument('--planning-horizon', type=int, default=17, metavar='H', help='Planning horizon distance')
  parser.add_argument('--optimisation-iters', type=int, default=10, metavar='I', help='Planning optimisation iterations')
  parser.add_argument('--candidates', type=int, default=1000, metavar='J', help='Candidate samples per iteration')
  parser.add_argument('--top-candidates', type=int, default=100, metavar='K', help='Number of top candidates to fit')
  parser.add_argument('--test', action='store_true', help='Test only')
  parser.add_argument('--test-interval', type=int, default=5, metavar='I', help='Test interval (episodes)')
  parser.add_argument('--test-episodes', type=int, default=100, metavar='E', help='Number of test episodes')
  parser.add_argument('--checkpoint-interval', type=int, default=10, metavar='I', help='Checkpoint interval (episodes)')
  parser.add_argument('--checkpoint-experience', action='store_true', help='Checkpoint experience replay')
  parser.add_argument('--models', type=str, default='', metavar='M', help='Load model checkpoint')
  parser.add_argument('--experience-replay', type=str, default='', metavar='ER', help='Load experience replay')
  parser.add_argument('--render', action='store_true', help='Render environment')
  parser.add_argument('--cross-class', action='store_true', help='Cross class or within class')
  args = parser.parse_args()
  args.overshooting_distance = min(args.chunk_size, args.overshooting_distance)  # Overshooting distance cannot be greater than chunk size
  
  return args

def update_belief_and_act(args, env, planner, transition_model, encoder, belief, posterior_state, action, observation, min_action=-inf, max_action=inf, explore=False):
  # Infer belief over current state q(s_t|o≤t,a<t) from the history
  belief, _, _, _, posterior_state, _, _ = transition_model(posterior_state, action.unsqueeze(dim=0), belief, encoder(observation).unsqueeze(dim=0))  # Action and observation need extra time dimension
  belief, posterior_state = belief.squeeze(dim=0), posterior_state.squeeze(dim=0)  # Remove time dimension from belief/state
  action = planner(belief, posterior_state)  # Get action from planner(q(s_t|o≤t,a<t), p)
  if explore:
    action = action + args.action_noise * torch.randn_like(action)  # Add exploration noise ε ~ p(ε) to the action
  action.clamp_(min=min_action, max=max_action)  # Clip action range
  next_observation, reward, done = env.step(action.cpu() if isinstance(env, EnvBatcher) else action[0].cpu())  # Perform environment step (action repeats handled internally)
  return belief, posterior_state, action, next_observation, reward, done


# Setup
args = arg_parser()
max_tasks_per_template = 100
phyre_env = True if args.env[:5] == 'phyre' else False
results_dir = os.path.join('results', args.id)
os.makedirs(results_dir, exist_ok=True)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
if torch.cuda.is_available() and not args.disable_cuda:
  args.device = torch.device('cuda:'+str(args.gpu))
  torch.cuda.manual_seed(args.seed)
else:
  args.device = torch.device('cpu')
metrics = {'steps': [], 'episodes': [], 'train_rewards': [], 'test_episodes': [], 'test_rewards': [], 'observation_loss': [], 'reward_loss': [], 'kl_loss': []}


env = Env(args.env, args.symbolic_env, args.seed, args.max_episode_length, args.action_repeat, args.bit_depth, episodes=args.episodes, within_class=False if args.cross_class else True)

transition_model = TransitionModel(args.belief_size, args.state_size, env.action_size, args.hidden_size, args.embedding_size, args.activation_function).to(device=args.device)
observation_model = ObservationModel(args.symbolic_env, env.observation_size, args.belief_size, args.state_size, args.embedding_size, args.activation_function, phyre_env=phyre_env).to(device=args.device)
reward_model = RewardModel(args.belief_size, args.state_size, args.hidden_size, args.activation_function).to(device=args.device)
encoder = Encoder(args.symbolic_env, env.observation_size, args.embedding_size, args.activation_function, phyre_env=phyre_env).to(device=args.device)

param_list = list(transition_model.parameters()) + list(observation_model.parameters()) + list(reward_model.parameters()) + list(encoder.parameters())
optimiser = optim.Adam(param_list, lr=0 if args.learning_rate_schedule != 0 else args.learning_rate, eps=args.adam_epsilon)

if args.models is not '' and os.path.exists(args.models):
  print("Loading models.....")
  model_dicts = torch.load(args.models)
  transition_model.load_state_dict(model_dicts['transition_model'])
  observation_model.load_state_dict(model_dicts['observation_model'])
  reward_model.load_state_dict(model_dicts['reward_model'])
  encoder.load_state_dict(model_dicts['encoder'])
  optimiser.load_state_dict(model_dicts['optimiser'])

planner = MPCPlanner(env.action_size, args.planning_horizon, args.optimisation_iters, args.candidates, args.top_candidates, transition_model, reward_model, env.action_range[0], env.action_range[1])
global_prior = Normal(torch.zeros(args.batch_size, args.state_size, device=args.device), torch.ones(args.batch_size, args.state_size, device=args.device))  # Global prior N(0, I)
free_nats = torch.full((1, ), args.free_nats, dtype=torch.float32, device=args.device)  # Allowed deviation in KL divergence

transition_model.eval()
reward_model.eval()
encoder.eval()
rewards = []
with torch.no_grad():
  total_reward = 0
  for _ in tqdm(range(args.test_episodes)):
    r = 0
    observation = env.reset()
    belief, posterior_state, action = torch.zeros(1, args.belief_size, device=args.device), torch.zeros(1, args.state_size, device=args.device), torch.zeros(1, env.action_size, device=args.device)
    pbar = tqdm(range(args.max_episode_length // args.action_repeat))
    for t in pbar:
      belief, posterior_state, action, observation, reward, done = update_belief_and_act(args, env, planner, transition_model, encoder, belief, posterior_state, action, observation.to(device=args.device), env.action_range[0], env.action_range[1])
      observation_model(belief, posterior_state)
      total_reward += reward
      r += reward
      if args.render:
        env.render()
      if done:
        pbar.close()
        rewards.append(r)
        break
print(rewards)
print('Average Reward:', total_reward / args.test_episodes)
env.close()
quit()