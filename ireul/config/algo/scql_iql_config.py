import torch

task = "d4rl-hopper-medium-replay-v2"
seed = 25

device = "cuda" if torch.cuda.is_available() else "cpu"
obs_shape = None
act_shape = None
max_action = None

max_epoch = 1000
steps_per_epoch = 1000
number_of_runs = 10

normalize_obs = True
normalize_reward = False

batch_size = 256
hidden_layer_size = 400
layer_num = 2
actor_lr = 3e-4
critic_lr = 3e-4
reward_scale = 1
use_automatic_entropy_tuning = True
target_entropy = None
discount = 0.99
soft_target_tau = 5e-3

lam = 0.1

# IQL
# MuJoCo locomotion 0.7
# Ant Maze 0.9
iql_tau = 0.7
