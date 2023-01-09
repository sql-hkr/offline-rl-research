import torch

task = "Hopper-v3"
task_data_type = "low"
task_train_num = 99

seed = 42

device = "cuda" if torch.cuda.is_available() else "cpu"
obs_shape = None
act_shape = None
max_action = None

actor_features = 256
actor_layers = 2

batch_size = 256
steps_per_epoch = 1000
max_epoch = 100

actor_lr = 1e-3

normalize_obs = True
normalize_reward = False
reward_scale = 1
