import os
import pickle

import d4rl
import gym
import numpy as np
from loguru import logger

from ireul.utils.data import SampleBatch
from ireul.utils.env import (
    compute_mean_std,
    get_env,
    modify_reward,
    normalize_states,
)


def load_d4rl_buffer(config):
    env = get_env(config["task"])
    dataset = d4rl.qlearning_dataset(env)
    if config["normalize_reward"]:
        modify_reward(dataset, config["task"])

    if config["normalize_obs"]:
        obs_mean, obs_std = compute_mean_std(dataset["observations"], eps=1e-3)
    else:
        obs_mean, obs_std = 0, 1

    dataset["observations"] = normalize_states(
        dataset["observations"], obs_mean, obs_std
    )
    dataset["next_observations"] = normalize_states(
        dataset["next_observations"], obs_mean, obs_std
    )

    buffer = SampleBatch(
        obs=dataset["observations"],
        obs_next=dataset["next_observations"],
        act=dataset["actions"],
        rew=np.expand_dims(np.squeeze(dataset["rewards"]), 1),
        done=np.expand_dims(np.squeeze(dataset["terminals"]), 1),
    )

    logger.info("obs shape: {}", buffer.obs.shape)
    logger.info("obs_next shape: {}", buffer.obs_next.shape)
    logger.info("act shape: {}", buffer.act.shape)
    logger.info("rew shape: {}", buffer.rew.shape)
    logger.info("done shape: {}", buffer.done.shape)
    logger.info("Episode reward: {}", buffer.rew.sum() / np.sum(buffer.done))
    logger.info("Number of terminals on: {}", np.sum(buffer.done))
    return buffer, obs_mean, obs_std
