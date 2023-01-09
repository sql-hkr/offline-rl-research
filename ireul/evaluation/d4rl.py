from collections import OrderedDict

import d4rl
import gym
import numpy as np
import torch
from d4rl.infos import REF_MAX_SCORE, REF_MIN_SCORE
from tqdm import tqdm


def d4rl_score(task, rew_mean, len_mean):
    score = (
        (rew_mean - REF_MIN_SCORE[task])
        / (REF_MAX_SCORE[task] - REF_MIN_SCORE[task])
        * 100
    )

    return score


def d4rl_eval_fn(env, eval_episodes=100):
    def d4rl_eval(policy):
        episode_rewards = []
        episode_lengths = []
        for _ in range(eval_episodes):
            state, done = env.reset(), False
            rewards = 0
            lengths = 0
            while not done:
                state = state[np.newaxis]
                action = policy.get_action(state).reshape(-1)
                state, reward, done, _ = env.step(action)
                rewards += reward
                lengths += 1

            episode_rewards.append(rewards)
            episode_lengths.append(lengths)

        len_mean = np.mean(episode_lengths)

        scores = env.get_normalized_score(np.array(episode_rewards)) * 100

        res = OrderedDict()
        res["eval/d4rl_score"] = np.mean(scores)
        res["eval/d4rl_score_std"] = np.std(scores)
        res["eval/length_mean"] = len_mean

        return res

    return d4rl_eval
