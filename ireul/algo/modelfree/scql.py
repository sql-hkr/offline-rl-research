import copy
import os

import numpy as np
import torch
import torch.nn.functional as F
from loguru import logger
from torch import nn, optim
from torch.distributions import Normal, kl_divergence

import wandb
from ireul.algo.base import BaseAlgo
from ireul.utils.exp import setup_seed
from ireul.utils.function import asymmetric_l2_loss, quantile_huber_loss
from ireul.utils.net.common import MLP, Net
from ireul.utils.net.continuous import Critic
from ireul.utils.net.tanhpolicy import TanhGaussianPolicy


def algo_init(args):
    logger.info("Run algo_init function")

    setup_seed(args["seed"])

    if args["obs_shape"] and args["action_shape"]:
        obs_shape, action_shape = args["obs_shape"], args["action_shape"]
    elif "task" in args.keys():
        from ireul.utils.env import get_env_action_range, get_env_shape

        obs_shape, action_shape = get_env_shape(args["task"])
        max_action, _ = get_env_action_range(args["task"])
        args["obs_shape"], args["action_shape"] = obs_shape, action_shape
    else:
        raise NotImplementedError

    net_a = Net(
        layer_num=args["layer_num"],
        state_shape=obs_shape,
        hidden_layer_size=args["hidden_layer_size"],
    )

    actor = TanhGaussianPolicy(
        preprocess_net=net_a,
        action_shape=action_shape,
        hidden_layer_size=args["hidden_layer_size"],
        conditioned_sigma=True,
    ).to(args["device"])

    actor_optim = optim.Adam(actor.parameters(), lr=args["actor_lr"])

    net_c1 = Net(
        layer_num=args["layer_num"],
        state_shape=obs_shape,
        action_shape=action_shape,
        concat=True,
        hidden_layer_size=args["hidden_layer_size"],
    )
    critic1 = Critic(
        preprocess_net=net_c1,
        hidden_layer_size=args["hidden_layer_size"],
    ).to(args["device"])
    critic1_optim = optim.Adam(critic1.parameters(), lr=args["critic_lr"])

    net_c2 = Net(
        layer_num=args["layer_num"],
        state_shape=obs_shape,
        action_shape=action_shape,
        concat=True,
        hidden_layer_size=args["hidden_layer_size"],
    )
    critic2 = Critic(
        preprocess_net=net_c2,
        hidden_layer_size=args["hidden_layer_size"],
    ).to(args["device"])
    critic2_optim = optim.Adam(critic2.parameters(), lr=args["critic_lr"])

    net_c2 = Net(
        layer_num=args["layer_num"],
        state_shape=obs_shape,
        concat=True,
        hidden_layer_size=args["hidden_layer_size"],
    )

    if args["use_automatic_entropy_tuning"]:
        if args["target_entropy"]:
            target_entropy = args["target_entropy"]
        else:
            target_entropy = -np.prod(args["action_shape"]).item()
        log_alpha = torch.zeros(1, requires_grad=True, device=args["device"])
        alpha_optimizer = optim.Adam(
            [log_alpha],
            lr=args["actor_lr"],
        )

    nets = {
        "actor": {"net": actor, "opt": actor_optim},
        "critic1": {"net": critic1, "opt": critic1_optim},
        "critic2": {"net": critic2, "opt": critic2_optim},
        "log_alpha": {
            "net": log_alpha,
            "opt": alpha_optimizer,
            "target_entropy": target_entropy,
        },
    }

    return nets


class AlgoTrainer(BaseAlgo):
    def __init__(self, algo_init, args):
        super(AlgoTrainer, self).__init__(args)
        self.args = args

        self.seed = args["seed"]
        self.device = self.args["device"]

        self.actor = algo_init["actor"]["net"]
        self.actor_opt = algo_init["actor"]["opt"]

        self.critic1 = algo_init["critic1"]["net"]
        self.critic1_opt = algo_init["critic1"]["opt"]
        self.critic2 = algo_init["critic2"]["net"]
        self.critic2_opt = algo_init["critic2"]["opt"]
        self.critic1_target = copy.deepcopy(self.critic1)
        self.critic2_target = copy.deepcopy(self.critic2)

        if args["use_automatic_entropy_tuning"]:
            self.log_alpha = algo_init["log_alpha"]["net"]
            self.alpha_opt = algo_init["log_alpha"]["opt"]
            self.target_entropy = algo_init["log_alpha"]["target_entropy"]

        self.critic_criterion = nn.MSELoss()

        self.lam = args["lam"]

        self._n_train_steps_total = 0
        self._current_epoch = 0

        from ireul.utils.env import get_env_action_range

        action_max, _ = get_env_action_range(args["task"])
        self.act_max = action_max
        self.task = args["task"]

    def forward(self, obs, reparameterize=True, return_log_prob=True):
        log_prob = None
        tanh_normal = self.actor(
            obs,
            reparameterize=reparameterize,
        )

        if return_log_prob:
            if reparameterize is True:
                action, pre_tanh_value = tanh_normal.rsample(
                    return_pretanh_value=True
                )
            else:
                action, pre_tanh_value = tanh_normal.sample(
                    return_pretanh_value=True
                )
            log_prob = tanh_normal.log_prob(
                action, pre_tanh_value=pre_tanh_value
            )
            log_prob = log_prob.sum(dim=1, keepdim=True)
        else:
            if reparameterize is True:
                action = tanh_normal.rsample()
            else:
                action = tanh_normal.sample()
        return action, log_prob

    def _train(self, batch, steps_per_epoch):
        self._current_epoch += 1
        batch = batch.to_torch(device=self.device)
        rewards = batch.rew
        terminals = batch.done
        obs = batch.obs
        actions = batch.act
        next_obs = batch.obs_next

        """
        Policy and Alpha Loss
        """
        new_obs_actions, log_pi = self.forward(obs)

        if self.args["use_automatic_entropy_tuning"]:
            alpha_loss = -(
                self.log_alpha * (log_pi + self.target_entropy).detach()
            ).mean()
            self.alpha_opt.zero_grad()
            alpha_loss.backward()
            self.alpha_opt.step()
            alpha = self.log_alpha.exp()
        else:
            alpha_loss = 0
            alpha = 1

        """
        QF Loss
        """
        q1_pred = self.critic1(obs, actions)
        q2_pred = self.critic2(obs, actions)

        new_next_actions, new_log_pi = self.forward(
            next_obs,
            reparameterize=True,
            return_log_prob=True,
        )

        target_q_values = torch.min(
            self.critic1_target(next_obs, new_next_actions),
            self.critic2_target(next_obs, new_next_actions),
        )

        target_q_values = target_q_values - alpha * new_log_pi
        q_target = (
            self.args["reward_scale"] * rewards
            + (1.0 - terminals)
            * self.args["discount"]
            * target_q_values.detach()
        )

        with torch.no_grad():
            _a, _ = self.forward(
                obs,
                reparameterize=True,
                return_log_prob=True,
            )

        qf1_ood_loss = self.critic1(obs, _a).mean()
        qf2_ood_loss = self.critic2(obs, _a).mean()

        qf1_loss = F.mse_loss(q1_pred, q_target) + self.lam * qf1_ood_loss
        qf2_loss = F.mse_loss(q2_pred, q_target) + self.lam * qf2_ood_loss

        """
        Update critic networks
        """
        self.critic1_opt.zero_grad()
        qf1_loss.backward(retain_graph=True)
        self.critic1_opt.step()

        self.critic2_opt.zero_grad()
        qf2_loss.backward()
        self.critic2_opt.step()

        q_new_actions = torch.min(
            self.critic1(obs, new_obs_actions),
            self.critic2(obs, new_obs_actions),
        )

        # TODO smooth (KL)
        # log_pi_noise = self.forward(obs+0.01*(torch.randn_like(obs)-0.5))
        policy_loss = (
            alpha * log_pi - q_new_actions
        ).mean()  # + F.kl_div(log_pi, log_pi_noise)
        self.actor_opt.zero_grad()
        policy_loss.backward()
        self.actor_opt.step()

        """
        Soft Updates target network
        """
        self._sync_weight(
            self.critic1_target, self.critic1, self.args["soft_target_tau"]
        )
        self._sync_weight(
            self.critic2_target, self.critic2, self.args["soft_target_tau"]
        )

        if self._current_epoch % int(steps_per_epoch / 2) == 0:
            wandb.log(
                {
                    "q1_pred": q1_pred.mean(),
                    "q2_pred": q2_pred.mean(),
                    "q_target": q_target.mean(),
                    "actor_q": q_new_actions.mean(),
                    "alpha": alpha,
                    "loss/q1": qf1_loss,
                    "loss/q2": qf2_loss,
                    "loss/alpha": alpha_loss,
                    "loss/policy": policy_loss,
                },
                step=self._current_epoch,
            )

        self._n_train_steps_total += 1

    def get_model(self):
        return self.actor

    def get_policy(self):
        return self.actor

    def train(self, train_buffer, val_buffer, callback_fn):

        for epoch in range(1, self.args["max_epoch"] + 1):
            for step in range(1, self.args["steps_per_epoch"] + 1):
                train_data = train_buffer.sample(self.args["batch_size"])
                self._train(train_data, self.args["steps_per_epoch"])

            res = callback_fn(self.get_policy())

            self.log_res(self._current_epoch, res)
            wandb.log(res, step=self._current_epoch)

        return self.get_policy()
