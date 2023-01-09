import json
import os
import uuid
from abc import ABC, abstractmethod
from collections import OrderedDict

import torch
from loguru import logger

import wandb
from ireul.utils.exp import wandb_init
from ireul.utils.io import create_dir, download_helper, read_json
from ireul.utils.logger import log_path


class BaseAlgo(ABC):
    def __init__(self, args):
        logger.info("Init AlgoTrainer")

        wandb_init(
            project="offline-rl",
            group=args["algo_name"] + "-" + args["task"],
            name=args["algo_name"] + "-seed:" + str(args["seed"]),
        )

    def log_res(self, epoch, result):
        logger.info("Epoch : {}", epoch)
        for k, v in result.items():
            logger.info("{} : {}", k, v)

    @abstractmethod
    def train(
        self,
        history_buffer,
        eval_fn=None,
    ):
        raise NotImplementedError

    def _sync_weight(self, net_target, net, soft_target_tau=5e-3):
        for o, n in zip(net_target.parameters(), net.parameters()):
            o.data.copy_(
                o.data * (1.0 - soft_target_tau) + n.data * soft_target_tau
            )

    @abstractmethod
    def get_policy(
        self,
    ):
        raise NotImplementedError

    # @abstractmethod
    def save_model(self, model_path):
        torch.save(self.get_policy(), model_path)

    # @abstractmethod
    def load_model(self, model_path):
        model = torch.load(model_path)

        return model
