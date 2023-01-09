import os
import random
import uuid

import numpy as np
import torch
import wandb
from loguru import logger

from ireul.utils.logger import log_path


def setup_seed(seed=1024):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def wandb_init(project, group, name) -> None:
    wandb.init(
        project=project,
        group=group,
        name=name,
        id=str(uuid.uuid4()),
    )
    # wandb.run.save()
