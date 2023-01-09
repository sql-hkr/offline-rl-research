import warnings

from loguru import logger

warnings.filterwarnings("ignore")


from ireul.algo.modelfree import bc, cql, mcq, mcq_bc, mcq_iql, mcq_vaegan
from ireul.config.algo import (
    bc_config,
    cql_config,
    mcq_bc_config,
    mcq_config,
    mcq_iql_config,
    mcq_vaegan_config,
)
from ireul.utils.config import parse_config

algo_dict = {
    "bc": {"algo": bc, "config": bc_config},
    "cql": {"algo": cql, "config": cql_config},
    "mcq": {"algo": mcq, "config": mcq_config},
    "mcq_bc": {"algo": mcq_bc, "config": mcq_bc_config},
    "mcq_iql": {"algo": mcq_iql, "config": mcq_iql_config},
    "mcq_vaegan": {"algo": mcq_vaegan, "config": mcq_vaegan_config},
}


def algo_select(command_args, algo_config_module=None):
    algo_name = command_args["algo_name"]
    logger.info("Use {} algorithm!", algo_name)
    assert algo_name in algo_dict.keys()
    algo = algo_dict[algo_name]["algo"]

    if algo_config_module is None:
        algo_config_module = algo_dict[algo_name]["config"]
    algo_config = parse_config(algo_config_module)
    algo_config.update(command_args)

    algo_init = algo.algo_init
    algo_trainer = algo.AlgoTrainer

    return algo_init, algo_trainer, algo_config
