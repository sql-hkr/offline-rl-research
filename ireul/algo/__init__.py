import warnings

from loguru import logger

warnings.filterwarnings("ignore")


from ireul.algo.modelfree import (
    bc,
    cql,
    mcq,
    mcq_bc,
    mcq_iql,
    mcq_vaegan,
    scql,
    scql_d,
    scql_gan,
    scql_iql,
)
from ireul.config.algo import (
    bc_config,
    cql_config,
    mcq_bc_config,
    mcq_config,
    mcq_iql_config,
    mcq_vaegan_config,
    scql_config,
    scql_d_config,
    scql_gan_config,
    scql_iql_config,
)
from ireul.utils.config import parse_config

algo_dict = {
    "bc": {"algo": bc, "config": bc_config},
    "cql": {"algo": cql, "config": cql_config},
    "scql": {"algo": scql, "config": scql_config},
    "scql_iql": {"algo": scql_iql, "config": scql_iql_config},
    "scql_gan": {"algo": scql_gan, "config": scql_gan_config},
    "scql_d": {"algo": scql_d, "config": scql_d_config},
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
