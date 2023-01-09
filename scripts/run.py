import subprocess

import fire

version = "v2"

envs = ["halfcheetah", "walker2d", "hopper"]
levels = ["random", "medium", "medium-replay", "medium-expert", "expert"]


def run(algo):
    for seed in [0, 1, 2]:
        for env in envs:
            for level in levels:
                subprocess.run(
                    [
                        "python",
                        "train_d4rl.py",
                        "--algo_name",
                        algo,
                        "--task",
                        f"{env}-{level}-{version}",
                        "--seed",
                        seed,
                    ]
                )


if __name__ == "__main__":
    fire.Fire(run)
