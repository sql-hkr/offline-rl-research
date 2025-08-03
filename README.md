# オフライン強化学習

[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Imports: isort](https://img.shields.io/badge/%20imports-isort-%231674b1?style=flat&labelColor=ef8336)](https://pycqa.github.io/isort/)

## 主要ライブラリ

- 機械学習ライブラリ：[PyTorch: An Imperative Style, High-Performance Deep Learning Library](https://arxiv.org/abs/1912.01703)
- デバッグツール：[Weights & Biases](https://wandb.ai/site)

### 強化学習ベンチマーク環境
- [mujoco-py](https://github.com/openai/mujoco-py)
- [Gym](https://github.com/openai/gym)
- [D4RL](https://github.com/Farama-Foundation/D4RL)

> [!IMPORTANT]
> Dockerで環境構築しない場合は，[MuJoCo](https://github.com/google-deepmind/mujoco)のインストールが必要です．

## 環境構築

```bash
git clone https://github.com/sql-hkr/offline-rl
cd offline-rl
python setup.py develop
```

### [Docker]([Docker](https://docs.docker.com/))で環境構築

Dockerをインストールした上で以下を実行してください．

```bash
git clone https://github.com/sql-hkr/offline-rl
cd offline-rl
docker build -t offline-rl .
docker run --gpus all -it -v $PWD:/workspace offline-rl
```

- Dockerイメージ：[Dockerfile](Dockerfile)

> [!NOTE]
> Dockerを用いた環境構築を推奨します．

## 実行手順
下記４項目を入力の上，実行してください．
- アルゴリズム名：`--algo_name`
- タスク：`--task`
- 乱数シード：`--seed`
- ペナルティ項の反映率：`--lam`

```bash
python train_d4rl.py --algo_name={algo} --task {env} --seed 0 --lam 0.1
```

> [!TIP]
> SCQLアルゴリズムを用いてタスク：walker2d-medium-v2を実行する場合
> ```bash
> python train_d4rl.py --algo_name=scql --task walker2d-medium-v2 --seed 0 --lam 0.1
> ```
