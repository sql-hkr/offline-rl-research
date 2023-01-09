# オフライン強化学習

[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Imports: isort](https://img.shields.io/badge/%20imports-isort-%231674b1?style=flat&labelColor=ef8336)](https://pycqa.github.io/isort/)

- 機械学習ライブラリ：[PyTorch: An Imperative Style, High-Performance Deep Learning Library](https://arxiv.org/abs/1912.01703)
- デバッグツール：[Weights & Biases](https://wandb.ai/site)

## 環境構築

```bash
git clone https://github.com/sql-hkr/offline-rl
cd offline-rl
python setup.py develop
```

### [Docker]([Docker](https://docs.docker.com/))

Dockerイメージ：[Dockerfile](Dockerfile)

```bash
git clone https://github.com/sql-hkr/offline-rl
cd offline-rl
docker build -t offline-rl .
docker run --gpus all -it -v $PWD:/workspace offline-rl
```

## 実行

```bash
python train_d4rl.py --algo_name={algo} --task {env} --seed 0 --lam 0.9
```

ex.

```bash
python train_d4rl.py --algo_name=mcq --task walker2d-medium-v2 --seed 6 --lam 0.9
```
