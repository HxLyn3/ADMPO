# ADMPO: Any-step Dynamics Model for Policy Optimization

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://github.com/HxLyn3/ADMPO/blob/main/LICENSE)

This is the code for the paper [Any-step Dynamics Model Improves Future Predictions for Online and Offline Reinforcement Learning](https://openreview.net/forum?id=JZCxlrwjZ8) in ICLR 2025.

## Requirements

To install all the required dependencies:

1. Install MuJoCo engine, which can be downloaded from [here](https://mujoco.org/download).
2. Install Python packages listed in `requirements.txt` using `pip install -r requirements.txt`. You should specify the version of `mujoco-py` in `requirements.txt` depending on the version of MuJoCo engine you have installed.
3. Manually download and install `d4rl` package from [here](https://github.com/rail-berkeley/d4rl).
4. Manually download and install `neorl` package from [here](https://github.com/polixir/NeoRL).

## Run an experiment 

### Online Setting

```shell
python main4online.py --env-name [Env name] 
```

The config files act as defaults for a task. They are all located in `config`. `--env-name` refers to the config files in `config/` including Hopper-v3, Walker2d-v3, AntTruncatedObs-v3, and HumanoidTruncatedObs-v3. All results will be stored in the `result` folder.

For example, run ADMPO-ON on Hopper:

```bash
python main4online.py --env-name Hopper-v3
```

### Offline Setting

```shell
python main4offline.py --env [Env] --env-name [Env name] 
```

The config files act as defaults for a task. They are all located in `config`. `--env` refers to the benchmark, D4RL or NeoRL. `--env-name` refers to the config files in `config/`. All results will be stored in the `result` folder.

For example, run ADMPO-OFF on hopper-medium-v2 dataset of D4RL benchmark:

```bash
python main4offline.py --env d4rl --env-name hopper-medium-v2
```

## Citation
If you find this repository useful for your research, please cite:
```bash
@inproceedings{
    admpo,
    author       = {Haoxin Lin and
                    Yu{-}Yan Xu and
                    Yihao Sun and
                    Zhilong Zhang and
                    Yi{-}Chen Li and
                    Chengxing Jia and
                    Junyin Ye and
                    Jiaji Zhang and
                    Yang Yu},
    title        = {Any-step Dynamics Model Improves Future Predictions for Online and Offline Reinforcement Learning},
    booktitle    = {The 13th International Conference on Learning Representations (ICLR'25)},
    year         = {2025},
    address      = {Singapore}
}
```
