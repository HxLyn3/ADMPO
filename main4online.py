import os
import copy
import yaml
import random
import argparse
import setproctitle

import torch
import numpy as np

from runner import TRAINER

def get_args():
    parser = argparse.ArgumentParser(description="Online MBRL")

    # environment settings
    parser.add_argument("--env", type=str, default="mujoco")
    # mujoco choices: ["InvertedPendulum-v2", "Hopper-v3", "Swimmer-v3", "Walker2d-v3", "HalfCheetah-v3", "AntTruncatedObs-v3", "HumanoidTruncatedObs-v3"]
    parser.add_argument("--env-name", type=str, default="AntTruncatedObs-v3")

    # policy parameters
    parser.add_argument("--algo", type=str, default="admpo")
    parser.add_argument("--ac-hidden-dims", type=list, default=[256, 256])              # dimensions of actor/critic hidden layers
    parser.add_argument("--actor-freq", type=int, default=1)
    parser.add_argument("--actor-lr", type=float, default=3e-4)                         # learning rate of actor
    parser.add_argument("--critic-lr", type=float, default=3e-4)                        # learning rate of critic
    parser.add_argument("--gamma", type=float, default=0.99)                            # discount factor
    parser.add_argument("--tau", type=float, default=0.005)                             # update rate of target network
    # (for sac)
    parser.add_argument("--alpha", type=float, default=0.2)                             # weight of entropy
    parser.add_argument("--auto-alpha", type=bool, default=True)                        # auto alpha adjustment
    parser.add_argument("--alpha-lr", type=float, default=3e-4)                         # learning rate of alpha
    parser.add_argument("--target-entropy", type=int, default=-1)                       # target entropy
    parser.add_argument("--penalty-coef", type=float, default=0.0)                      # penalty coefficient

    # armpo parameters
    parser.add_argument("--max-arm-step", type=int, default=10)                          # maximum length of rnn input
    parser.add_argument("--arm-hidden-dim", type=int, default=200)

    # replay-buffer parameters
    parser.add_argument("--buffer-size", type=int, default=int(1e6))

    # dynamics-model parameters
    parser.add_argument("--model-lr", type=float, default=3e-4)
    parser.add_argument("--rollout-batch-size", type=int, default=int(1e5))
    parser.add_argument("--rollout-schedule", type=int, nargs='*', default=[int(1e4), int(1e5), 1, 20])
    parser.add_argument("--model-update-interval", type=int, default=250)
    parser.add_argument("--model-rollout-interval", type=int, default=250)
    parser.add_argument("--model-retain-steps", type=int, default=1000)
    parser.add_argument("--real-ratio", type=float, default=0.05)

    # running parameters
    parser.add_argument("--n-steps", type=int, default=int(1e5))
    parser.add_argument("--start-learning", type=int, default=int(5e3))
    parser.add_argument("--update-interval", type=int, default=1)
    # UTD
    parser.add_argument("--updates-per-step", type=int, default=20)                     # only use for model-based algos
    parser.add_argument("--batch-size", type=int, default=256)                          # mini-batch size
    parser.add_argument("--eval-interval", type=int, default=int(1e3))
    parser.add_argument("--eval-n-episodes", type=int, default=10)
    parser.add_argument("--test-n-episodes", type=int, default=int(1e3))
    parser.add_argument("--save-interval", type=int, default=int(1e4))
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--seeds", type=int, nargs='*', default=list(range(2, 3)))

    args = parser.parse_args()
    return args

def main():
    """ main function """
    args = get_args()
    algo_yml_path = "./config/{}/{}.yml".format(args.env, args.env_name.split("-v")[0])
    algo_yml = yaml.load(open(algo_yml_path, 'r'), Loader=yaml.FullLoader)
    for key, value in algo_yml.items():
        setattr(args, key, value)

    setproctitle.setproctitle("{} {}".format(args.algo.upper(), args.env_name))

    for seed in args.seeds:
        args.seed = seed
        random.seed(args.seed)
        np.random.seed(args.seed)
        os.environ["PYTHONHASHSEED"] = str(args.seed)

        # set seed of torch
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        runner = TRAINER["online"](copy.deepcopy(args))
        runner.run()

if __name__ == "__main__":
    main()
