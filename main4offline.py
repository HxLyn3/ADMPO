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
    parser = argparse.ArgumentParser(description="Offline MBRL")

    # environment settings
    parser.add_argument("--env", type=str, default="d4rl")
    parser.add_argument("--env-name", type=str, default="hopper-medium-v2")

    # policy parameters
    parser.add_argument("--algo", type=str, default="admpo")
    parser.add_argument("--ac-hidden-dims", type=list, default=[256, 256])              # dimensions of actor/critic hidden layers
    parser.add_argument("--actor-freq", type=int, default=1)
    parser.add_argument("--actor-lr", type=float, default=1e-4)                         # learning rate of actor
    parser.add_argument("--lr-schedule", type=bool, default=True)
    parser.add_argument("--critic-lr", type=float, default=3e-4)                        # learning rate of critic
    parser.add_argument("--gamma", type=float, default=0.99)                            # discount factor
    parser.add_argument("--tau", type=float, default=0.005)                             # update rate of target network
    parser.add_argument("--alpha", type=float, default=0.1)                             # weight of entropy
    parser.add_argument("--auto-alpha", type=bool, default=True)                        # auto alpha adjustment
    parser.add_argument("--alpha-lr", type=float, default=1e-4)                         # learning rate of alpha
    parser.add_argument("--target-entropy", type=int, default=None)                     # target entropy
    parser.add_argument("--penalty-coef", type=float, default=1.0)                      # penalty coefficient
    parser.add_argument("--deterministic-backup", type=bool, default=False)
    parser.add_argument("--q-clip", type=float, default=None)

    # armpo parameters
    parser.add_argument("--max-arm-step", type=int, default=10)                          # maximum length of rnn input
    parser.add_argument("--arm-hidden-dim", type=int, default=200)

    # replay-buffer parameters
    parser.add_argument("--buffer-size", type=int, default=int(1e6))

    # dynamics-model parameters
    parser.add_argument("--model-lr", type=float, default=3e-4)
    parser.add_argument("--rollout-freq", type=int, default=1000)
    parser.add_argument("--rollout-batch-size", type=int, default=int(5e4))
    parser.add_argument("--rollout-length", type=int, default=1)
    parser.add_argument("--model-retain-epochs", type=int, default=5)
    parser.add_argument("--real-ratio", type=float, default=0.05)

    # running parameters
    parser.add_argument("--n-epochs", type=int, default=5000)
    parser.add_argument("--step-per-epoch", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=256)                          # mini-batch size
    parser.add_argument("--eval-n-episodes", type=int, default=10)
    parser.add_argument("--test-n-episodes", type=int, default=int(1e3))
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--seeds", type=int, nargs='*', default=list(range(10)))

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

        runner = TRAINER["offline"](copy.deepcopy(args))
        runner.run()

if __name__ == "__main__":
    main()
