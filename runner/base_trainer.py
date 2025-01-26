import os
import json
import datetime
import numpy as np
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

from env import ENV

class BASETrainer:
    """ base trainer """

    def __init__(self, args):
        # init env
        self.env = ENV[args.env](args.env_name)
        self.env.action_space.seed(args.seed)

        self.eval_env = ENV[args.env](args.env_name)
        self.eval_env.action_space.seed(args.seed)

        if args.env == "adroit" or args.env == "maze":
            self.env.seed(args.seed)
            self.eval_env.seed(args.seed)
        else:
            self.env.reset(seed=args.seed)
            self.eval_env.reset(seed=args.seed)

        args.obs_shape = self.env.observation_space.shape
        args.action_space = self.env.action_space
        args.action_dim = np.prod(args.action_space.shape)

        # running parameters
        self.batch_size = args.batch_size
        self.eval_n_episodes = args.eval_n_episodes
        self.device = args.device
        self.seed = args.seed
        self.args = args

        dtime = datetime.datetime.now().strftime("%y-%m%d-%H%M%S")
        if args.env == "neorl": args.env_name += f"-{args.data_type}"
        self.model_dir = "./result/{}/{}/{}/{}/model".format(args.env, args.env_name, args.algo, dtime)
        self.record_dir = "./result/{}/{}/{}/{}/record".format(args.env, args.env_name, args.algo, dtime)
        self.log_dir = "./result/{}/{}/{}/{}/log".format(args.env, args.env_name, args.algo, dtime)
        os.makedirs(self.model_dir, exist_ok=True)
        os.makedirs(self.record_dir, exist_ok=True)
        os.makedirs(self.log_dir, exist_ok=True)

        self.logger = SummaryWriter(self.log_dir)

    def _warm_up(self):
        """ randomly sample a lot of transitions into buffer before starting learning """
        obs, _ = self.env.reset()

        # step for {self.start_learning} time-steps
        pbar = tqdm(range(self.start_learning), desc="Warming up")
        for _ in pbar:
            action = self.env.action_space.sample()
            next_obs, reward, terminated, truncated, _ = self.env.step(action)
            self.memory.store(obs, action, reward, next_obs, terminated, truncated)

            obs = next_obs
            if terminated or truncated: obs, _ = self.env.reset()

        return obs

    def _eval_policy(self):
        """ evaluate policy """
        episode_rewards = []
        for _ in range(self.eval_n_episodes):
            done = False
            episode_rewards.append(0)
            obs, _ = self.eval_env.reset()
            while not done:
                action, _ = self.agent.act(obs, deterministic=True)
                obs, reward, terminated, truncated, _ = self.eval_env.step(action)
                done = terminated or truncated
                episode_rewards[-1] += reward
        return episode_rewards

    def _save(self, records):
        """ save model and record """
        self.agent.save_model(os.path.join(self.model_dir, "agent_seed-{}.pth".format(self.seed)))
        with open(os.path.join(self.record_dir, "record_seed-{}.txt".format(self.seed)), "w") as f:
            json.dump(records, f)
