import gym
import d4rl
from .mujoco_env import make_mujoco_env

ENV = {
    "mujoco": make_mujoco_env,
    "d4rl": lambda env_name: gym.make(env_name),
    "neorl": None
}
