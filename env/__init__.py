import gym
import d4rl
import neorl
from .mujoco_env import make_mujoco_env

ENV = {
    "mujoco": make_mujoco_env,
    "d4rl": lambda env_name: gym.make(env_name),
    "neorl": lambda env_name: neorl.make(env_name)
}
