import gymnasium as gym

MBPO_ENVIRONMENT_SPECS = (
	{
        "id": "AntTruncatedObs-v3",
        "entry_point": (f"env.ant:AntTruncatedObsEnv"),
        "max_episode_steps": 1000
    },

	{
        "id": "HumanoidTruncatedObs-v3",
        "entry_point": (f"env.humanoid:HumanoidTruncatedObsEnv"),
        "max_episode_steps": 1000
    },
)

# register XxxTruncatedObs
for mbpo_environment in MBPO_ENVIRONMENT_SPECS:
    gym.register(**mbpo_environment)

make_mujoco_env = lambda env_name: gym.make(env_name)
