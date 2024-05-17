import numpy as np
from gymnasium.spaces import Box
from gymnasium.envs.mujoco.ant_v3 import AntEnv

DEFAULT_CAMERA_CONFIG = {
    "distance": 4.0,
}

class AntTruncatedObsEnv(AntEnv):
    """ External forces (sim.data.cfrc_ext) are removed from the observation """

    def __init__(
        self,
        xml_file="ant.xml",
        ctrl_cost_weight=0.5,
        contact_cost_weight=5e-4,
        healthy_reward=1.0,
        terminate_when_unhealthy=True,
        healthy_z_range=(0.2, 1.0),
        contact_force_range=(-1.0, 1.0),
        reset_noise_scale=0.1,
        exclude_current_positions_from_observation=True,
    ):
        super(AntTruncatedObsEnv, self).__init__(
            xml_file,
            ctrl_cost_weight,
            contact_cost_weight,
            healthy_reward,
            terminate_when_unhealthy,
            healthy_z_range,
            contact_force_range,
            reset_noise_scale,
            exclude_current_positions_from_observation
        )

        if exclude_current_positions_from_observation:
            self.observation_space = Box(
                low=-np.inf, high=np.inf, shape=(27,), dtype=np.float64
            )
        else:
            self.observation_space = Box(
                low=-np.inf, high=np.inf, shape=(29,), dtype=np.float64
            )

    def _get_obs(self):
        position = self.sim.data.qpos.flat.copy()
        velocity = self.sim.data.qvel.flat.copy()
        # contact_force = self.contact_forces.flat.copy()

        if self._exclude_current_positions_from_observation:
            position = position[2:]

        # observations = np.concatenate((position, velocity, contact_force))
        observations = np.concatenate((position, velocity))

        return observations
