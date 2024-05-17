import numpy as np
from gymnasium.spaces import Box
from gymnasium.envs.mujoco.humanoid_v3 import HumanoidEnv

class HumanoidTruncatedObsEnv(HumanoidEnv):
    """
        COM inertia (cinert), COM velocity (cvel), actuator forces (qfrc_actuator), 
        and external forces (cfrc_ext) are removed from the observation.
    """
    def __init__(
        self,
        xml_file="humanoid.xml",
        forward_reward_weight=1.25,
        ctrl_cost_weight=0.1,
        contact_cost_weight=5e-7,
        contact_cost_range=(-np.inf, 10.0),
        healthy_reward=5.0,
        terminate_when_unhealthy=True,
        healthy_z_range=(1.0, 2.0),
        reset_noise_scale=1e-2,
        exclude_current_positions_from_observation=True,
    ):
        super(HumanoidTruncatedObsEnv, self).__init__(
            xml_file,
            forward_reward_weight,
            ctrl_cost_weight,
            contact_cost_weight,
            contact_cost_range,
            healthy_reward,
            terminate_when_unhealthy,
            healthy_z_range,
            reset_noise_scale,
            exclude_current_positions_from_observation
        )
        
        if exclude_current_positions_from_observation:
            self.observation_space = Box(
                low=-np.inf, high=np.inf, shape=(45,), dtype=np.float64
            )
        else:
            self.observation_space = Box(
                low=-np.inf, high=np.inf, shape=(47,), dtype=np.float64
            )

    def _get_obs(self):
        position = self.sim.data.qpos.flat.copy()
        velocity = self.sim.data.qvel.flat.copy()

        # com_inertia = self.sim.data.cinert.flat.copy()
        # com_velocity = self.sim.data.cvel.flat.copy()

        # actuator_forces = self.sim.data.qfrc_actuator.flat.copy()
        # external_contact_forces = self.sim.data.cfrc_ext.flat.copy()

        if self._exclude_current_positions_from_observation:
            position = position[2:]

        return np.concatenate(
            (
                position,
                velocity,
                # com_inertia,
                # com_velocity,
                # actuator_forces,
                # external_contact_forces,
            )
        )
