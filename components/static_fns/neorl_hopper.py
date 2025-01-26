import numpy as np

class StaticFns:

    @staticmethod
    def termination_fn(obs, act, next_obs):
        assert len(obs.shape) == len(next_obs.shape) == len(act.shape) == 2

        z = next_obs[:, 1:2]
        angle = next_obs[:, 2:3]
        state = next_obs[:, 3:]

        min_state, max_state = (-100.0, 100.0)
        min_z, max_z = (0.7, float('inf'))
        min_angle, max_angle = (-0.2, 0.2)

        healthy_state = np.all(np.logical_and(min_state < state, state < max_state), axis=-1, keepdims=True)
        healthy_z = np.logical_and(min_z < z, z < max_z)
        healthy_angle = np.logical_and(min_angle < angle, angle < max_angle)

        is_healthy = np.logical_and(np.logical_and(healthy_state, healthy_z), healthy_angle)

        done = np.logical_not(is_healthy).reshape(-1, 1)
        return done
