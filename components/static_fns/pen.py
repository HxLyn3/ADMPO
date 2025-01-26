import numpy as np

class StaticFns:

    @staticmethod
    def termination_fn(obs, act, next_obs):
        assert len(obs.shape) == len(next_obs.shape) == len(act.shape) == 2

        obj_pos = next_obs[:, 24:27]
        done = obj_pos[:, 2] < 0.075

        done = done[:,None]
        return done
