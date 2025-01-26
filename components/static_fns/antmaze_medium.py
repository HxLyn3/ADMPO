import numpy as np

class StaticFns:

    @staticmethod
    def termination_fn(obs, act, next_obs):
        center = np.array((20.75, 20.75))
        radius = 0.25

        assert len(obs.shape) == len(next_obs.shape) == len(act.shape) == 2
        done = ((next_obs[:, :2] - center[None, :]) ** 2).sum(axis=1) < radius * radius
        done = done[:, None]
        return done
