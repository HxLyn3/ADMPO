import numpy as np

class ReplayBuffer:
    """ general replay buffer """

    def __init__(self, buffer_size, obs_shape, action_dim):
        self.obs_shape = obs_shape
        self.action_dim = action_dim
        self.memory = {
            "s":       np.zeros((buffer_size, *self.obs_shape), dtype=np.float32),
            "a":       np.zeros((buffer_size, self.action_dim), dtype=np.float32),
            "r":       np.zeros((buffer_size, 1), dtype=np.float32),
            "s_":      np.zeros((buffer_size, *self.obs_shape), dtype=np.float32),
            "done":    np.zeros((buffer_size, 1), dtype=np.float32),
            "timeout": np.zeros((buffer_size, 1), dtype=np.float32)
        }

        self.capacity = buffer_size
        self.reset()

    def reset(self):
        self.size = 0
        self.cnt = 0

        # not used
        self.cur_epi_start = 0

    def store(self, s, a, r, s_, done, timeout):
        """ store transition (s, a, r, s_, done, timeout) """
        self.memory["s"][self.cnt] = s
        self.memory["a"][self.cnt] = a
        self.memory["r"][self.cnt] = r
        self.memory["s_"][self.cnt] = s_
        self.memory["done"][self.cnt] = done
        self.memory["timeout"][self.cnt] = timeout

        self.cnt = (self.cnt+1)%self.capacity
        self.size = min(self.size+1, self.capacity)

    def store_batch(self, s, a, r, s_, done, timeout):
        """ store batch transitions (s, a, r, s_, done, timeout) """
        batch_size = len(s)

        indices = np.arange(self.cnt, self.cnt+batch_size)%self.capacity
        self.memory["s"][indices] = s
        self.memory["a"][indices] = a
        self.memory["r"][indices] = r
        self.memory["s_"][indices] = s_
        self.memory["done"][indices] = done
        self.memory["timeout"][indices] = timeout

        self.cnt = (self.cnt+batch_size)%self.capacity
        self.size = min(self.size+batch_size, self.capacity)

    def load_dataset(self, dataset, max_episode_step):
        """ load dataset """
        have_next_obs = "next_observations" in dataset.keys()
        use_timeout = "timeouts" in dataset.keys()

        N = dataset["rewards"].shape[0]
        episode_step = 0
        for i in range(N-1):
            obs = dataset["observations"][i].astype(np.float32)
            if have_next_obs:
                next_obs = dataset["next_observations"][i].astype(np.float32)
            else:
                next_obs = dataset["observations"][i+1].astype(np.float32)
            action = dataset["actions"][i].astype(np.float32)
            reward = dataset["rewards"][i].astype(np.float32)
            done = bool(dataset["terminals"][i])

            if use_timeout:
                timeout = dataset["timeouts"][i]
            else:
                timeout = (episode_step == max_episode_step - 1)
            if done or timeout:
                episode_step = -1
                if not have_next_obs:
                    episode_step = 0
                    self.cur_epi_start = self.cnt
                    continue

            self.store(obs, action, reward, next_obs, done, timeout)
            episode_step += 1

    def cal_mu_std(self):
        """ calculate mean and std of obs and action """
        obs_mu = np.mean(self.memory["s"][:self.size], axis=0)
        obs_std = np.std(self.memory["s"][:self.size], axis=0)
        obs_std[obs_std < 1e-12] = 1.0
        act_mu = np.mean(self.memory["a"][:self.size], axis=0)
        act_std = np.std(self.memory["a"][:self.size], axis=0)
        act_std[act_std < 1e-12] = 1.0
        return obs_mu, obs_std, act_mu, act_std

    def sample(self, batch_size):
        """ sample a batch of data """
        indices = np.random.randint(0, self.size, batch_size)
        return {var: self.memory[var][indices] for var in self.memory.keys()}

    def sample_all(self):
        """ sample all data """
        indices = np.arange(self.size)
        return {var: self.memory[var][indices] for var in self.memory.keys()}
