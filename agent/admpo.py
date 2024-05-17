import copy
import torch
import numpy as np
from tqdm import tqdm

from .sac import SACAgent
from dynamics import ARMDynamics

class ADMPOAgent(SACAgent):
    """ Any-step RNN-based Dynamics Model for Policy Optimization """

    def __init__(
        self, 
        obs_shape, 
        hidden_dims, 
        action_dim,
        action_space,
        static_fn,
        max_arm_step,
        arm_hidden_dim,
        actor_freq,
        actor_lr,
        critic_lr,
        model_lr,
        tau=0.005, 
        gamma=0.99, 
        alpha=0.2,
        auto_alpha=True,
        alpha_lr=3e-4,
        target_entropy=-1,
        penalty_coef=1,
        deterministic_backup=False,
        device="cuda:0"
    ):
        super().__init__(obs_shape, hidden_dims, action_dim, action_space, actor_freq, actor_lr, critic_lr, 
            tau, gamma, alpha, auto_alpha, alpha_lr, target_entropy, deterministic_backup, device)
        self.penalty_coef = penalty_coef

        self.static_fn = static_fn
        self.max_arm_step = max_arm_step
        self.make_dynamics = lambda: ARMDynamics(
            obs_dim=np.prod(obs_shape),
            action_dim=action_dim,
            hidden_dim=arm_hidden_dim,
            device=device
        )
        self.model_lr = model_lr
        self.reset_dyna()

    def reset_dyna(self):
        self.dynamics = self.make_dynamics()
        self.dyna_optim = torch.optim.Adam(self.dynamics.parameters(), lr=self.model_lr)
        self.train()

    def train(self):
        super().train()
        self.dynamics.train()

    def eval(self):
        super().eval()
        self.dynamics.eval()

    def rollout(self, init_seq_transitions, rollout_len):
        """ rollout in dynamics model """
        init_len = init_seq_transitions["s"].shape[1]

        transitions = {}
        for key in init_seq_transitions.keys():
            transitions[key] = copy.deepcopy(init_seq_transitions[key])
        transitions["mask"] = np.ones(transitions["timeout"].shape[:2], dtype=bool)

        obs = init_seq_transitions["s_"][:, -1]
        mask = np.ones(obs.shape[0], dtype=bool)
        for t in range(rollout_len):
            # make decision
            action, _ = self.act(obs)
            transitions["s"] = np.concatenate((transitions["s"], obs[:, None]), axis=1)
            transitions["a"] = np.concatenate((transitions["a"], action[:, None]), axis=1)

            # predict next state (any steps as input)
            max_step = min(transitions["s"].shape[1], self.max_arm_step)
            k = np.random.randint(max_step) + 1
            input_s = transitions["s"][:, -k]
            input_a = transitions["a"][:, -k:]
            next_obs, reward = self.dynamics.step(input_s, input_a)
            done = self.static_fn.termination_fn(transitions["s"][:, -1], transitions["a"][:, -1], next_obs)
            timeout = np.zeros(done.shape, dtype=bool)

            # uncertainty
            if self.penalty_coef != 0:
                next_obs_means = []
                for k in range(1, max_step+1):
                    input_s = transitions["s"][:, -k]
                    input_a = transitions["a"][:, -k:]
                    next_obs_mean, _ = self.dynamics.dstep(input_s, input_a)
                    next_obs_means.append(next_obs_mean)
                next_obs_means = np.stack(next_obs_means, axis=0)
                penalty = np.sqrt(next_obs_means.var(axis=0).mean(axis=-1, keepdims=True))
                reward -= self.penalty_coef * penalty
            
            # store
            transitions["r"] = np.concatenate((transitions["r"], reward[:, None]), axis=1)
            transitions["s_"] = np.concatenate((transitions["s_"], next_obs[:, None]), axis=1)
            transitions["done"] = np.concatenate((transitions["done"], done[:, None]), axis=1)
            transitions["timeout"] = np.concatenate((transitions["timeout"], timeout[:, None]), axis=1)
            transitions["mask"] = np.concatenate((transitions["mask"], mask[:, None]), axis=1)

            # to next step
            mask[done.flatten()] = False
            if mask.sum() == 0: break
            obs = next_obs

        # mask terminated steps
        for key in transitions.keys():
            if key != "mask": 
                transitions[key] = transitions[key][:, init_len:].reshape(-1, transitions[key].shape[-1])
                transitions[key] = transitions[key][transitions["mask"][:, init_len:].flatten()]

        transitions.pop("mask")
        return transitions

    def learn_dynamics_from(self, buffer, batch_size, max_holdout=1000, min_epochs=1):
        """ learn any-step rnn-based dynamics model """
        self.reset_dyna()

        # set mean and std
        obs_mu, obs_std, act_mu, act_std = buffer.cal_mu_std()
        self.dynamics.set_mu_std(obs_mu, obs_std, act_mu, act_std)
        saved_dynamics = copy.deepcopy(self.dynamics)

        data_size = buffer.size
        holdout_size = min(int(data_size * 0.2), max_holdout)
        train_size = data_size - holdout_size

        epoch = 0
        holdout_losses = [1e10]*self.max_arm_step
        cnt = 0

        while True:
            epoch += 1

            pbar = tqdm(range(train_size//batch_size*self.max_arm_step), desc=f"[M][Epoch {epoch} @ ARM Dynamics Model Training]")
            for _ in pbar:
                # sample any-step data
                k = np.random.randint(self.max_arm_step) + 1
                any_step_seq = buffer.sample_nstep(batch_size, k, end_idx=train_size)
                s = any_step_seq["s"][:, 0]
                a_seq = any_step_seq["a"]
                r = any_step_seq["r"][:, -1]
                s_ = any_step_seq["s_"][:, -1]
                trgt = np.concatenate((s_-s, r), axis=-1)

                # to tensor
                s = torch.as_tensor(s, device=self.device)
                a_seq = torch.as_tensor(a_seq, device=self.device)
                trgt = torch.as_tensor(trgt, device=self.device)

                # any-step loss
                mean, logvar = self.dynamics(s, a_seq)
                inv_var = torch.exp(-logvar)
                mse_loss = (torch.pow(mean - trgt, 2) * inv_var).mean()
                var_loss = logvar.mean()
                loss = mse_loss + var_loss
                loss = loss + 0.01 * self.dynamics.max_logvar.sum() - 0.01 * self.dynamics.min_logvar.sum()

                # backward
                self.dyna_optim.zero_grad()
                loss.backward()
                self.dyna_optim.step()

                pbar.set_postfix(
                    train_loss=loss.item(),
                    holdout_loss=np.mean(holdout_losses)
                )

            new_val_losses, improve_ks = [], []
            for k in range(1, self.max_arm_step+1):
                k_step_seq = buffer.sample_nstep(batch_size, k, start_idx=train_size)
                k_val_loss = self.validate_dynamics_from(
                    s=k_step_seq["s"][:, 0],
                    a=k_step_seq["a"],
                    r=k_step_seq["r"][:, -1],
                    s_=k_step_seq["s_"][:, -1]
                )
                new_val_losses.append(k_val_loss)
                k_improvement = (holdout_losses[k-1] - k_val_loss) / holdout_losses[k-1]
                if k_improvement > 0:
                    improve_ks.append(k)

            if len(improve_ks) > 0 and np.mean(new_val_losses) < np.mean(holdout_losses):
                saved_dynamics = copy.deepcopy(self.dynamics)
                holdout_losses = new_val_losses
                cnt = 0
            else:
                cnt += 1

            if cnt >= 25 and epoch >= min_epochs:
                break

        self.dynamics = saved_dynamics
        return float(np.mean(holdout_losses))

    def validate_dynamics_from(self, s, a, r, s_):
        """ validate any-step rnn-based dynamics model (1-step validation) """
        s = torch.as_tensor(s, device=self.device)
        a = torch.as_tensor(a, device=self.device)
        r = torch.as_tensor(r, device=self.device)
        s_ = torch.as_tensor(s_, device=self.device)
        trgt = torch.cat((s_-s, r), dim=-1)

        mean, _ = self.dynamics(s, a)
        loss = ((mean - trgt) ** 2).mean()
        return float(loss.cpu().detach().numpy())

    def save_model(self, filepath):
        """ save model """
        super().save_model(filepath)
        torch.save(self.dynamics.state_dict(), filepath.replace("agent", "dyna"))

    def load_model(self, filepath):
        """ load model """
        super().load_model(filepath)
        state_dict = torch.load(filepath.replace("agent", "dyna"))
        self.dynamics.load_state_dict(state_dict)
