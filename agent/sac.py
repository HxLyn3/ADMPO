import torch
import numpy as np
from copy import deepcopy

from components import ACTOR, CRITIC

class SACAgent:
    """ soft actor-critic """

    def __init__(
        self, 
        obs_shape, 
        hidden_dims, 
        action_dim,
        action_space,
        actor_freq,
        actor_lr,
        critic_lr,
        tau=0.005, 
        gamma=0.99, 
        alpha=0.2,
        auto_alpha=True,
        alpha_lr=3e-4,
        target_entropy=-1,
        deterministic_backup=False,
        q_clip=None,
        device="cuda:0"
    ):
        # actor
        self.actor = ACTOR["prob"](obs_shape, hidden_dims, action_dim).to(device)

        # critic
        self.critic1 = CRITIC["q"](obs_shape, hidden_dims, action_dim).to(device)
        self.critic2 = CRITIC["q"](obs_shape, hidden_dims, action_dim).to(device)
        # target critic
        self.critic1_trgt = deepcopy(self.critic1)
        self.critic2_trgt = deepcopy(self.critic2)
        self.critic1_trgt.eval()
        self.critic2_trgt.eval()

        # optimizer
        self.actor_optim = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic1_optim = torch.optim.Adam(self.critic1.parameters(), lr=critic_lr)
        self.critic2_optim = torch.optim.Adam(self.critic2.parameters(), lr=critic_lr)

        # action space
        self.action_space = action_space

        # actor update frequency
        self.actor_freq = actor_freq
        self.critic_cnt = 0

        # alpha: weight of entropy
        self._auto_alpha = auto_alpha
        if self._auto_alpha:
            if not target_entropy:
                target_entropy = -np.prod(self.action_space.shape)
            self._target_entropy = target_entropy
            self._log_alpha = torch.zeros(1, requires_grad=True, device=device)
            self._alpha = self._log_alpha.detach().exp()
            self._alpha_optim = torch.optim.Adam([self._log_alpha], lr=alpha_lr)
        else:
            self._alpha = alpha*torch.ones(1, device=device)

        # other parameters
        self._tau = tau
        self._gamma = gamma
        self._deterministic_backup = deterministic_backup
        self._q_clip = q_clip
        self._eps = np.finfo(np.float32).eps.item()
        self.device = device

    def train(self):
        self.actor.train()
        self.critic1.train()
        self.critic2.train()

    def eval(self):
        self.actor.eval()
        self.critic1.eval()
        self.critic2.eval()

    def _sync_weight(self):
        """ synchronize weight """
        for trgt, src in zip(self.critic1_trgt.parameters(), self.critic1.parameters()):
            trgt.data.copy_(trgt.data*(1.0-self._tau) + src.data*self._tau)
        for trgt, src in zip(self.critic2_trgt.parameters(), self.critic2.parameters()):
            trgt.data.copy_(trgt.data*(1.0-self._tau) + src.data*self._tau)

    def actor4ward(self, obs, deterministic=False):
        """ forward propagation of actor """
        dist = self.actor(obs)
        if deterministic:
            action = dist.mode()
        else:
            action = dist.rsample()
        log_prob = dist.log_prob(action)

        action_scale = torch.tensor((self.action_space.high-self.action_space.low)/2, device=self.device)
        squashed_action = torch.tanh(action)
        log_prob = log_prob - torch.log(action_scale*(1-squashed_action.pow(2))+self._eps).sum(-1, keepdim=True)

        return action_scale*squashed_action, log_prob

    def act(self, obs, deterministic=False):
        """ sample action """
        with torch.no_grad():
            obs = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
            action, log_prob = self.actor4ward(obs, deterministic)
            action = action.cpu().detach().numpy()
            log_prob = log_prob.cpu().detach().numpy()
        return action, log_prob

    def learn(self, s, a, r, s_, done):
        """ learn from (s, a, r, s_, done) """
        s    = torch.as_tensor(s, device=self.device)
        a    = torch.as_tensor(a, device=self.device)
        r    = torch.as_tensor(r, device=self.device)
        s_   = torch.as_tensor(s_, device=self.device)
        done = torch.as_tensor(done, device=self.device)

        # update critic
        q1, q2 = self.critic1(s, a).flatten(), self.critic2(s, a).flatten()
        with torch.no_grad():
            a_, log_prob_ = self.actor4ward(s_)
            q_ = torch.min(self.critic1_trgt(s_, a_), self.critic2_trgt(s_, a_))
            if self._q_clip is not None:
                q_ = q_.clip(None, self._q_clip)
            if not self._deterministic_backup:
                q_ -= self._alpha*log_prob_
            q_trgt = r.flatten() + self._gamma*(1-done.flatten())*q_.flatten()

        critic1_loss = ((q1-q_trgt).pow(2)).mean()
        self.critic1_optim.zero_grad()
        critic1_loss.backward()
        self.critic1_optim.step()

        critic2_loss = ((q2-q_trgt).pow(2)).mean()
        self.critic2_optim.zero_grad()
        critic2_loss.backward()
        self.critic2_optim.step()

        self.critic_cnt += 1

        actor_loss, alpha_loss = [None]*2
        if self.critic_cnt % self.actor_freq == 0:
            # update actor
            a, log_prob = self.actor4ward(s)
            q1, q2 = self.critic1(s, a).flatten(), self.critic2(s, a).flatten()
            actor_loss = (self._alpha*log_prob.flatten() - torch.min(q1, q2)).mean()
            self.actor_optim.zero_grad()
            actor_loss.backward()
            self.actor_optim.step()

            # update alpha
            if self._auto_alpha:
                log_prob = log_prob.detach() + self._target_entropy
                alpha_loss = -(self._log_alpha*log_prob).mean()
                self._alpha_optim.zero_grad()
                alpha_loss.backward()
                self._alpha_optim.step()
                self._alpha = self._log_alpha.detach().exp()

        # synchronize weight
        self._sync_weight()

        info = {
            "loss": {
                "actor": actor_loss.item() if actor_loss else None,
                "critic1": critic1_loss.item(),
                "critic2": critic2_loss.item()
            }
        }

        if self._auto_alpha:
            info["loss"]["alpha"] = alpha_loss.item() if alpha_loss else None
            info["alpha"] = self._alpha.item()
        else:
            info["loss"]["alpha"] = 0
            info["alpha"] = self._alpha.item()

        return info

    def save_model(self, filepath):
        """ save model """
        state_dict = {
            "actor": self.actor.state_dict(),
            "critic1": self.critic1.state_dict(),
            "critic2": self.critic2.state_dict(),
            "alpha": self._alpha
        }
        torch.save(state_dict, filepath)

    def load_model(self, filepath):
        """ load model """
        state_dict = torch.load(filepath)
        self.actor.load_state_dict(state_dict["actor"])
        self.critic1.load_state_dict(state_dict["critic1"])
        self.critic2.load_state_dict(state_dict["critic2"])
        self._alpha = state_dict["alpha"]
