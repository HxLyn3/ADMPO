import torch
import numpy as np
import torch.nn as nn
from components.network import MLP

class NormalWrapper(torch.distributions.Normal):
    """ wrapper of normal distribution """
    def log_prob(self, actions):
        return super().log_prob(actions).sum(-1, keepdim=True)

    def entropy(self):
        return super().entropy().sum(-1)

    def mode(self):
        return self.mean

class DiagGaussian(nn.Module):
    """ independent Gaussian """
    def __init__(
        self, 
        latent_dim, 
        output_dim, 
        unbounded=False, 
        conditioned_sigma=False, 
        max_mu=1.0, 
        sigma_min=-20, 
        sigma_max=2
    ):
        super(DiagGaussian, self).__init__()
        self.mu = nn.Linear(latent_dim, output_dim)
        self._c_sigma = conditioned_sigma
        if conditioned_sigma:
            self.sigma = nn.Linear(latent_dim, output_dim)
        else:
            self.sigma_param = nn.Parameter(torch.zeros(output_dim, 1))
        self._unbounded = unbounded
        self._max = max_mu
        self._sigma_min = sigma_min
        self._sigma_max = sigma_max

    def forward(self, logits):
        mu = self.mu(logits)
        if not self._unbounded:
            mu = self._max * torch.tanh(mu)
        if self._c_sigma:
            sigma = torch.clamp(self.sigma(logits), min=self._sigma_min, max=self._sigma_max).exp()
        else:
            shape = [1] * len(mu.shape)
            shape[1] = -1
            sigma = (self.sigma_param.view(shape) + torch.zeros_like(mu)).exp()
        return NormalWrapper(mu, sigma)

class ProbActor(nn.Module):
    """ stochastic actor for PPO/A2C/SAC """
    def __init__(self, obs_shape, hidden_dims, action_dim):
        super(ProbActor, self).__init__()
        self.backbone = MLP(input_dim=np.prod(obs_shape), hidden_dims=hidden_dims)
        self.dist_net = DiagGaussian(
            latent_dim=getattr(self.backbone, "output_dim"),
            output_dim=action_dim,
            unbounded=True,
            conditioned_sigma=True
        )

    def forward(self, obs):
        """ return prob distribution among actions """
        logits = self.backbone(obs)
        dist = self.dist_net(logits)
        return dist

class DeterActor(nn.Module):
    """ deterministic actor for DDPG/TD3 """
    def __init__(self, obs_shape, hidden_dims, action_dim, max_action):
        super(DeterActor, self).__init__()
        self.backbone = MLP(input_dim=np.prod(obs_shape), hidden_dims=hidden_dims)
        self.to_action = nn.Linear(getattr(self.backbone, "output_dim"), action_dim)
        self.max_action = max_action

    def forward(self, obs):
        """ return deterministic action """
        logits = self.backbone(obs)
        a = self.max_action*torch.tanh(self.to_action(logits))
        return a
