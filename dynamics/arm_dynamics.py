import numpy as np

import torch
import torch.nn as nn
from torch.nn import functional as F

from .arm import ARModel

def soft_clamp(x : torch.Tensor, _min=None, _max=None):
    # clamp tensor values while maintaining the gradient
    if _max is not None:
        x = _max - F.softplus(_max - x)
    if _min is not None:
        x = _min + F.softplus(x - _min)
    return x

class ARMDynamics(nn.Module):
    """ Any-step RNN-based Dynamics """

    def __init__(
        self,
        obs_dim,
        action_dim,
        hidden_dim=200,
        rnn_num_layers=3,
        dropout=0.1,
        device="cuda:0"
    ):
        super().__init__()
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.output_dim = (self.obs_dim + 1) * 2
        self.device = device

        self.model = ARModel(
            obs_dim=obs_dim,
            action_dim=action_dim,
            output_dim=self.output_dim,
            hidden_dim=hidden_dim,
            rnn_num_layers=rnn_num_layers,
            dropout=dropout,
            device=device
        )

        # 'mean' and 'std' for normalization
        self.register_parameter("obs_mu", nn.Parameter(torch.zeros(self.obs_dim), requires_grad=False))
        self.register_parameter("obs_std", nn.Parameter(torch.zeros(self.obs_dim), requires_grad=False))
        self.register_parameter("act_mu", nn.Parameter(torch.zeros(self.action_dim), requires_grad=False))
        self.register_parameter("act_std", nn.Parameter(torch.zeros(self.action_dim), requires_grad=False))

        self.register_parameter(
            "max_logvar",
            nn.Parameter(torch.ones(self.obs_dim + 1) * 0.5, requires_grad=True)
        )
        self.register_parameter(
            "min_logvar",
            nn.Parameter(torch.ones(self.obs_dim + 1) * -10, requires_grad=True)
        )

        self.to(self.device)

    def set_mu_std(self, obs_mu, obs_std, act_mu, act_std):
        self.obs_mu.data = torch.as_tensor(obs_mu, dtype=torch.float32, device=self.device)
        self.obs_std.data = torch.as_tensor(obs_std, dtype=torch.float32, device=self.device)
        self.act_mu.data = torch.as_tensor(act_mu, dtype=torch.float32, device=self.device)
        self.act_std.data = torch.as_tensor(act_std, dtype=torch.float32, device=self.device)

    def forward(self, obs, action):
        # shape@obs: (bs, obs_dim)
        # shape@actoins: (bs, h_step, act_dim)
        # normalization
        _obs = (obs - self.obs_mu) / self.obs_std
        _action = (action - self.act_mu) / self.act_std

        model_out, _ = self.model(_obs, _action)
        mean, logvar = torch.chunk(model_out, 2, dim=-1)
        logvar = soft_clamp(logvar, self.min_logvar, self.max_logvar)
        return mean, logvar

    @ torch.no_grad()
    def step(self, obs, action):
        _obs = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
        _action = torch.as_tensor(action, dtype=torch.float32, device=self.device)
        mean, logvar = self.forward(_obs, _action)
        mean = mean.cpu().detach().numpy()
        logvar = logvar.cpu().detach().numpy()

        mean[:, :-1] += obs
        std = np.sqrt(np.exp(logvar))
        sample = (mean + np.random.normal(size=mean.shape) * std).astype(np.float32)

        next_obs = sample[:, :-1]
        reward = sample[:, -1:]
        return next_obs, reward

    @ torch.no_grad()
    def dstep(self, obs, action):
        """ deterministic step """
        _obs = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
        _action = torch.as_tensor(action, dtype=torch.float32, device=self.device)
        mean, _ = self.forward(_obs, _action)
        mean = mean.cpu().detach().numpy()
        mean[:, :-1] += obs
        return mean[:, :-1], mean[:, -1:]
