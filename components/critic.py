import torch
import numpy as np
import torch.nn as nn
from components.network import MLP

class Critic(nn.Module):
    """ Q(s,a) """
    def __init__(self, obs_shape, hidden_dims, action_dim):
        super(Critic, self).__init__()
        self.backbone = MLP(input_dim=np.prod(obs_shape)+action_dim, hidden_dims=hidden_dims)
        latent_dim = getattr(self.backbone, "output_dim")
        self.last = nn.Linear(latent_dim, 1)

    def forward(self, obs, actions):
        """ return Q(s,a) """
        net_in = torch.cat([obs, actions], dim=1)
        logits = self.backbone(net_in)
        values = self.last(logits)
        return values
