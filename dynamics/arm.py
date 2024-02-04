import torch
import torch.nn as nn
from torch.nn import functional as F

class Swish(nn.Module):
    def __init__(self):
        super(Swish, self).__init__()

    def forward(self, x):
        x = x * torch.sigmoid(x)
        return x

class ResBlock(nn.Module):
    def __init__(
        self,
        input_dim,
        output_dim,
        activation=Swish(),
        layer_norm=True,
        with_residual=True,
        dropout=0.1
    ):
        super().__init__()

        self.linear = nn.Linear(input_dim, output_dim)
        self.activation = activation
        self.layer_norm = nn.LayerNorm(output_dim) if layer_norm else None
        self.dropout = nn.Dropout(dropout) if dropout else None
        self.with_residual = with_residual
    
    def forward(self, x):
        y = self.activation(self.linear(x))
        if self.dropout is not None:
            y = self.dropout(y)
        if self.with_residual:
            y = x + y
        if self.layer_norm is not None:
            y = self.layer_norm(y)
        return y

class ARModel(nn.Module):
    """ Any-step RNN-based Dynamics Model (ARM) """

    def __init__(
        self,
        obs_dim,
        action_dim,
        output_dim,
        hidden_dim=200,
        rnn_num_layers=3,
        dropout=0.1,
        device="cuda:0"
    ):
        super().__init__()
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.device = device

        # rnn with any-step action sequence as input
        self.rnn_layer = nn.GRU(
            input_size=obs_dim+action_dim,
            hidden_size=hidden_dim,
            num_layers=rnn_num_layers,
            batch_first=True
        )
        # merging layer
        self.out_layer = nn.Sequential(
            ResBlock(hidden_dim, hidden_dim, dropout=dropout),
            ResBlock(hidden_dim, hidden_dim, dropout=dropout),
            ResBlock(hidden_dim, hidden_dim, dropout=dropout),
            ResBlock(hidden_dim, hidden_dim, dropout=dropout),
            nn.Linear(hidden_dim, self.output_dim)
        )

        self.to(device)

    def forward(self, obs, act_seq, h_state=None):
        self.rnn_layer.flatten_parameters()
        obs = obs[:, None].expand(-1, act_seq.shape[1], -1)
        rnn_in = torch.cat((obs, act_seq), dim=-1)
        rnn_out, h_state = self.rnn_layer(rnn_in, h_state)
        rnn_out = rnn_out[:, -1]
        output = self.out_layer(rnn_out)
        return output, h_state
