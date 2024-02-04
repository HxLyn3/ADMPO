import torch.nn as nn

class MLP(nn.Module):
    """ multi-layer perceptron """
    def __init__(self, input_dim, hidden_dims, output_dim=None, activation=nn.ReLU):
        super(MLP, self).__init__()

        # hidden layers
        dims = [input_dim] + list(hidden_dims)
        layers = []
        for in_dim, out_dim in zip(dims[:-1], dims[1:]):
            layers += [nn.Linear(in_dim, out_dim), activation()]
        
        self.output_dim = dims[-1]
        if output_dim is not None:
            layers += [nn.Linear(dims[-1], output_dim)]
            self.output_dim = output_dim
        self.net = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.net(x)
