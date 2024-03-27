import torch
import torch.nn as nn

class MLP(nn.Moudule):
    def __init__(self, batch_size, in_n, hidden_n, out_n):
        super(MLP, self).__init__()
        self.batch_size = batch_size

        layers = []
        layers += [nn.Linear(in_features=in_n, out_features=hidden_n),
                   nn.RELU(inplace=True),
                   nn.Linear(hidden_n, hidden_n),
                   nn.RELU(inplace=True),
                   nn.Linear(hidden_n, out_n)]

        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        x = x.view(self.batch_size, -1)
        return self.layers(x)
