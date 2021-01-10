import numpy as np
import torch
import torch.nn as nn

class Model(nn.Module):
    def __init__(self, seed, state_size, action_size):
        super(Model, self).__init__()

        self.seed = torch.manual_seed(seed)

        self.layers = nn.Sequential(
            nn.Linear(state_size, 256, bias=False),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Linear(256, 64, bias=True),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(64, action_size, bias=False)
        )

    def forward(self, state):
        return self.layers(state)
