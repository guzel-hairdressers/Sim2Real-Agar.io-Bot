"""Learned Bernoulli event head, separate from continuous movement control."""
import torch
from torch import nn


class BinarySplitHead(nn.Module):
    def __init__(self, obs_dim=70):
        super().__init__()
        self.register_buffer('mean', torch.zeros(obs_dim))
        self.register_buffer('scale', torch.ones(obs_dim))
        self.net = nn.Sequential(nn.Linear(obs_dim, 128), nn.Tanh(),
                                 nn.Linear(128, 64), nn.Tanh(), nn.Linear(64, 1))

    def forward(self, obs):
        normalized = ((obs-self.mean)/self.scale).clamp(-8, 8)
        return self.net(normalized)
