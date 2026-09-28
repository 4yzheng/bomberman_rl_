"""Small Q-network and a single shared encoding for training and gameplay."""
from pathlib import Path
import torch
from torch import nn
from torch.nn import functional as F

ACTIONS = ('UP', 'LEFT', 'DOWN', 'RIGHT', 'WAIT', 'BOMB')
CHECKPOINT = Path(__file__).with_name('mikltea.pt')
ENCODING = 'mode-onehot3_direction-onehot6_bomb1_bipolar_v1'


def encode(triples):
    """(..., 3) raw triples -> (..., 10) categorical features in {-1, +1}."""
    x = triples.long()
    return torch.cat((F.one_hot(x[..., 0], 3),
                      F.one_hot(x[..., 1] + 1, 6), x[..., 2:3]), -1).float() * 2 - 1


def state_table(device):
    ids = torch.arange(36, device=device)
    return encode(torch.stack((ids // 12, (ids % 12) // 2 - 1, ids % 2), -1))


class QNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(10, 32), nn.ReLU(),
                                 nn.Linear(32, 32), nn.ReLU(), nn.Linear(32, 6))

    def forward(self, x):
        return self.net(x)
