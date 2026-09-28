"""Greedy CPU inference from the offline-trained checkpoint."""
import torch
from .features import state_to_features
from .model import ACTIONS, CHECKPOINT, ENCODING, QNetwork, state_table


def setup(self):
    if not CHECKPOINT.exists():
        raise FileNotFoundError(f'Train with train.ipynb first; expected {CHECKPOINT}')
    saved = torch.load(CHECKPOINT, map_location='cpu', weights_only=True)
    assert saved['encoding'] == ENCODING and tuple(saved['actions']) == ACTIONS
    net = QNetwork().eval()
    net.load_state_dict(saved['online'])
    # The network has only 36 distinct inputs. Cache its exact greedy policy.
    with torch.inference_mode():
        self.policy = net(state_table('cpu')).argmax(-1).tolist()


def act(self, game_state):
    mode, direction, bomb = state_to_features(game_state)
    return ACTIONS[self.policy[12 * int(mode) + 2 * (int(direction) + 1) + int(bomb)]]
