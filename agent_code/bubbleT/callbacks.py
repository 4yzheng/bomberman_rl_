"""Choose the action with the highest Q-value."""
from .features import state_to_features
from .q_table import ACTIONS, read_checkpoint, state_id


def setup(self):
    self.q, _ = read_checkpoint()


def act(self, game_state):
    state = state_id(state_to_features(game_state))
    action = self.q[state].argmax()
    return ACTIONS[action]
