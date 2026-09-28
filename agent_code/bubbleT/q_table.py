"""Q-table indexing and loading."""
import json
from pathlib import Path
import numpy as np

ACTIONS = ('UP', 'LEFT', 'DOWN', 'RIGHT', 'WAIT', 'BOMB')
CHECKPOINT = Path(__file__).with_name('bubbleT.npz')


def state_id(features):
    mode, direction, useful_bomb = features
    return 12 * mode + 2 * (direction + 1) + int(useful_bomb)


def read_checkpoint(path=CHECKPOINT):
    with np.load(path, allow_pickle=False) as saved:
        q = saved['q']
        metadata = json.loads(str(saved['metadata']))
    if q.shape != (36, 6) or tuple(metadata['actions']) != ACTIONS:
        raise ValueError('Expected a 36 x 6 Q-table with the same action order.')
    return q, metadata
