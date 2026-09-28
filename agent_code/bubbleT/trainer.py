"""Load transitions and train a Q-table with ordinary Q-learning."""
import json
from pathlib import Path
import numpy as np
import torch
from .q_table import ACTIONS, CHECKPOINT, read_checkpoint


def load_replay(directories, validation_fraction=0.1):
    """Load saved transitions, keeping whole episodes in the validation set."""
    chunks = []
    for directory in directories:
        directory = Path(directory)
        manifest = json.loads((directory / 'manifest.json').read_text())
        metadata = json.loads((directory / 'metadata.json').read_text())
        assert tuple(metadata['actions']) == ACTIONS
        assert metadata['state'] == ['mode', 'direction', 'good_bomb']
        assert metadata['state_id'] == '12*mode + 2*(direction+1) + good_bomb'
        for entry in manifest['shards']:
            shard = torch.load(directory / entry['file'], map_location='cpu', weights_only=True)
            transitions = shard['transitions']
            done = transitions['terminated'] | transitions['truncated']
            next_state = transitions['next_state_id'].long()
            next_state[done] = 0
            # A fixed episode-based split keeps related transitions together.
            episode = transitions['episode']
            split = ((episode * 1103515245 + metadata['stage'] * 12345) % 2147483647).float() / 2147483647
            chunks.append({
                'state': transitions['state_id'].numpy(),
                'next_state': next_state.numpy(),
                'action': transitions['action'].numpy(),
                'score': (transitions['reward_score'] + transitions['posthumous_score']).numpy(),
                'guidance': transitions['reward_guidance'].numpy(),
                'done': done.numpy(),
                'validation': (split < validation_fraction).numpy(),
            })
    data = {}
    for key in chunks[0]:
        data[key] = np.concatenate([chunk[key] for chunk in chunks])
    validation = data.pop('validation')
    data['train_index'] = np.flatnonzero(~validation)
    data['validation_index'] = np.flatnonzero(validation)
    print(f"Loaded {len(data['action']):,} transitions")
    return data


class Trainer:
    def __init__(self, alpha=0.05, gamma=0.92, guidance_weight=1., seed=42):
        self.config = dict(alpha=alpha, gamma=gamma, guidance_weight=guidance_weight, seed=seed)
        self.q = np.zeros((36, 6), dtype=np.float32)
        self.rng = np.random.default_rng(seed)
        self.steps = 0
        self.history = []

    def fit(self, data, epochs, stage, checkpoint=CHECKPOINT, on_epoch=None):
        train = data['train_index']
        validation = data['validation_index']
        if not len(train) or not len(validation):
            raise ValueError('Both training and validation episodes are required.')
        # Use a fixed subset to keep validation quick on large datasets.
        positions = np.linspace(0, len(validation) - 1, min(16384, len(validation)), dtype=int)
        validation = validation[positions]
        rewards = data['score'] + self.config['guidance_weight'] * data['guidance']
        alpha = self.config['alpha']
        gamma = self.config['gamma']

        for _ in range(epochs):
            total_error = 0.
            for i in self.rng.permutation(train):
                state = data['state'][i]
                action = data['action'][i]
                next_state = data['next_state'][i]
                target = float(rewards[i])
                if not data['done'][i]:
                    target += gamma * float(self.q[next_state].max())
                error = target - float(self.q[state, action])
                self.q[state, action] += alpha * error
                total_error += error ** 2

            # Measure TD error on transitions that were not used for updates.
            states = data['state'][validation]
            actions = data['action'][validation]
            next_states = data['next_state'][validation]
            targets = rewards[validation] + gamma * (~data['done'][validation]) * self.q[next_states].max(axis=1)
            errors = targets - self.q[states, actions]
            self.steps += len(train)
            self.history.append({
                'epoch': len(self.history) + 1,
                'stage': stage,
                'steps': self.steps,
                'train_loss': total_error / len(train),
                'validation_loss': float(np.mean(errors ** 2)),
            })
            self.save(checkpoint)
            if on_epoch is not None:
                on_epoch(self.history)

    def save(self, path=CHECKPOINT):
        path = Path(path)
        metadata = dict(actions=ACTIONS, config=self.config, steps=self.steps,
                        history=self.history, rng=self.rng.bit_generator.state)
        # Replace the checkpoint only after the new file has been fully written.
        temporary = path.with_suffix('.tmp')
        with temporary.open('wb') as file:
            np.savez(file, q=self.q, metadata=json.dumps(metadata))
        temporary.replace(path)

    @classmethod
    def resume(cls, path=CHECKPOINT):
        q, saved = read_checkpoint(path)
        trainer = cls(**saved['config'])
        trainer.q = q
        trainer.steps = saved['steps']
        trainer.history = saved['history']
        trainer.rng.bit_generator.state = saved['rng']
        return trainer
