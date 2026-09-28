"""GPU-resident replay and offline Double DQN; no game-engine changes required."""
import hashlib
import json
import math
from pathlib import Path
import torch
from torch.nn import functional as F
from .model import ACTIONS, CHECKPOINT, ENCODING, QNetwork, state_table


def load_replay(directories, device='cuda', validation_fraction=.1):
    """Read each committed shard once; keep whole episodes in a fixed holdout split."""
    assert 0 < validation_fraction < 1
    chunks, sources, seen = [], [], set()
    feature_root = Path(__file__).parent
    for directory in map(Path, directories):
        manifest_path = directory/'manifest.json'
        if not manifest_path.exists():
            print(f'No committed data yet: {directory}')
            continue
        manifest_bytes = manifest_path.read_bytes()
        manifest = json.loads(manifest_bytes)
        meta = json.loads((directory/'metadata.json').read_text())
        assert tuple(meta['actions']) == ACTIONS
        for name in ('features.py', 'legacy_features.py', 'config.py'):
            expected = meta['source_sha256'][f'engine/agent_code/scraper/{name}']
            assert hashlib.sha256((feature_root/name).read_bytes()).hexdigest() == expected
        sources.append(dict(path=str(directory),sha256=hashlib.sha256(manifest_bytes).hexdigest(),
                            transitions=manifest['transitions'],complete=manifest['complete']))
        for entry in manifest['shards']:
            path = directory/entry['file']
            import io
            raw = path.read_bytes()
            assert hashlib.sha256(raw).hexdigest() == entry['sha256'], f'Corrupt shard: {path}'
            shard = torch.load(io.BytesIO(raw), map_location='cpu', weights_only=True)
            for episode in shard['episodes']:
                key = (meta['stage'], episode['episode'])
                assert key not in seen, f'Duplicate episode {key}'
                seen.add(key)
            t = shard['transitions']
            assert len(t['action']) == entry['transitions']
            # The 400-step game is treated as an episodic task: no bootstrap at either end.
            done = t['terminated'] | t['truncated']
            next_id = t['next_state_id'].long().masked_fill(done, 0)
            assert (t['state_id'] < 36).all() and (next_id < 36).all()
            # Stable per-episode split across all extensions; no adjacent-step leakage.
            split = ((t['episode'] * 1103515245 + meta['stage'] * 12345) % 2147483647).float() / 2147483647
            chunks.append(dict(state=t['state_id'].long(),next_state=next_id,action=t['action'].long(),
                score=t['reward_score']+t['posthumous_score'],guidance=t['reward_guidance'],
                done=done,holdout=split < validation_fraction))
        print(f"Loaded {manifest['transitions']:,} transitions from {directory.name} "
              f"({'complete' if manifest['complete'] else 'committed snapshot; collection still running'})")
    if not chunks:
        raise ValueError('No committed transitions found')
    data = {k:torch.cat([c[k] for c in chunks]).to(device) for k in chunks[0]}
    holdout = data.pop('holdout')
    data['train_index'] = (~holdout).nonzero().flatten()
    data['validation_index'] = holdout.nonzero().flatten()
    assert len(data['train_index']) and len(data['validation_index'])
    gib = sum(t.numel()*t.element_size() for t in data.values())/1024**3
    print(f"Replay on {device}: {len(data['action']):,} rows, {gib:.3f} GiB")
    return data, sources


class Trainer:
    def __init__(self, device='cuda', lr=3e-4, gamma=.92, guidance_weight=1.,
                 batch_size=4096, target_interval=250, seed=42):
        torch.manual_seed(seed)
        self.device = torch.device(device)
        self.config = dict(lr=lr,gamma=gamma,guidance_weight=guidance_weight,
                           batch_size=batch_size,target_interval=target_interval,seed=seed)
        self.online = QNetwork().to(device)
        self.target = QNetwork().to(device).eval()
        self.target.load_state_dict(self.online.state_dict())
        self.optimizer = torch.optim.Adam(self.online.parameters(), lr=lr)
        self.inputs = state_table(device)
        self.steps, self.history, self.sources = 0, [], []

    def loss(self, data, index):
        q = self.online(self.inputs[data['state'][index]]).gather(1,data['action'][index,None]).squeeze(1)
        with torch.no_grad():
            nxt = self.inputs[data['next_state'][index]]
            # Double DQN: online chooses the action; target evaluates it.
            action = self.online(nxt).argmax(1,keepdim=True)
            value = self.target(nxt).gather(1,action).squeeze(1)
            reward = data['score'][index]+self.config['guidance_weight']*data['guidance'][index]
            target = reward+self.config['gamma']*(~data['done'][index])*value
        return F.smooth_l1_loss(q,target)

    def fit(self, data, epochs, stage, checkpoint=CHECKPOINT, on_epoch=None):
        train = data['train_index']
        # Fixed validation subset, to keep plots comparable from epoch to epoch.
        validation = data['validation_index']
        val = validation[torch.linspace(0,len(validation)-1,min(16384,len(validation)),device=self.device).long()]
        batches = math.ceil(len(train)/self.config['batch_size'])
        for _ in range(epochs):
            total = torch.zeros((),device=self.device)
            for _ in range(batches):
                index = train[torch.randint(len(train),(self.config['batch_size'],),device=self.device)]
                loss = self.loss(data,index)
                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.online.parameters(),10.)
                self.optimizer.step()
                self.steps += 1
                total += loss.detach()
                if self.steps % self.config['target_interval'] == 0:
                    self.target.load_state_dict(self.online.state_dict())
            with torch.no_grad():
                validation = float(self.loss(data,val))
            row = dict(epoch=len(self.history)+1,stage=stage,steps=self.steps,
                       train_loss=float(total/batches),validation_loss=validation,
                       guidance_weight=self.config['guidance_weight'])
            self.history.append(row)
            self.save(checkpoint)
            if on_epoch: on_epoch(self.history)
        return self.history

    def save(self, path=CHECKPOINT):
        path = Path(path)
        temporary = path.with_suffix('.pt.tmp')
        torch.save(dict(encoding=ENCODING,actions=list(ACTIONS),online=self.online.state_dict(),
            target=self.target.state_dict(),optimizer=self.optimizer.state_dict(),config=self.config,
            steps=self.steps,history=self.history,sources=self.sources,
            cpu_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state(self.device) if self.device.type=='cuda' else None),temporary)
        temporary.replace(path)

    @classmethod
    def resume(cls, path=CHECKPOINT, device='cuda'):
        saved = torch.load(path,map_location='cpu',weights_only=True)
        assert saved['encoding']==ENCODING and tuple(saved['actions'])==ACTIONS
        trainer = cls(device=device,**saved['config'])
        trainer.online.load_state_dict(saved['online'])
        trainer.target.load_state_dict(saved['target'])
        trainer.optimizer.load_state_dict(saved['optimizer'])
        trainer.steps, trainer.history, trainer.sources = saved['steps'],saved['history'],saved['sources']
        torch.set_rng_state(saved['cpu_rng'])
        if trainer.device.type=='cuda' and saved['cuda_rng'] is not None:
            torch.cuda.set_rng_state(saved['cuda_rng'],trainer.device)
        return trainer
