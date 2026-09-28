"""Seeded full-engine evaluation; does not modify agents or checkpoints."""
import argparse
import json
import logging
import os
from pathlib import Path
import random
import sys
import time
from types import SimpleNamespace

os.environ.setdefault('SDL_VIDEODRIVER', 'dummy')
os.environ.setdefault('SDL_AUDIODRIVER', 'dummy')
os.environ.setdefault('OMP_NUM_THREADS', '1')
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from environment import BombeRLeWorld


def evaluate(candidate, opponent, scenario, seeds, output):
    output.mkdir(parents=True, exist_ok=True)
    args = SimpleNamespace(no_gui=True, fps=10, turn_based=False, update_interval=.1,
        save_replay=False, replay=None, make_video=False, continue_without_training=True,
        log_dir=str(output), save_stats=False, match_name=None, seed=0,
        silence_errors=False, scenario=scenario)
    logging.disable(logging.CRITICAL)
    world = BombeRLeWorld(args, [(candidate, False)] + [(opponent, False)] * 3)
    rows = []
    start = time.perf_counter()
    # Legacy setup reseeds numpy from entropy. Reset RNGs after setup.
    for seed in seeds:
        random.seed(seed)
        np.random.seed(seed)
        world.rng = np.random.default_rng(seed)
        world.new_round()
        while world.running:
            world.do_step()
        scores = [a.score for a in world.agents]
        rows.append(dict(seed=seed, scores=scores, survived=not world.agents[0].dead,
                         stats=[dict(a.statistics) for a in world.agents]))
        if len(rows) % 10 == 0:
            print(f'{candidate} vs {opponent}: {len(rows)} rounds completed', flush=True)
    world.end()
    scores = np.array([r['scores'] for r in rows])
    summary = dict(candidate=candidate, opponent=opponent, scenario=scenario,
        rounds=len(rows), candidate_score=float(scores[:, 0].mean()),
        opponent_score=float(scores[:, 1:].mean()),
        outright_win_rate=float((scores[:, 0] > scores[:, 1:].max(axis=1)).mean()),
        tied_or_outright_win_rate=float((scores[:, 0] >= scores[:, 1:].max(axis=1)).mean()),
        survival_rate=float(np.mean([r['survived'] for r in rows])),
        suicides=sum(r['stats'][0].get('suicides', 0) for r in rows),
        mean_action_ms=1000 * sum(r['stats'][0].get('time', 0) for r in rows) /
                       max(1, sum(r['stats'][0].get('steps', 0) for r in rows)),
        elapsed_seconds=time.perf_counter()-start)
    path = output / f'{candidate}_vs_{opponent}_{scenario}.json'
    path.write_text(json.dumps(dict(summary=summary, rounds=rows), indent=2))
    print(json.dumps(summary), flush=True)
    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--candidate', default='planner_agent')
    parser.add_argument('--opponent', default='rule_based_agent')
    parser.add_argument('--scenario', default='classic')
    parser.add_argument('--rounds', type=int, default=100)
    parser.add_argument('--seed-start', type=int, default=1000)
    parser.add_argument('--output', type=Path, default=Path('benchmarks/results'))
    args = parser.parse_args()
    evaluate(args.candidate, args.opponent, args.scenario,
             range(args.seed_start, args.seed_start + args.rounds), args.output)
