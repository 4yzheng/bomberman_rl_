# Planner agent

A deterministic rule-based agent, inspired by the safety → coins → bombing priority
in `FML_Report/main.html`. It uses NumPy and the engine settings, with no PyTorch,
checkpoint, training phase, or dependencies on the old agents.

The policy has three priorities:

1. Leave active blast lanes immediately using a safe route.
2. Take the shortest safe route to a visible coin.
3. Approach productive bomb sites, and plant only after finding an escape route.

`Forecast` models each known bomb through its explosion, then runs backwards
reachability and breadth-first search over positions and time. This lets the agent
wait for explosions to clear and distinguishes an apparently safe next move from
one that leads into a trap. Bomb-site utility is `(crates + 3 * opponents) /
(distance + 4)`, with a small recent-visit penalty. Distance is the actual safe path
length, rather than Manhattan distance. Crates already covered by existing bombs
are excluded from utility. Opponents in a blast are potential targets, not guaranteed
kills.

The physics follows this repository: blasts pass through crates and bombs, stop at
walls, and do not trigger chain reactions. Existing explosions are timed using the
engine's `explosion_map`. Bombs and destroyed crates become traversable on the move
after their explosion. Planning conservatively treats current opponent positions
as blocked. It cannot guarantee safety against future opponent moves or new bombs.
Leaving blast lanes early reduces the chance of opponents blocking a late escape.

## Run

From `bomberman_rl-master`:

```bash
python main.py play --agents planner_agent rule_based_agent rule_based_agent rule_based_agent
sbatch play_h200.sbatch --agents planner_agent rule_based_agent rule_based_agent rule_based_agent --n-rounds 10
```

The Slurm command records videos in `/export/scratch/ra26xel/bomberman/<job_id>/`.
The agent itself runs on CPU. It is an evaluation baseline and has no training
callbacks; do not include it among the agents selected by `--train`.

## Evaluation

See `../../benchmarks/README.md` for seeded game results and reproduction commands.
Safety regression tests are in `../../benchmarks/test_planner.py`.
