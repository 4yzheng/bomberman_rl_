# Planner evaluation

The candidate is a small rule-based policy inspired by `FML_Report/main.html`.
The report describes the supplied rule-based agent as its baseline and crazy_turtle
as the learned agent. We test against `rule_based_agent`, not the scraper's default
100% random exploration policy.

## Protocol

- Full, unmodified game engine; one candidate versus three `rule_based_agent`s.
- Development: classic seeds 0–19. The first version delayed escaping and scored
  3.15 points/round, won 30% outright, and survived 20%. Giving immediate escape
  priority improved these to 4.55 points, 50% wins, and 85% survival on those seeds.
- Freeze the revised policy before evaluating classic seeds 1000–1049.
- Evaluate pretrained crazy_turtle on the same classic seeds and opponent lineup.
- Additional scenario check: coin-heaven seeds 2000–2019.
- Reset Python random, NumPy random, and the engine RNG for every round, after
  legacy agent setup has performed its entropy-based reseeding. Starting positions
  are randomized by the engine. All four agents play without training.
- Score is the engine reward (coin = 1, opponent kill = 5). Outright wins require
  strictly more points than every opponent; tied wins are recorded separately.
- Store per-round seeds, scores, survival and event counters in the JSON artifacts.
  Timings are local dav3 CPU timings, not H200 GPU measurements.

These samples are evidence for this opponent population, not a tournament-wide
ranking. The standard opponents' random action streams diverge as games diverge,
so equal seeds do not imply identical opponent actions throughout both matches.
Opponent motion and newly placed bombs can invalidate a previously safe path.

## Reproduce

From `bomberman_rl-master`, with `dav3` activated:

```bash
python benchmarks/test_planner.py
python benchmarks/evaluate_planner.py --rounds 50 --seed-start 1000 --output benchmarks/reproduced
python benchmarks/evaluate_planner.py --candidate crazy_turtle --rounds 50 --seed-start 1000 --output benchmarks/reproduced
python benchmarks/evaluate_planner.py --scenario coin-heaven --rounds 20 --seed-start 2000 --output benchmarks/reproduced
```

The harness supplies dummy SDL drivers and disables game logging during benchmarks.
No model checkpoint is modified. Do not use `--train` for the planner: it has no
training callbacks.

## Implementation size

The planner is one 157-line file (129 Python statements), versus 210 lines (145
statements) for the standard rule-based callback. Its three priorities share one
forecast/search implementation. Unlike crazy_turtle, it needs no neural network,
feature-extractor module, training code or checkpoint. NumPy and engine settings are
its only non-standard-library imports.

## Held-out results

Classic, 50 rounds per candidate, seeds 1000–1049:

| Candidate | Points/round | Points/opponent/round | Outright wins | Wins including ties | Survival | Suicides | Mean action ms |
|---|---:|---:|---:|---:|---:|---:|---:|
| planner_agent | 5.08 | 2.84 | 42% | 52% | 80% | 5 | 6.98 |
| crazy_turtle | 4.58 | 2.79 | 42% | 46% | 58% | 14 | 6.01 |

The planner scores above the average standard opponent, with 80% survival.
Compared with crazy_turtle, it has a modestly higher mean score and fewer suicides,
but the same outright win rate; this sample does not establish a decisive advantage
over the learned agent. It is substantially smaller than the learned pipeline.

In 20 coin-heaven rounds (seeds 2000–2019), the planner scored 16.35 versus
13.88 per standard opponent, with 40% outright wins and 50% survival. Seven suicides
show that moving opponents and new bombs remain a weakness.

All six physics/policy regression tests passed. Final classic per-round records:

- [Planner](heldout/planner_agent_vs_rule_based_agent_classic.json)
- [crazy_turtle](heldout/crazy_turtle_vs_rule_based_agent_classic.json)
- [Coin-heaven](heldout/planner_agent_vs_rule_based_agent_coin-heaven.json)
