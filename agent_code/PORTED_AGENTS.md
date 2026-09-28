# Ported agents

`crazy_turtle` and `rule_based_scrape` were copied from the old project, excluding
logs and Python caches. The original `turtle_final.pt` and artwork are included.

Run from the current project directory:

```bash
sbatch play_h200.sbatch --agents crazy_turtle rule_based_scrape random_agent random_agent
```

To train the first two agents and collect scraper transitions:

```bash
sbatch play_h200.sbatch --agents crazy_turtle rule_based_scrape random_agent random_agent --train 2 --n-rounds 10
```

The port fixes missing `crazy_turtle_scrape.config` imports, updates the optional
visualization to the current state API, and uses explicit weights-only checkpoint
loading and the in-place Xavier initializer. The global CUDA visibility override
was removed; the original CPU configuration stays.

Original behavior is preserved:

- crazy_turtle always loads `turtle_final.pt`, including when training. Training
  writes `trutle.pt` in its agent directory; it does not automatically resume that
  file on the next run.
- rule_based_scrape depends on crazy_turtle's features, rewards, and replay buffer.
  Enable training to collect transitions and decay exploration. Its exploration
  also applies during non-training play.
- The scraper has a 7,000-transition buffer, saves every 10 rounds and when full,
  and exits the run when full. Memory files go in its agent directory. Serialized
  buffers include unused slots until full, as in the old code.
- Videos use the Slurm script's scratch directory. Model and replay-buffer paths
  are controlled by the agent configs. Use distinct paths for concurrent training
  jobs to avoid overwriting their output.

Validation in dav3: two inference rounds; two training rounds with temporary smaller
batch/save intervals to exercise optimization and serialization; one rendered MP4
game. Test checkpoints, replay buffers, and video outputs were written under `/tmp`.
