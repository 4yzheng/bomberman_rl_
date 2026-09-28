# mikltea

Offline Double DQN over the scraper's triple state, with a 10 → 32 → 32 → 6 network (1,606 parameters). Mode and direction use categorical indicators; all ten inputs are in {-1, +1}. Numeric direction labels are not treated as ordered magnitudes.

`features.py`, `legacy_features.py` and `config.py` are unchanged copies of the scraping teacher's extractor (candidate 0022). The learned decision replaces the teacher's rule; no game-engine changes are needed. `model.py` contains the encoder/network, `offline.py` the dataset loader/trainer, and `callbacks.py` inference. Existing feature code is retained to guarantee identical preprocessing, rather than reimplementing it.

Open **train.ipynb** with the dav3 kernel in a Slurm H200 allocation. The notebook loads both stages once onto the GPU, trains stage 2 then stage 3, and plots training/held-out TD loss each epoch. It uses committed manifest snapshots even while collection continues; reload data to include later shards. At roughly two million transitions the replay tensors need well below 1 GiB of GPU memory. The small model does not require mixed precision or a DataLoader.

Launch a notebook server:

```bash
sbatch /export/home/ra26xel/bb_rl/bomberman_rl-master/agent_code/mikltea/notebook.sbatch
```

Read `autoresearch/mikltea_checks/results/notebook-JOBID.out` for Jupyter's authenticated URL/token. Connect from your computer with an SSH tunnel (substitute your cluster login host; compute-node SSH must be permitted):

```bash
ssh -J ra26xel@LOGIN_HOST -L 8888:127.0.0.1:8888 ra26xel@mvl-gpu05
```

Open the localhost URL with the token from the log. The server binds only to loopback. If your cluster provides its own notebook/IDE forwarding, use that instead. dav3 needs torch, numpy, matplotlib, jupyterlab and ipykernel.

The notebook automatically saves **mikltea.pt in this directory** after every epoch, with weights, target weights, Adam state, RNG state, settings, loss history and dataset manifest hashes. It prevents accidental fresh-run overwrites; set `RESUME=True` to continue. No trained production checkpoint is supplied before running the notebook. Smoke checks use disposable weights elsewhere.

Default reward: actual game points + guidance, including posthumous points on the final transition. The saved guidance scheme is escape/bomb +0.10, coin/target movement +0.05, disagreement −0.05, invalid −0.10, otherwise 0. The weight is configurable and can be reduced during later training. Gamma starts at the old agent's 0.92; both death and 400-step truncation stop bootstrapping. These are initial choices, not tuned performance claims.

Validation splits whole episodes, checksums every shard, rejects duplicate episodes and checks the feature-extractor hashes. Held-out TD loss is a diagnostic; select policies using points per game on independent seeds against three rule-based opponents. Offline Double DQN alone does not prevent errors on poorly sampled actions.

After training, play through your existing Slurm script with `--agents mikltea rule_based_agent rule_based_agent rule_based_agent`; omit `--train`. Inference loads weights on CPU and caches the exact network policy over the 36 possible states. The spelling `mikltea` follows the requested agent name.
