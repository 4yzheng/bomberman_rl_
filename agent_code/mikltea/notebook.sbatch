#!/usr/bin/env bash
#SBATCH --job-name=mikltea-notebook
#SBATCH --partition=h200
#SBATCH --nodelist=mvl-gpu05
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:h200:1
#SBATCH --time=05:00:00
#SBATCH --output=/export/home/ra26xel/bb_rl/autoresearch/mikltea_checks/results/notebook-%j.out
set -eo pipefail
source /export/scratch/ra26xel/miniforge3/etc/profile.d/conda.sh
conda activate dav3
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
cd /export/home/ra26xel/bb_rl/bomberman_rl-master/agent_code/mikltea
# Register dav3 for the notebook; Jupyter retains its default token authentication.
python -m ipykernel install --user --name dav3 --display-name 'Python (dav3)'
srun jupyter lab --no-browser --ip=127.0.0.1 --port=8888 --ServerApp.port_retries=0
