#! /bin/bash

# (srun --pty -n 1 -c 2 --time=01:00:00 --mem=16G bash)

debug_scripts=$(dirname "$0")
scripts=$debug_scripts/..
base=$scripts/..
base=$(realpath $base)

module load miniforge3

source activate $base/venvs/default

python $base/scripts/debugging/debug_reproducibility.py \
    --checkpoint-1 $base/models/phoenix_1/setup/model \
    --checkpoint-2 $base/models/phoenix_2/setup/model
