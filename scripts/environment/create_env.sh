#! /bin/bash

module load gpumem32gb cuda/13.0.2 cudnn/9.8.0.87-12 miniforge3

environment_scripts=$(dirname "$0")
scripts=$environment_scripts/..
base=$scripts/..

venvs=$base/venvs

# perhaps not necessary anymore
# export TMPDIR="/var/tmp"

mkdir -p $venvs

# venv for HF

conda create -y --prefix $venvs/mediapipe python=3.9

# venv for mmposewholebody

conda create -y --prefix $venvs/mmposewholebody python=3.8
