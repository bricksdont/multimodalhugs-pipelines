#! /bin/bash

module load gpumem32gb cuda/13.0.2 cudnn/9.8.0.87-12 miniforge3

environment_scripts=$(dirname "$0")
base=$environment_scripts/../..
base=$(realpath $base)

venvs=$base/venvs
tools=$base/tools

venv_name="o_brien_et_al_2026"

mkdir -p $venvs $tools/$venv_name

# create venv only if it does not yet exist

if [[ -d $venvs/$venv_name ]]; then
    echo "Venv $venvs/$venv_name already exists, skipping creation"
else
    conda create -y --prefix $venvs/$venv_name python=3.11.13
fi

source activate $venvs/$venv_name

# install multimodalhugs, pinned to exact commit for reproducibility
# Using bricksdont fork, branch cherry-pick-catherine-one-person-per-frame,
# which adds support for multi-person pose files (e.g. openpose sometimes detects >1 person).

git clone https://github.com/bricksdont/multimodalhugs.git $tools/$venv_name/multimodalhugs

(cd $tools/$venv_name/multimodalhugs && git checkout "881f4b7121577f9e4a8ea276c30f9499d81f111b")

(cd $tools/$venv_name/multimodalhugs && pip install .)

# install SL datasets

pip install git+https://github.com/sign-language-processing/datasets.git

# pose-format fork, pinned to exact commit for reproducibility
# Cloned without submodules to avoid SSH auth failure on the pose-pipelines submodule.

git clone --no-recurse-submodules https://github.com/GerrySant/pose.git $tools/$venv_name/pose-format

(cd $tools/$venv_name/pose-format && git checkout "c38880312aaefdf07298dce1548ad619734420ba")

pip install $tools/$venv_name/pose-format/src/python

# TF keras, because keras 3 is not supported in Transformers

pip install tf-keras

# bleurt not supported out of the box with evaluate

pip install git+https://github.com/google-research/bleurt.git

# openGL is no longer available on the cluster

OPENCV_VERSION=$(python - <<'EOF'
import importlib.metadata as m
try:
    print(m.version("opencv-python"))
except m.PackageNotFoundError:
    try:
        print(m.version("opencv-python-headless"))
    except m.PackageNotFoundError:
        print("")
EOF
)

pip uninstall -y opencv-python opencv-python-headless
if [[ -n $OPENCV_VERSION ]]; then
    pip install "opencv-python-headless==${OPENCV_VERSION}"
else
    pip install opencv-python-headless
fi

# a missing dependency of etils (tfds)

pip install importlib_resources

