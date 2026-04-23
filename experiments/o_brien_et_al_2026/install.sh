#! /bin/bash

module load gpumem32gb cuda/13.0.2 cudnn/9.8.0.87-12 miniforge3

environment_scripts=$(dirname "$0")
base=$environment_scripts/../..
base=$(realpath $base)

venvs=$base/venvs
tools=$base/tools

venv_name="o_brien_et_al_2026"

mkdir -p $venvs $tools

# create venv only if it does not yet exist

if [[ -d $venvs/$venv_name ]]; then
    echo "Venv $venvs/$venv_name already exists, skipping creation"
else
    conda create -y --prefix $venvs/$venv_name python=3.11.13
fi

source activate $venvs/$venv_name

# install multimodalhugs, pinned to exact commit for reproducibility

git clone https://github.com/GerrySant/multimodalhugs.git $tools/multimodalhugs-$venv_name

(cd $tools/multimodalhugs-$venv_name && git checkout "5201c80f27aa70c460e8297a799dc5daccbd1b3b")

(cd $tools/multimodalhugs-$venv_name && pip install .)

# install SL datasets

pip install git+https://github.com/sign-language-processing/datasets.git

# pose-format fork, pinned to exact commit for reproducibility
# Cloned without submodules to avoid SSH auth failure on the pose-pipelines submodule.

git clone --no-recurse-submodules https://github.com/GerrySant/pose.git $tools/pose-format-$venv_name

(cd $tools/pose-format-$venv_name && git checkout "c38880312aaefdf07298dce1548ad619734420ba")

pip install $tools/pose-format-$venv_name/src/python

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
    print(m.version("opencv-python-headless"))
EOF
)

pip uninstall -y opencv-python opencv-python-headless
pip install "opencv-python-headless==${OPENCV_VERSION}"

# a missing dependency of etils (tfds)

pip install importlib_resources

