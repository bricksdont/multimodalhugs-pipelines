#! /bin/bash

module load gpumem32gb cuda/13.0.2 cudnn/9.8.0.87-12 miniforge3

environment_scripts=$(dirname "$0")
scripts=$environment_scripts/..
base=$scripts/..

venvs=$base/venvs
tools=$base/tools

mkdir -p $tools

source activate $venvs/huggingface

# install multimodalhugs

git clone https://github.com/GerrySant/multimodalhugs.git $tools/multimodalhugs

# pin commit  https://github.com/GerrySant/multimodalhugs/commit/5201c80f27aa70c460e8297a799dc5daccbd1b3b
# to avoid unintentionally breaking the code

(cd $tools/multimodalhugs && git checkout "5201c80f27aa70c460e8297a799dc5daccbd1b3b")

(cd $tools/multimodalhugs && pip install .)

# install SL datasets

pip install git+https://github.com/sign-language-processing/datasets.git

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

# a missing dependency of etils (tfds)

pip install importlib_resources

pip uninstall -y opencv-python opencv-python-headless
pip install "opencv-python-headless==${OPENCV_VERSION}"
