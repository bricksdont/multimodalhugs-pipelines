#! /bin/bash

set -e

# calling script needs to set:
# $base
# $dry_run
# $model_name
# $dataset
# $feature_type
# $pose_type

base=$1
dry_run=$2
model_name=$3
dataset=$4
feature_type=$5
pose_type=$6

venvs=$base/venvs
configs=$base/configs
configs_sub=$configs/$model_name

models=$base/models
models_sub=$models/$model_name

translations=$base/translations
translations_sub=$translations/$model_name

mkdir -p $translations
mkdir -p $translations_sub

# measure time

SECONDS=0

################################

echo "Python before activating:"
which python

echo "activate path:"
which activate

echo "Executing: source activate $venvs/huggingface"

source activate $venvs/huggingface

echo "Python after activating:"
which python

# necessary?
# export CUDA_VISIBLE_DEVICES=0

# if in doubt, check with:
# echo "CUDA is available:"
# python -c 'import torch; print(torch.cuda.is_available())'

################################

# check if there are any checkpoints

model_name_or_path=$(ls -d "$models_sub"/train/checkpoint-* 2>/dev/null | sort -V | tail -1 || true)

if [ -z "$model_name_or_path" ]; then
  echo "No checkpoints found in $models_sub"
  exit 1
fi

if [[ -s $translations_sub/generated_predictions.txt ]]; then
  echo "Translations exist: $translations_sub/generated_predictions.txt"
  echo "Skipping."
  exit 0
fi

if [[ $dry_run == "true" ]]; then
    use_cpu_arg="--use_cpu"
else
    use_cpu_arg=""
fi

multimodalhugs-generate \
    --task "translation" \
    --config_path $configs_sub/config.yaml \
    --metric_name "sacrebleu" \
    --generate_output_dir $translations_sub \
    --setup_path $models_sub/setup \
    --model_name_or_path $models_sub/train/checkpoint-best \
    --num_beams 5 $use_cpu_arg

echo "time taken:"
echo "$SECONDS seconds"
