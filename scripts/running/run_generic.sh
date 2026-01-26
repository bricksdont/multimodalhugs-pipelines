#! /bin/bash

# Default values for parameters if not set

: "${base:="/shares/sigma.ebling.cl.uzh/mathmu/multimodalhugs-examples"}"
: "${dry_run:="false"}"
: "${model_name:="phoenix"}"
: "${estimator:="mmposewholebody"}"
: "${learning_rate:="5e-05"}"
: "${gradient_accumulation_steps:=1}"
: "${warmup_steps:=0}"
: "${batch_size:=8}"
: "${label_smoothing_factor:="0.0"}"
: "${dataloader_num_workers:=2}"
: "${fp16:="true"}"
: "${seed:=42}"

################################

module load miniforge3 cuda/11.8.0 cudnn/8.7.0.84-11.8

scripts=$base/scripts
logs=$base/logs
logs_sub=$logs/$model_name/$estimator

# logging

mkdir -p $logs $logs_sub

SLURM_DEFAULT_FILE_PATTERN="slurm-%j.out"
SLURM_LOG_ARGS="-o $logs_sub/$SLURM_DEFAULT_FILE_PATTERN -e $logs_sub/$SLURM_DEFAULT_FILE_PATTERN"

echo "##############################################" | tee -a $logs_sub/MAIN
date | tee -a $logs_sub/MAIN
echo "##############################################" | tee -a $logs_sub/MAIN

log_vars() {
  for var in "$@"; do
    # Check if variable is set
    if [[ -v $var ]]; then
      # Use indirect expansion to get value
      echo "${var^^}: ${!var}" | tee -a $logs_sub/MAIN
    fi
  done
}

log_vars base dry_run model_name learning_rate gradient_accumulation_steps warmup_steps batch_size \
    label_smoothing_factor dataloader_num_workers fp16 seed gpu_type

echo "##############################################" | tee -a $logs_sub/MAIN
echo "Using estimator: $estimator"

# SLURM job args
gpu_parameters="--gpus=1 --partition=lowprio"

DRY_RUN_PREPROCESS_SLURM_ARGS="--time=02:00:00 $gpu_parameters --cpus-per-task=2 --mem=16G" # GPU processing is required for pose estimation, even in a dry run
DRY_RUN_TRAINING_SLURM_ARGS="--time=02:00:00 $gpu_parameters --cpus-per-task=2 --mem=32G" # got OOM with 16G (try removing GPU to see if this still runs)
DRY_RUN_GENERIC_SLURM_ARGS="--cpus-per-task=2 --time=02:00:00 --mem=16G"

SLURM_ARGS_PREPROCESS="--time=24:00:00 $gpu_parameters --cpus-per-task=8 --mem=16G"
SLURM_ARGS_TRAIN="--time=24:00:00 $gpu_parameters --cpus-per-task=8 --mem=16G"
SLURM_ARGS_TRANSLATE="--time=12:00:00 $gpu_parameters --cpus-per-task=8 --mem=16G"
SLURM_ARGS_EVALUATE="--time=01:00:00 $gpu_parameters --cpus-per-task=8 --mem=16G"

if [[ $dry_run == "true" ]]; then
  SLURM_ARGS_PREPROCESS=$DRY_RUN_PREPROCESS_SLURM_ARGS # preprocessing with mmpose still currently requires GPU (fix to come)
  SLURM_ARGS_TRAIN=$DRY_RUN_TRAINING_SLURM_ARGS
  SLURM_ARGS_TRANSLATE=$DRY_RUN_GENERIC_SLURM_ARGS
  SLURM_ARGS_EVALUATE=$DRY_RUN_GENERIC_SLURM_ARGS
fi

# preprocess data

id_preprocess=$(
    $scripts/running/sbatch_bare.sh \
    $SLURM_ARGS_PREPROCESS \
    $SLURM_LOG_ARGS \
    $scripts/preprocessing/phoenix_dataset_preprocessing.sh \
    $base $dry_run $estimator
)

echo "  id_preprocess: $id_preprocess | $logs_sub/slurm-$id_preprocess.out" | tee -a $logs_sub/MAIN

# HF train (depends on preprocess)

id_train=$(
    $scripts/running/sbatch_bare.sh \
    $SLURM_ARGS_TRAIN \
    --dependency=afterok:$id_preprocess \
    $SLURM_LOG_ARGS \
    $scripts/training/train_phoenix.sh \
    $base $dry_run $estimator $model_name \
    $learning_rate $gradient_accumulation_steps $warmup_steps $batch_size $label_smoothing_factor \
    $dataloader_num_workers $fp16 $seed 
)

echo "  id_train: $id_train | $logs_sub/slurm-$id_train.out"  | tee -a $logs_sub/MAIN

# HF translate + evaluate (depends on train)

id_translate=$(
    $scripts/running/sbatch_bare.sh \
    $SLURM_ARGS_TRANSLATE \
    --dependency=afterok:$id_train \
    $SLURM_LOG_ARGS \
    $scripts/translation/translate_phoenix.sh \
    $base $dry_run $estimator $model_name
)

echo "  id_translate: $id_translate | $logs_sub/slurm-$id_translate.out"  | tee -a $logs_sub/MAIN

# evaluate (depends on translate)

id_evaluate=$(
    $scripts/running/sbatch_bare.sh \
    $SLURM_ARGS_EVALUATE \
    --dependency=afterok:$id_translate \
    $SLURM_LOG_ARGS \
    $scripts/evaluation/evaluate.sh \
    $base $dry_run $estimator $model_name
)

echo "  id_evaluate: $id_evaluate | $logs_sub/slurm-$id_evaluate.out"  | tee -a $logs_sub/MAIN