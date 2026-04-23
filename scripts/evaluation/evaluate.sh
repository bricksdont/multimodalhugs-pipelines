#! /bin/bash

set -euo pipefail

# calling process needs to set:
# base
# $dry_run
# $model_name

base=$1
dry_run=$2
model_name=$3

data=$base/data
scripts=$base/scripts
venvs=$base/venvs

translations=$base/translations
translations_sub=$translations/$model_name

evaluations=$base/evaluations
evaluations_sub=$evaluations/$model_name

mkdir -p $evaluations $evaluations_sub

################################

echo "Python before activating:"
which python

echo "activate path:"
which activate

# perhaps not necessary anymore
# eval "$(conda shell.bash hook)"

echo "Executing: source activate $venvs/${venv:-default}"

source activate $venvs/${venv:-default}

echo "Python after activating:"
which python

################################

# avoid downloading metric files (e.g. BLEURT model) to ~/.cache/huggingface

export HF_HOME=$data/huggingface

# extract refs

sed -n 's/^L \[[0-9]\+\]\s*//p' $translations_sub/predictions_labels.txt > $translations_sub/labels.txt

hyp=$translations_sub/generated_predictions.txt
ref=$translations_sub/labels.txt

output=$evaluations_sub/test_score.bleu

. $scripts/evaluation/evaluate_bleu_generic.sh

output=$evaluations_sub/test_score.bleurt

if [[ $dry_run == "true" ]]; then
    bleurt_checkpoint="bleurt-tiny-128"
else
    bleurt_checkpoint="BLEURT-20"
fi

. $scripts/evaluation/evaluate_bleurt_generic.sh
