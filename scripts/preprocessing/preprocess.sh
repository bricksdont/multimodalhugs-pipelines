#! /bin/bash

set -euo pipefail

# calling script needs to set:
# $base
# $dry_run
# $dataset
# $feature_type
# $pose_type

base=$1
dry_run=$2
dataset=$3
feature_type=$4
pose_type=$5

scripts=$base/scripts
data=$base/data
venvs=$base/venvs

features=$data/$dataset/$feature_type/$pose_type
preprocessed=$data/$dataset/preprocessed/$feature_type/$pose_type

mkdir -p $data
mkdir -p $features $preprocessed

# maybe skip

if [[ -s $preprocessed/train.tsv ]]; then
    echo "Preprocessed files exist in: $preprocessed"
    echo "Skipping"
    exit 0
else
    echo "Preprocessed files do not exist yet"
fi

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

################################

if [[ $dry_run == "true" ]]; then
    dry_run_arg="--dry-run"
else
    dry_run_arg=""
fi

max_attempts=5
attempt=0
until python $scripts/preprocessing/preprocess.py \
    --dataset $dataset \
    --feature-type $feature_type \
    --pose-type $pose_type \
    --feature-dir $features \
    --output-dir $preprocessed \
    --tfds-data-dir $data/tensorflow_datasets $dry_run_arg
do
    attempt=$((attempt + 1))
    if [[ $attempt -ge $max_attempts ]]; then
        echo "Preprocessing failed after $max_attempts attempts, giving up"
        exit 1
    fi
    echo "Attempt $attempt failed, retrying in 60 seconds..."
    sleep 60
done

# sizes
echo "Sizes of preprocessed TSV files:"

wc -l $preprocessed/*.tsv

echo "time taken:"
echo "$SECONDS seconds"
