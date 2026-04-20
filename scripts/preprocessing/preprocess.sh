#! /bin/bash

set -e

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

python $scripts/preprocessing/preprocess.py \
    --dataset $dataset \
    --feature-type $feature_type \
    --pose-type $pose_type \
    --feature-dir $features \
    --output-dir $preprocessed \
    --tfds-data-dir $data/tensorflow_datasets $dry_run_arg

# sizes
echo "Sizes of preprocessed TSV files:"

wc -l $preprocessed/*.tsv

echo "time taken:"
echo "$SECONDS seconds"
