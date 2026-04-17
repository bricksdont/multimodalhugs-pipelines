#! /bin/bash

set -e

scripts=$(dirname "$0")/../scripts
scripts=$(realpath $scripts)

BASE=${BASE:-/tmp/multimodalhugs_pipeline_test}

DATASET=phoenix
FEATURE_TYPE=pose
POSE_TYPE=mediapipe
MODEL_NAME=ci_test

mkdir -p $BASE/data/$DATASET/$FEATURE_TYPE/$POSE_TYPE
mkdir -p $BASE/data/$DATASET/preprocessed/$FEATURE_TYPE/$POSE_TYPE
mkdir -p $BASE/configs/$MODEL_NAME
mkdir -p $BASE/models/$MODEL_NAME
mkdir -p $BASE/translations/$MODEL_NAME
mkdir -p $BASE/evaluations/$MODEL_NAME

export HF_HUB_DISABLE_XET=1
export HF_HOME=$BASE/huggingface

echo "##############################################"
echo "Step 1: Preprocess (fake data)"
echo "##############################################"

python $scripts/preprocessing/preprocess.py \
    --dataset $DATASET \
    --feature-type $FEATURE_TYPE \
    --pose-type $POSE_TYPE \
    --fake \
    --feature-dir $BASE/data/$DATASET/$FEATURE_TYPE/$POSE_TYPE \
    --output-dir $BASE/data/$DATASET/preprocessed/$FEATURE_TYPE/$POSE_TYPE

echo "##############################################"
echo "Step 2: Create config"
echo "##############################################"

python $scripts/training/create_config.py \
    --run-name $DATASET \
    --config-dir $BASE/configs/$MODEL_NAME \
    --train-metadata-file $BASE/data/$DATASET/preprocessed/$FEATURE_TYPE/$POSE_TYPE/train.tsv \
    --validation-metadata-file $BASE/data/$DATASET/preprocessed/$FEATURE_TYPE/$POSE_TYPE/validation.tsv \
    --test-metadata-file $BASE/data/$DATASET/preprocessed/$FEATURE_TYPE/$POSE_TYPE/test.tsv \
    --new-vocabulary "__dgs__" \
    --reduce-holistic-poses \
    --dry-run

echo "##############################################"
echo "Step 3: Setup model"
echo "##############################################"

multimodalhugs-setup \
    --modality ${FEATURE_TYPE}2text \
    --config_path $BASE/configs/$MODEL_NAME/config.yaml \
    --output_dir $BASE/models/$MODEL_NAME \
    --seed 42

echo "##############################################"
echo "Step 4: Train (dry run, CPU)"
echo "##############################################"

multimodalhugs-train \
    --task translation \
    --config_path $BASE/configs/$MODEL_NAME/config.yaml \
    --setup_path $BASE/models/$MODEL_NAME/setup \
    --output_dir $BASE/models/$MODEL_NAME \
    --seed 42 \
    --report_to none \
    --use_cpu

echo "##############################################"
echo "Step 5: Translate (dry run, CPU)"
echo "##############################################"

CHECKPOINT=$(ls -d $BASE/models/$MODEL_NAME/train/checkpoint-* 2>/dev/null | sort -V | tail -1)

multimodalhugs-generate \
    --task translation \
    --config_path $BASE/configs/$MODEL_NAME/config.yaml \
    --metric_name sacrebleu \
    --generate_output_dir $BASE/translations/$MODEL_NAME \
    --setup_path $BASE/models/$MODEL_NAME/setup \
    --model_name_or_path $CHECKPOINT \
    --num_beams 5 \
    --use_cpu

echo "##############################################"
echo "Step 6: Evaluate BLEU"
echo "##############################################"

sed -n 's/^L \[[0-9]\+\]\s*//p' \
    $BASE/translations/$MODEL_NAME/predictions_labels.txt \
    > $BASE/translations/$MODEL_NAME/labels.txt

cat $BASE/translations/$MODEL_NAME/generated_predictions.txt \
    | sacrebleu $BASE/translations/$MODEL_NAME/labels.txt -w 3 \
    > $BASE/evaluations/$MODEL_NAME/test_score.bleu

cat $BASE/evaluations/$MODEL_NAME/test_score.bleu

echo "##############################################"
echo "Pipeline test complete."
echo "##############################################"
