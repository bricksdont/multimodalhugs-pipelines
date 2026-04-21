#! /bin/bash

running_scripts=$(dirname "$0")
base=$running_scripts/../..
base=$(realpath $base)
scripts=$base/scripts

# set to "false" or "true":

dry_run="false"

dataset="phoenix"
feature_type="pose"

# best hyperparams found so far

learning_rate="1e-5"
warmup_steps=500
label_smoothing_factor="0.1"
gradient_accumulation_steps=3

for pose_type in alphapose_136 mediapipe mmposewholebody openpifpaf openpose sapiens sdpose smplest_x; do

    model_name="o_brien_et_al_2026+pose_type.$pose_type"

    . $scripts/running/run_generic.sh

done
