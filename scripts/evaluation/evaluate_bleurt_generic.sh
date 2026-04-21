#! /bin/bash

# calling script needs to set

# $scripts
# $hyp
# $ref
# $output
# $bleurt_checkpoint

if [[ ! -s $output ]]; then

    # avoid TF JIT compiler error

    export XLA_FLAGS=--xla_gpu_cuda_data_dir=$(dirname $(dirname $(which nvcc)))

    python $scripts/evaluation/evaluate_bleurt.py \
        --references $ref \
        --predictions $hyp \
        --checkpoint $bleurt_checkpoint \
        > $output

    echo "$output"
    cat $output

fi
