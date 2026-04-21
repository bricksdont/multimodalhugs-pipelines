#! /bin/bash

# calling script needs to set

# $hyp
# $ref
# $output

if [[ ! -s $output ]]; then

    cat $hyp | sacrebleu $ref -w 3 > $output

    echo "$output"
    cat $output

fi
