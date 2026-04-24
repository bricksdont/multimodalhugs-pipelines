#! /bin/bash

summary_scripts=$(dirname "$0")
scripts=$summary_scripts/..
base=$scripts/..
base=$(realpath $base)

venvs=$base/venvs
scripts=$base/scripts
evaluations=$base/evaluations

summaries=$base/summaries

mkdir -p $summaries

grep "\"score\"" $evaluations/*/test_score.bleu | awk -F'"score": ' '{print $2 "\t" $0}' | sort -k1,1nr | cut -f2- \
    > $summaries/summary.txt

cat $summaries/summary.txt
