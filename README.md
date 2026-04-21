# multimodalhugs-pipelines

Experiment management code for training models with [MultimodalHugs](https://github.com/GerrySant/multimodalhugs) (MMH). MMH is an extension of Huggingface offering better support for sign language processing research.

## Basic setup

Download the code:

    git clone https://github.com/bricksdont/multimodalhugs-pipelines
    cd multimodalhugs-pipelines

Create a venv:

    ./scripts/environment/create_env.sh

Then install required software:

    ./scripts/environment/install.sh

## Run experiments

### Basic experiment

It is a good idea to use `dry_run="true"` the first time you are running code, which creates all files, executes all
the code but uses only a fraction of the training data, trains for very few steps only, etc - as a
general sanity check. If you want to launch a real run after a dry run you will need to manually
delete folders that the dry run created (e.g. a sub-folder of `models`), otherwise the steps
will not be repeated.

Then to train a basic model:

    ./scripts/running/run_basic.sh

This will first download and prepare the PHOENIX training data,
and then train a basic MultimodalHugs model. All steps are submitted
as SLURM jobs.

If the process is fully reproducible, this should result in a test set BLEU score of `10.691`. This
value is inside the file `evaluations/phoenix/test_score.bleu`.

### Further experiments

Actual experiments are in the [experiments](experiments) folder, each with its own README.

| Experiment                                                            | Description                                        |
|-----------------------------------------------------------------------|----------------------------------------------------|
| [Hyperparameter search](experiments/hyperparameter_search/README.md) | Train ~50 models to find good hyperparameters      |
| [Reproducibility](experiments/reproducibility/README.md)             | Train three identical models to test repeatability |
| [O'Brien et al. 2026](experiments/o_brien_et_al_2026/README.md)      | Compare 8 pose estimators on Phoenix               |
