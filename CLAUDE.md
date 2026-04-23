# CLAUDE.md

## Project overview

SLURM-based pipeline for pose-to-text sign language translation using the [multimodalhugs](https://github.com/GerrySant/multimodalhugs) framework. Runs on the university cluster (`sigma.ebling.cl.uzh`). The typical experiment trains an M2M-100 backbone with a linear multimodal mapper on mediapipe holistic pose features extracted from a sign language dataset (currently Phoenix).

## Repository layout

```
scripts/
  environment/   install.sh
  running/       run_basic.sh, run_generic.sh (SLURM orchestrator), sbatch_bare.sh
  preprocessing/ preprocess.sh + preprocess.py
  training/      train.sh + create_config.py
  translation/   translate.sh
  evaluation/    evaluate.sh, evaluate_bleu_generic.sh, evaluate_bleurt_generic.sh
  summaries/     summarize.py / summarize.sh
  debugging/     debug.sh
tests/
  test_pipeline.sh   end-to-end CI test using fake pose data
.github/workflows/ci.yml
```

## Entry points

- **`scripts/running/run_basic.sh`** — set experiment variables, sources `run_generic.sh`
- **`scripts/running/run_generic.sh`** — submits the SLURM chain: preprocess → train → translate → evaluate

## Key variables

All configurable at the top of `run_basic.sh` / `run_generic.sh`:

| Variable | Default | Meaning |
|---|---|---|
| `base` | `/shares/.../multimodalhugs-examples` | Root directory for all data, models, logs |
| `dataset` | `phoenix` | Dataset name (maps to TFDS identifier in `preprocess.py`) |
| `feature_type` | `pose` | Feature modality; used in directory paths and `--modality` flag |
| `pose_type` | `mediapipe` | Pose extractor; triggers `--reduce-holistic-poses` in training |
| `model_name` | `phoenix` | Experiment name; scopes config/model/log directories |
| `dry_run` | `false` | If `true`, runs on CPU with tiny steps; skips GPU allocation |

## Directory conventions

- Raw features: `$base/data/$dataset/$feature_type/$pose_type/`
- Preprocessed TSVs: `$base/data/$dataset/preprocessed/$feature_type/$pose_type/{train,validation,test}.tsv`
- Configs: `$base/configs/$model_name/config.yaml`
- Models: `$base/models/$model_name/`
- Logs: `$base/logs/$model_name/`
- HuggingFace cache: `$base/data/huggingface/` (set via `HF_HOME`)
- TFDS cache: `$base/data/tensorflow_datasets/`

## Pipeline scripts

Each step script (`preprocess.sh`, `train.sh`, `translate.sh`) follows the same pattern:

- `set -euo pipefail` at the top — required so SLURM `afterok` dependencies fail correctly when a step errors
- Positional args only (no named flags), in a fixed documented order
- Skip logic at the top: if output already exists, exits 0 immediately
- Activates the venv via `source activate $venvs/${venv:-huggingface}`
- Times execution with `$SECONDS`

## Pose features

- Pose type `mediapipe` uses MediaPipe holistic format: 5 components (`POSE_LANDMARKS`, `FACE_LANDMARKS`, `LEFT_HAND_LANDMARKS`, `RIGHT_HAND_LANDMARKS`, `POSE_WORLD_LANDMARKS`), format `XYZC`, 33/468/21/21/33 points respectively
- `reduce_holistic_poses: true` in the config (set automatically when `pose_type == mediapipe`) reduces face mesh points; output dimension is 534 — this is what `feat_dim` in the YAML config must match
- `reduce_holistic_poses` is applied by the multimodalhugs processor at runtime, not during preprocessing

## multimodalhugs dependency

- The default `install.sh` installs the latest version; experiment-specific install scripts (e.g. `experiments/o_brien_et_al_2026/install.sh`) pin to an exact commit for reproducibility
- Pinned to commit `5201c80f27aa70c460e8297a799dc5daccbd1b3b` in `ci.yml` and experiment install scripts
- Exposes CLI tools: `multimodalhugs-setup`, `multimodalhugs-train`, `multimodalhugs-generate`
- Set `HF_HUB_DISABLE_XET=1` before calling these (see [issue #50](https://github.com/GerrySant/multimodalhugs/issues/50))
- Transformers 4.44 warns about `reduce_holistic_poses` not being in `valid_kwargs` — this is cosmetic; the processor still applies the reduction correctly

## Dependency install notes

- `tf-keras` must be installed separately because Keras 3 is not supported by Transformers
- `opencv-python` must be replaced with `opencv-python-headless` because OpenGL is unavailable on the cluster; `install.sh` detects the installed version and reinstalls the headless variant
- `importlib_resources` must be installed explicitly — it is a missing transitive dependency of `etils`/`tensorflow_datasets` in Python 3.11
- mediapipe is **not** installed; fake pose generation in `preprocess.py` hardcodes MediaPipe holistic component definitions directly to avoid the mediapipe/protobuf/tensorflow version conflict
- TF/TFDS are not used for Phoenix text labels; labels are downloaded directly from `ANNOTATIONS_URL` in `preprocess.py` as a plain 800 KB tar.gz, parsed with the `csv` module — no protobuf pin needed
- `pose-format` is installed from the `GerrySant/pose` fork (`multiple_support` branch) to support non-mediapipe pose types; the default install tracks the branch tip, experiment installs pin to an exact commit
- Each experiment that requires a reproducible environment has its own `install.sh` under `experiments/<name>/` which creates `venvs/<name>/`; `run_generic.sh` picks up the venv via the `$venv` variable (default: `huggingface`)

## CI

- `tests/test_pipeline.sh` runs the full pipeline (preprocess → config → setup → train → translate → BLEU) using `--fake` pose data; no real dataset download
- `.github/workflows/ci.yml` is intentionally thin; all logic lives in the test script
- HuggingFace model weights are cached in CI under key `hf-m2m100-418M`

## Known pitfalls

- The preprocessing step downloads ~5.5 GB from the RFDS server; compute nodes have flaky network connections. `preprocess.sh` retries the Python call up to 5 times with a 60-second pause; TFDS resumes partial downloads automatically from `$base/data/tensorflow_datasets/`
- SLURM `afterok` dependencies only work correctly if the upstream job exits non-zero on failure — hence `set -euo pipefail` in every step script
- GPU type is set via `gpu_type` variable (`v100`, `h100`, or anything else). L4 nodes have insufficient memory and are excluded via `--constraint=GPUMEM32GB` for the generic fallback
