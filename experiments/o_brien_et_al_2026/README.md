# O'Brien et al. 2026: Comparing Pose Estimators for Sign Language Translation

This experiment trains one model per pose estimator on the Phoenix dataset and compares their BLEU scores, following the setup of O'Brien et al. (2026).

## Pose types

Eight pose estimators are evaluated:

| Pose type | Keypoints | Format | Expected feat_dim |
|---|---|---|---|
| `alphapose_136` | 136 | XY (2D) | 272 |
| `mediapipe` | 534 (after reduction) | XYZC | 534 |
| `mmposewholebody` | 133 | XY | 266 |
| `openpifpaf` | 133 | XY | 266 |
| `openpose` | 137 | XY | 274 |
| `sapiens` | — | — | 620 |
| `sdpose` | 133 | XYC | 399 |
| `smplest_x` | 139 | XY | 278 |

Pre-estimated poses are downloaded automatically from Cloudflare during preprocessing.
See [`POSE_DOWNLOAD_URLS`](../../scripts/preprocessing/preprocess.py) for the exact URLs.

## How to run

```bash
bash scripts/running/run_o_brien_et_al_2026.sh
```

This submits 8 × 3 × 4 SLURM jobs (preprocess → train → translate → evaluate) with model names of the form `o_brien_et_al_2026+pose_type.<pose_type>+seed.<seed>`.

## Hyperparameters

Best hyperparameters from the hyperparameter search experiment are used:

- `learning_rate`: 1e-5
- `warmup_steps`: 500
- `label_smoothing_factor`: 0.1
- `gradient_accumulation_steps`: 3

## Results

| Pose type | Seed | BLEU |
|---|---|---|
| `alphapose_136` | 375678 | — |
| `alphapose_136` | 534 | — |
| `alphapose_136` | 42 | — |
| `mediapipe` | 375678 | — |
| `mediapipe` | 534 | — |
| `mediapipe` | 42 | — |
| `mmposewholebody` | 375678 | — |
| `mmposewholebody` | 534 | — |
| `mmposewholebody` | 42 | — |
| `openpifpaf` | 375678 | — |
| `openpifpaf` | 534 | — |
| `openpifpaf` | 42 | — |
| `openpose` | 375678 | — |
| `openpose` | 534 | — |
| `openpose` | 42 | — |
| `sapiens` | 375678 | — |
| `sapiens` | 534 | — |
| `sapiens` | 42 | — |
| `sdpose` | 375678 | — |
| `sdpose` | 534 | — |
| `sdpose` | 42 | — |
| `smplest_x` | 375678 | — |
| `smplest_x` | 534 | — |
| `smplest_x` | 42 | — |
