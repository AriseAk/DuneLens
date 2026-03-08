# DuneLens — Offroad Semantic Segmentation

DuneLens is a semantic segmentation system for desert / offroad UGV imagery. It uses a fine-tuned **SegFormer-B2** as the primary model, with a **CNN baseline** included for comparison. A Next.js frontend provides a live prediction interface backed by a Python inference server.

---

## Table of Contents

- [Project Structure](#project-structure)
- [Classes](#classes)
- [Models](#models)
- [Environment Setup](#environment-setup)
- [Dataset Layout](#dataset-layout)
- [Training](#training)
  - [SegFormer (primary)](#segformer-primary)
  - [CNN (baseline)](#cnn-baseline)
- [Evaluation / Testing](#evaluation--testing)
  - [SegFormer](#segformer-test)
  - [CNN](#cnn-test)
- [Running the Full App](#running-the-full-app)
- [Experiments](#experiments)
- [Training Stats](#training-stats)
- [Configuration Reference](#configuration-reference)

---

## Project Structure

```
ariseak-dunelens/
├── main.py                          # Backend entry point (inference server)
├── backend/
│   ├── Scripts/
│   │   ├── segformer_train.py       # SegFormer-B2 training (primary model)
│   │   ├── segformer_test.py        # SegFormer-B2 evaluation on test set
│   │   ├── CNN_training.py          # CNN baseline training
│   │   ├── CNN_testing.py           # CNN baseline evaluation
│   │   └── ensemble_final.py        # Ensemble inference (SegFormer + CNN)
│   └── train_stats/
│       ├── metrics3_b2_final.txt    # SegFormer-B2 final run metrics
│       ├── metrics_b0.txt           # SegFormer-B0 experiment metrics
│       └── metrics_b2_old.txt       # SegFormer-B2 earlier run metrics
└── frontend/
    ├── app/
    │   ├── page.tsx                 # Home page
    │   ├── about/page.tsx           # About page
    │   └── prediction/page.tsx      # Live prediction interface
    └── components/
        └── navbar.tsx
```

---

## Classes

The model segments images into 11 classes:

| Index | Class           | Raw Pixel Value |
|-------|-----------------|-----------------|
| 0     | Background      | 0               |
| 1     | Trees           | 100             |
| 2     | Lush Bushes     | 200             |
| 3     | Dry Grass       | 300             |
| 4     | Dry Bushes      | 500             |
| 5     | Ground Clutter  | 550             |
| 6     | Flowers         | 600             |
| 7     | Logs            | 700             |
| 8     | Rocks           | 800             |
| 9     | Landscape       | 7100            |
| 10    | Sky             | 10000           |

---

## Models

Pre-trained model weights are hosted on Google Drive. Download the relevant checkpoint(s) and place them in the `backend/` root folder (or update the `MODEL_PATH` / `CKPT_PATH` variables in the relevant scripts to point to a custom location).

| Model | File | Google Drive |
|-------|------|--------------|
| SegFormer-B2 (final) | `best_model.pth` | *https://drive.google.com/drive/folders/1Kb3svtOPlLsey1_5CXUDD8mhn5EXKCFb?usp=sharing* |
| CNN baseline | `best_unet_model.pth` | *https://drive.google.com/drive/folders/1cSzW3QO_VLRYrZqGZxPkO7PdgkVU2jHQ?usp=sharing* |

> **Custom path:** If you prefer to store the weights elsewhere, update `CKPT_PATH` in `segformer_train.py` / `segformer_test.py` and the equivalent variable in the CNN scripts.

---

## Environment Setup

**Python >= 3.10** is required. A conda environment is recommended.

```bash
conda create -n dunelens python=3.11
conda activate dunelens

pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install transformers albumentations opencv-python-headless tqdm matplotlib
```

For the frontend:

```bash
cd frontend
npm install
```

---

## Dataset Layout

Both training and test directories must follow this structure:

```
<split_dir>/
    Color_Images/       ← RGB images (.png)
    Segmentation/       ← Raw label images (.png, same filenames)
```

Update the path constants at the top of each script to point to your local dataset:

```python
# segformer_train.py
TRAIN_DIR = "./Offroad_Segmentation_Training_Dataset/.../train"
VAL_DIR   = "./Offroad_Segmentation_Training_Dataset/.../val"

# segformer_test.py
DATA_DIR  = "./Offroad_Segmentation_testImages/..."
```

---

## Training

### SegFormer (primary)

**Script:** `backend/Scripts/segformer_train.py`

The training pipeline uses SegFormer-B2 with a staged training strategy — the MiT encoder is frozen for the first `WARMUP_EPOCHS` (decoder-only warm-up), then unfrozen for full end-to-end fine-tuning with layer-wise learning rates. Loss is Focal + Dice combined.

```bash
cd backend/Scripts
python segformer_train.py
```

The script will:
1. Train for 20 epochs (3 warm-up + 17 end-to-end by default)
2. Save the best checkpoint by validation IoU to `best_model.pth`
3. Save training plots to `train_stats/all_metrics.png`
4. Save per-epoch metrics to `train_stats/metrics.txt`
5. Run a latency benchmark (50-pass, GPU events) after training

**Resuming from a checkpoint:** If `best_model.pth` already exists in the working directory, the script will automatically resume from where it left off.

Key hyperparameters (edit at the top of the script):

```python
N_EPOCHS      = 20
WARMUP_EPOCHS = 3
BATCH_SIZE    = 2       # effective batch = 8 with ACCUM_STEPS=4
LR            = 6e-5
WEIGHT_DECAY  = 1e-4
FOCAL_GAMMA   = 2.0
IMG_H, IMG_W  = 640, 640
```

---

### CNN (baseline)

**Script:** `backend/Scripts/CNN_training.py`

```bash
cd backend/Scripts
python CNN_training.py
```

Update the dataset paths and hyperparameters at the top of the script as needed. The best checkpoint is saved to the path defined by `CKPT_PATH` in that file.

---

## Evaluation / Testing

### SegFormer Test

**Script:** `backend/Scripts/segformer_test.py`

Runs inference on a test set and reports mean IoU across all images. Predictions are optionally saved to an output directory.

```bash
cd backend/Scripts
python segformer_test.py
```

Or using the training script's built-in eval mode:

```bash
python segformer_train.py --eval
```

Update these variables at the top of `segformer_test.py` before running:

```python
MODEL_PATH = "../best_model.pth"          # path to checkpoint
DATA_DIR   = "../Offroad_Segmentation_testImages/..."
OUTPUT_DIR = "../predictions1"                  # where to save prediction masks
```

---

### CNN Test

**Script:** `backend/Scripts/CNN_testing.py`

```bash
cd backend/Scripts
python CNN_testing.py
```

---

### Ensemble

**Script:** `backend/Scripts/ensemble_final.py`

Combines SegFormer and CNN predictions. Requires both model checkpoints to be available. Update the paths at the top of the script, then:

```bash
cd backend/Scripts
python ensemble_final.py
```

---

## Running the Full App

**Step 1 — Start the backend inference server:**

```bash
cd backend
python ../main.py
```

**Step 2 — Start the frontend:**

```bash
cd frontend
npm run dev
```

The app will be available at `http://localhost:3000`. Upload an image on the Prediction page to get a segmentation mask back from the model.

---

## Experiments

The `experiments/` folder (not tracked in this repo) contains the original Jupyter notebooks used during development:

- SegFormer-B0 experiments (faster, lower accuracy)
- SegFormer-B2 experiments (final model)
- CNN baseline experiments

These notebooks were converted to the clean `.py` scripts in `backend/Scripts/`. The training metrics from these runs are saved in `backend/train_stats/`.

---

## Training Stats

Pre-recorded metrics from past training runs are stored in `backend/train_stats/`:

| File | Description |
|------|-------------|
| `metrics3_b2_final.txt` | SegFormer-B2 final training run — **best results** |
| `metrics_b2_old.txt`    | Earlier B2 run (shorter training / different LR) |
| `metrics_b0.txt`        | SegFormer-B0 run (faster, ~3% lower IoU) |

Each file contains per-epoch loss, IoU, Dice score, and pixel accuracy for both train and validation splits.

---

## Configuration Reference

All key settings live as constants at the top of each script — no config files or CLI flags needed beyond `--eval`.

| Variable | Default | Description |
|----------|---------|-------------|
| `N_CLASSES` | 11 | Number of segmentation classes |
| `IMG_H / IMG_W` | 640 | Input resolution |
| `BATCH_SIZE` | 2 | Per-GPU batch size |
| `ACCUM_STEPS` | 4 | Gradient accumulation steps |
| `N_EPOCHS` | 20 | Total training epochs |
| `WARMUP_EPOCHS` | 3 | Encoder-frozen warm-up epochs |
| `LR` | 6e-5 | Peak learning rate (decoder) |
| `WEIGHT_DECAY` | 1e-4 | AdamW weight decay |
| `FOCAL_GAMMA` | 2.0 | Focal loss gamma |
| `CKPT_PATH` | `best_model_final.pth` | Checkpoint save/load path |
| `OUTPUT_DIR` | `./train_stats` | Directory for plots and metrics |
