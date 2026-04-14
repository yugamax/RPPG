# rPPG Pipeline (UBFC Dataset 2)

This project preprocesses UBFC videos into training-ready RGB/PPG sequences and trains a 1D CNN-based rPPG model.

## 1) Required Folder Structure

Set up the project exactly like this:

```text
RPPG/
|-- preprocess.py
|-- trainrppg.py
|-- requirements.txt
|-- README.md
|-- UBFC_DATASET/
|   `-- DATASET_2/
|       |-- subject1/
|       |   |-- vid.avi
|       |   `-- ground_truth.txt
|       |-- subject2/
|       |   |-- vid.avi
|       |   `-- ground_truth.txt
|       `-- ...
|-- processed/                  (auto-created by preprocess.py)
|   |-- subject1_rgb.npy
|   |-- subject1_ppg.npy
|   |-- subject1_fps.npy
|   `-- ...
|-- checkpoint.pth              (auto-created during training)
`-- best_model.pth              (auto-created during training)
```

Important:
- `preprocess.py` expects `DATA_ROOT = "UBFC_DATASET/DATASET_2"`.
- Each subject folder must contain both `vid.avi` and `ground_truth.txt`.

## 2) Environment Setup (Windows PowerShell)

From the project root:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If you use Conda, create and activate your environment first, then run:

```powershell
pip install -r requirements.txt
```

## 3) Preprocessing

Run:

```powershell
python preprocess.py
```

What it does:
- Detects face landmarks (MediaPipe FaceMesh).
- Builds forehead + cheek ROIs.
- Extracts mean RGB per frame.
- Aligns ground-truth PPG to video timestamps.
- Applies bandpass filtering and normalization.
- Saves per-subject arrays in `processed/`.

Expected outputs in `processed/` for each subject:
- `<subject>_rgb.npy` shape `(T, 3)`
- `<subject>_ppg.npy` shape `(T,)`
- `<subject>_fps.npy` shape `(1,)`

## 4) Training

Run:

```powershell
python trainrppg.py
```

Training behavior:
- Reads all `*_rgb.npy` files from `processed/`.
- Splits subjects 80/20 into train/validation.
- Trains TSCAN-like 1D model with Pearson loss.
- Uses mixed precision (autocast + GradScaler).
- Saves:
  - `checkpoint.pth` every epoch
  - `best_model.pth` on best validation loss
- Supports resume from `checkpoint.pth` automatically.

## 5) Configuration Knobs

Edit constants in `trainrppg.py`:
- `SEQ_LEN`
- `BATCH_SIZE`
- `LR`
- `EPOCHS`
- `PATIENCE`
- `NUM_WORKERS`

Edit constants in `preprocess.py`:
- `DATA_ROOT`
- `SAVE_DIR`

## 6) Quick Validation Checklist

Before preprocessing:
- Confirm dataset path exists: `UBFC_DATASET/DATASET_2`.
- Confirm each subject has both required files.

After preprocessing:
- Confirm `processed/` is populated.
- Confirm each subject has all three `.npy` files.

Before training:
- Confirm at least a few subjects exist in `processed/`.
- Ensure GPU is available if you want CUDA acceleration.

## 7) Common Issues and Fixes

1. `Dataset path not found`
- Fix `DATA_ROOT` in `preprocess.py` or correct folder placement.

2. `No subject folders found. Nothing to preprocess.`
- Check that subject directories are directly under `UBFC_DATASET/DATASET_2`.

3. Many `[SKIP]` lines during preprocessing
- Subject may be missing files, have malformed ground truth, too-short videos, or invalid FPS.

4. `FileNotFoundError` for `processed/` during training
- Run preprocessing first.

5. Out-of-memory during training
- Lower `BATCH_SIZE` and/or `SEQ_LEN`.

6. CPU is too slow
- Reduce `NUM_WORKERS` if system is overloaded, or use a CUDA-enabled setup.

## 8) Minimal Run Order

```powershell
python preprocess.py
python trainrppg.py
```

That is the full pipeline.
