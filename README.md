# rPPG Training using PhysNet (3D CNN)

🚀 Train a PhysNet-style 3D CNN for remote photoplethysmography (rPPG) using facial video.

![Python](https://img.shields.io/badge/Python-3.10+-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-DeepLearning-orange)
![License](https://img.shields.io/badge/License-MIT-purple)

---

## 📌 Overview

This project trains a **PhysNet-style 3D CNN** to extract rPPG signals from facial video and estimate physiological signals such as heart rate.

---

## ✨ Features

- 🧠 Train an rPPG model on UBFC Dataset 1  
- 🎥 Extract per-frame rPPG signal from face ROI video  
- ❤️ Estimate Heart Rate (BPM) using frequency analysis  
- 🩸 Estimate SpO2 proxy using AC/DC ratio  
- 📈 Training logs and visualization outputs  

---

## 📁 Project Structure

```bash
.
├── trainrppg.py        # Training, evaluation, CLI inference
├── checkpoints/        # Saved models
├── logs/               # Training logs & plots
└── data/               # Dataset directory
```

---

## 📦 Dataset Layout (Expected)

```bash
data/
└── UBFC_DATASET/
    └── DATASET_1/
        ├── 5-gt/
        │   ├── vid.avi
        │   └── gtdump.xmp
        ├── 6-gt/
        │   ├── vid.avi
        │   └── gtdump.xmp
        └── ...
```

Each subject folder must contain:

- `vid.avi` → video file  
- `gtdump.xmp` → ground truth signals  

Expected columns in `gtdump.xmp`:

1. timestep (ms)  
2. heart rate  
3. SpO2  
4. PPG signal  

---

## ⚙️ Requirements

Install dependencies:

```bash
pip install tensorflow keras opencv-python scipy scikit-learn matplotlib
```

Or:

```bash
pip install -r requirements.txt
```

---

## 🚀 Quick Start

### 1. Train the model

```bash
python trainrppg.py --mode train
```

---

## 🧠 Training Outputs

- ✅ Best model → `checkpoints/best_physnet.keras`  
- 📄 Logs → `logs/training_log.csv`  
- 📈 Curves → `logs/training_curves.png`  

---

## 💻 Script Inference (Optional)

Run one-off inference using the training script:

```bash
python trainrppg.py --mode infer \
  --model checkpoints/best_physnet.keras \
  --video path/to/video.avi
```

Outputs:

- Console: BPM & SpO2  
- Plot: `rppg_prediction.png`  

---

## ⚠️ Notes

- Video must have enough frames (default: 128 frames)  
- SpO2 is an approximation (not medically accurate)  
- Parser may need adjustment if `gtdump.xmp` contains metadata  

---

## 🛠️ Troubleshooting

**Model not found**
- Ensure `checkpoints/best_physnet.keras` exists  

**No frames read**
- Check video path and format  
- Ensure OpenCV supports decoding  

**Video too short**
- Use longer clips  
- Or reduce `CLIP_LEN` consistently  

---

## 📄 License


MIT License
