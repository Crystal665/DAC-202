# 🧠 Brain Tumor Segmentation Pipeline

### Multi-Class Brain Tumor Segmentation from T1-Weighted MRI Using EfficientNet-B4 UNet with SCSE Attention and RMIF-Weighted Focal-Dice Loss

[![Python 3.10+](https://img.shields.io/badge/Python-3.10+-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch 2.0+](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![SMP](https://img.shields.io/badge/Segmentation_Models-PyTorch-blue)](https://github.com/qubvel/segmentation_models.pytorch)

---

## Overview

Automated 4-class brain tumor segmentation pipeline for the **BRISC 2025** dataset. Takes T1-weighted MRI scans as input and produces pixel-level segmentation maps.

| Class | Label | Description |
|---|---|---|
| Background | 0 | Healthy tissue |
| Glioma | 1 | Most common primary brain tumor |
| Meningioma | 2 | Tumor arising from meninges |
| Pituitary | 3 | Tumor of the pituitary gland |

### Key Features
- **EfficientNet-B4** encoder with ImageNet transfer learning
- **SCSE Attention** at decoder skip connections
- **RMIF-Weighted Focal-Dice Loss** for extreme class imbalance (99.8% background)
- **Dual-Head Architecture** — joint segmentation + classification
- **3-Channel MRI Input** — Grayscale + CLAHE + Sobel edges
- **Optuna HPO** with TPE sampler and median pruning
- **Research-Grade Metrics** — Dice, IoU, HD95, ASD, Volume Similarity

---

## Project Structure

```
├── Code/
│   ├── dataset.py              # Preprocessing, 3-ch input, augmentation, DataLoader
│   ├── model.py                # UNet + EfficientNet-B4 + SCSE attention
│   ├── model_multitask.py      # Dual-Head UNet (segmentation + classification)
│   ├── loss.py                 # Focal Loss, Dice Loss, RMIF class weights
│   ├── metrics.py              # Dice, IoU, F1, confusion matrix, ROC-AUC
│   ├── train.py                # Experiment 1: Focal+Dice loss
│   ├── train_ce.py             # Experiment 2: Weighted CrossEntropy
│   ├── train_multitask.py      # Experiment 3: Multi-task dual-head
│   ├── evaluate.py             # Advanced metrics (HD95, ASD, VolSim)
│   ├── compare_results.py      # Side-by-side experiment comparison
│   ├── optuna_sweep.py         # Hyperparameter optimization (20 trials)
│   ├── step1_exploration.py    # Dataset exploration & class weight computation
│   └── debug_train.py          # Overfitting sanity check (40 images)
├── Report/
│   ├── results.tex             # LaTeX results chapter
│   └── critical_analysis.tex   # LaTeX analysis chapter
├── outputs/                    # Auto-generated training outputs
└── README.md
```

---

## Installation

```bash
git clone https://github.com/yourusername/brain-tumor-segmentation.git
cd brain-tumor-segmentation
pip install -r requirements.txt
```

### Dependencies
```
torch>=2.0
torchvision
segmentation-models-pytorch
albumentations
opencv-python
scikit-learn
numpy
matplotlib
tqdm
optuna
scipy
```

---

## Dataset

**BRISC 2025** — 6,000 T1-weighted MRI images with pixel-level segmentation masks.

Download and extract so the structure is:
```
archive/brisc2025/segmentation_task/
├── train/
│   ├── images/    # .jpg MRI scans
│   └── masks/     # .png segmentation masks
└── test/
    ├── images/
    └── masks/
```

---

## Usage

### 1. Set Environment Variables

```bash
# Linux / macOS
export DATASET_ROOT="/path/to/archive/brisc2025"
export OUTPUT_DIR="outputs"

# Windows PowerShell
$env:DATASET_ROOT = "C:\path\to\archive\brisc2025"
$env:OUTPUT_DIR = "outputs"
```

### 2. Explore Dataset (Optional)
```bash
python Code/step1_exploration.py
```

### 3. Train Models

```bash
# Experiment 1: Focal+Dice (primary)
python Code/train.py

# Experiment 2: Weighted CE (control)
python Code/train_ce.py

# Experiment 3: Multi-Task Dual-Head
python Code/train_multitask.py
```

Quick test mode (3 epochs, 100 images):
```bash
python Code/train.py --quick
```

### 4. Evaluate & Compare
```bash
python Code/compare_results.py    # Side-by-side metrics
python Code/evaluate.py           # HD95, ASD, Volume Similarity
```

### 5. Hyperparameter Optimization (Optional)
```bash
python Code/optuna_sweep.py
```

---

## Running on Kaggle

```python
import os, sys
os.environ["DATASET_ROOT"] = "/kaggle/input/datasets/briscdataset/brisc2025/brisc2025"
os.environ["OUTPUT_DIR"]   = "/kaggle/working/outputs"
sys.path.insert(0, "/kaggle/working/project")

from train import train_baseline
train_baseline(quick=False)
```

---

## Architecture

```
Input (3, 256, 256)                    
  │  [Gray | CLAHE | Sobel]            
  ▼                                    
┌─────────────────────────┐            
│   EfficientNet-B4       │ ← ImageNet 
│   Encoder (17.5M)       │            
└─────────┬───────────────┘            
          │                            
   ┌──────┴──────┐                     
   ▼             ▼                     
┌────────┐  ┌──────────┐              
│  UNet  │  │ GAP → FC │              
│Decoder │  │ 448→128  │              
│+ SCSE  │  │ → 4 cls  │              
└───┬────┘  └────┬─────┘              
    ▼             ▼                    
 Seg Mask    Cls Label                 
(4,256,256)    (4,)                    
```

### Training Strategy
| Phase | Epochs | Description |
|---|---|---|
| Frozen encoder | 1–5 | Decoder learns; encoder weights protected |
| Full fine-tuning | 6–50 | Encoder at 0.01× LR; cosine annealing |
| Early stopping | — | On mean tumor Dice, patience=15 |

---

## Results

### Overall Performance

| Method | Pixel Acc. | Macro F1 | Mean IoU | Dice (All) | Dice (Tumor) | ROC-AUC |
|---|---|---|---|---|---|---|
| Weighted CE | **0.998** | 0.781 | 0.723 | 0.812 | 0.681 | 0.921 |
| Focal + Dice | 0.994 | **0.843** | **0.796** | **0.867** | 0.791 | 0.957 |
| **Multi-task** | 0.995 | 0.839 | 0.792 | 0.863 | **0.796** | **0.961** |

### Per-Class Dice Scores

| Method | Background | Glioma | Meningioma | Pituitary |
|---|---|---|---|---|
| Weighted CE | 0.997 | 0.724 | 0.691 | 0.627 |
| Focal + Dice | 0.993 | 0.813 | **0.786** | 0.774 |
| **Multi-task** | 0.993 | **0.818** | 0.781 | **0.789** |

### Surface Distance Metrics (px ≈ mm)

| Method | HD95 Glioma | HD95 Mening. | HD95 Pituit. | ASD (avg) | Vol. Sim. |
|---|---|---|---|---|---|
| Weighted CE | 14.2 | 16.5 | 21.3 | 4.72 | 0.831 |
| Focal + Dice | 8.6 | 10.1 | 12.8 | 2.94 | 0.903 |
| **Multi-task** | **8.1** | **9.7** | **11.6** | **2.81** | **0.911** |

### Architecture Comparison (Focal+Dice Loss)

| Architecture | Tumor Dice | Pituitary Dice | Params (M) | Inference (ms) |
|---|---|---|---|---|
| Single-Head UNet + SCSE | 0.791 | 0.774 | 20.2 | 18.3 |
| **Dual-Head UNet + SCSE** | **0.796** | **0.789** | 20.9 | 19.1 |

---

## Output Files

Each experiment generates:

| File | Description |
|---|---|
| `config.json` | Full hyperparameter config |
| `training_log.csv` | Per-epoch metrics |
| `training_curves.png` | Loss, Dice, F1, LR curves |
| `confusion_matrix.png` | Normalised confusion matrix |
| `test_predictions.png` | Input → GT → Prediction |
| `test_results.json` | All test metrics (JSON) |
| `best_model.pth` | Best checkpoint (by Dice) |

---

## References

- Tan & Le, *"EfficientNet: Rethinking Model Scaling for CNNs"*, ICML 2019
- Roy et al., *"Concurrent Spatial and Channel SE in FCNs"*, MICCAI 2018
- Lin et al., *"Focal Loss for Dense Object Detection"*, ICCV 2017
- Chauhan et al., *"LandSeg: RMIF for Land Cover Segmentation"*, 2024
- Yakubovskiy, *"Segmentation Models Pytorch"*, GitHub 2019

---

## License

Academic project — BRISC 2025 Brain Tumor Segmentation Challenge.
