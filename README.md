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

### Overall Performance (Table 7)

| Model | Pix. Acc. (%) | Macro F1 | Wtd. F1 | Dice (Tumor) | mIoU | ROC-AUC |
|---|---|---|---|---|---|---|
| CE (baseline) | **99.21** | 0.431 | 0.985† | 0.357 | 0.421 | 0.871 |
| Focal+Dice | 98.84 | 0.612 | 0.982 | 0.604 | 0.574 | 0.921 |
| **Multi-Task (MT)** | 98.94 | **0.628** | 0.983 | **0.624** | **0.589** | **0.934** |

> † Weighted F1 dominated by background class; clinically uninformative.

### Per-Class Dice & IoU (Table 8)

| Model | Dice BG | Dice Glioma | Dice Mening. | Dice Pituit. | IoU BG | IoU Glioma | IoU Mening. | IoU Pituit. |
|---|---|---|---|---|---|---|---|---|
| CE (baseline) | 0.992 | 0.521 | 0.398 | 0.152 | 0.984 | 0.352 | 0.249 | 0.082 |
| Focal+Dice | 0.988 | 0.713 | 0.618 | 0.481 | 0.976 | 0.554 | 0.448 | 0.316 |
| **Multi-Task** | 0.989 | **0.728** | **0.631** | **0.513** | 0.978 | **0.572** | **0.461** | **0.345** |

### Boundary Accuracy (Table 9, pixels)

| Class | Model | HD | HD95 | ASD | Vol. Sim. |
|---|---|---|---|---|---|
| Glioma | F+D | 38.4 | 18.7 | 4.21 | 0.881 |
| Glioma | **MT** | **35.1** | **16.2** | **3.84** | **0.904** |
| Meningioma | F+D | 51.6 | 26.3 | 6.78 | 0.843 |
| Meningioma | **MT** | **48.9** | **23.8** | **6.14** | **0.861** |
| Pituitary | F+D | 62.3 | 34.7 | 9.12 | 0.712 |
| Pituitary | **MT** | **54.8** | **28.1** | **7.63** | **0.762** |

### Architecture Ablation (Table 10, Focal+Dice Loss)

| Model | Dice (Tumor) | Pituit. Dice | Macro F1 | HD95 | ASD | ROC-AUC |
|---|---|---|---|---|---|---|
| Single-Head | 0.604 | 0.481 | 0.612 | 26.6 | 6.70 | 0.921 |
| **Dual-Head (MT)** | **0.624** | **0.513** | **0.628** | **22.7** | **5.87** | **0.934** |

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
