# Comparative Analysis of Deep Learning Models for Agricultural Land Potential Mapping Using Multispectral Satellite Imagery

Official repository for the paper accepted at **FDSM 2026 — International Conference on Fuzzy Systems and Data Mining** (*Frontiers in Artificial Intelligence and Applications*, IOS Press).

---

## Overview

Accurate mapping of agricultural land potential is essential for sustainable land management and precision agriculture. This repository provides a standardized data mining, preprocessing, and benchmarking pipeline for multi-class semantic segmentation on 12-channel multispectral Sentinel-2 imagery (**10 optical bands + NDVI + NDWI**).

### Key Contributions
- **Data Mining & Pruning Pipeline**: Systematic label auditing (clipping the non-existent *Very High* class into a robust 5-class schema) and targeted patch pruning (filtering `< 0.5%` foreground and pure-background patches, reducing `97,790` patches to `85,376` high-quality patches).
- **Spectral Feature Engineering**: Derivation of Normalized Difference Vegetation Index (**NDVI**) and Normalized Difference Water Index (**NDWI**) concatenated with 10 Sentinel-2 bands into a **12-channel** input tensor.
- **Standardized Benchmark**: Fair comparison of 5 representative encoder-decoder architectures under a unified training protocol (Joint Weighted Cross-Entropy + Dice Loss, AdamW, ReduceLROnPlateau, Early Stopping, and AMP).
- **Physical Area Estimation**: Evaluation using **mIoU**, **Overall Accuracy (OA)**, and macroscopic physical area error in **km²** ($5\text{m} \times 5\text{m} = 25\text{ m}^2/\text{pixel}$).

---

## Dataset

This benchmark uses the **AgriPotential** dataset:
- **Source**: [AgriPotential Dataset on Zenodo (DOI: 10.5281/zenodo.17787322)](https://doi.org/10.5281/zenodo.17787322)
- **Spatial Resolution**: $5\text{ m/px}$, cropped into $128 \times 128$ pixel patches.
- **Input Channels (12)**: 10 Sentinel-2 optical bands (Visible, NIR, SWIR) + **NDVI** + **NDWI**.
- **Target Classes (5)**:
  - `Class 0`: Background (`54.35%`)
  - `Class 1`: Very Low (`14.71%`)
  - `Class 2`: Low (`9.55%`)
  - `Class 3`: Average (`11.46%`)
  - `Class 4`: High (`9.93%`)

---

## Repository Structure

```text
.
├── data_pipeline/          # Data mining, pruning, splitting, and 12-channel Dataset
│   ├── export_h5.py        # Extract 10-band Sentinel-2 TIFs & 5-class masks from HDF5
│   ├── merge_months.py     # Merge 11 monthly folders into unified split directories
│   ├── prune_dataset.py    # Filter low-info (<0.5% fg) & pure-background patches
│   ├── analyze_labels.py   # Compute class distribution & inverse-frequency weights
│   ├── split_dataset.py    # Deduplicate & generate 70/20/10 train/val/test index files
│   └── dataset.py          # PyTorch Dataset (10 bands + NDVI + NDWI -> 12 channels)
├── models/                 # Benchmarked segmentation architectures (in_channels=12, num_classes=5)
│   ├── __init__.py         # Unified model registry (build_model)
│   ├── unet.py             # Standard U-Net
│   ├── unetpp.py           # Standard U-Net++
│   ├── deeplabv3.py        # DeepLabV3+ (ResNet-50 + ASPP)
│   ├── ghost_unetpp.py     # Ghost-ResNet + U-Net++
│   └── convnext_upernet.py # ConvNeXt-V2 + UperNet
├── metrics.py              # Shared confusion matrix, mIoU, and OA computation
├── train.py                # Unified training script (JointLoss, AdamW, AMP, Early Stopping)
├── evaluate.py             # Unified evaluation script (mIoU, OA, physical area in km²)
├── requirements.txt        # Python dependencies
└── .gitignore
```

---

## Installation

```bash
git clone https://github.com/daoduyphat/fdsm2026-agripotential-segmentation.git
cd fdsm2026-agripotential-segmentation
pip install -r requirements.txt
```

---

## Usage

### 1. Data Mining & Preprocessing
1. Export raw Sentinel-2 patches and clipped 5-class masks from HDF5:
   ```bash
   python data_pipeline/export_h5.py
   python data_pipeline/merge_months.py
   ```
2. Prune low-information and pure-background patches (`97,790 -> 85,376`):
   ```bash
   python data_pipeline/prune_dataset.py
   ```
3. Analyze class distribution and generate `70/20/10` split index files (`train.txt`, `val.txt`, `test.txt`):
   ```bash
   python data_pipeline/analyze_labels.py
   python data_pipeline/split_dataset.py
   ```

### 2. Training
Train any of the 5 benchmarked architectures (`unet`, `unetpp`, `deeplabv3`, `ghost_unetpp`, `convnext_upernet`):
```bash
python train.py --model deeplabv3 --data-root dataset_final --split-dir . --epochs 150 --batch-size 32 --lr 1e-4
```

### 3. Evaluation
Evaluate a trained checkpoint on the test set (computes mIoU, OA, per-class IoU, and area error in km²):
```bash
python evaluate.py --model deeplabv3 --weights outputs/deeplabv3_best.pth --data-root dataset_final --split-file test.txt
```

---

## Benchmark Results

### Quantitative Comparison (Test Set)

| Model Architecture | Params | FLOPs | mIoU | OA |
| :--- | :---: | :---: | :---: | :---: |
| U-Net | 31.04 M | 27.48 G | 0.3732 | 0.5936 |
| U-Net++ | 36.62 M | 69.09 G | 0.5873 | 0.7641 |
| **DeepLabV3** | 39.66 M | 20.75 G | **0.8959** | **0.9538** |
| Ghost-ResNet + U-Net++ | 15.18 M | 9.29 G | 0.6920 | 0.8456 |
| ConvNeXt-V2 + UperNet | **5.28 M** | **2.02 G** | 0.5030 | 0.7316 |

### Per-Class Performance & Area Estimation (DeepLabV3)

| Class Name | IoU | Pred Area (km²) | GT Area (km²) | Diff (km²) |
| :--- | :---: | :---: | :---: | :---: |
| Background | 0.9373 | 2952.43 | 3084.97 | -132.54 |
| Level 1 (Very Low) | 0.8666 | 891.39 | 846.18 | +45.21 |
| Level 2 (Low) | 0.8557 | 589.21 | 549.10 | +40.12 |
| Level 3 (Average) | 0.9017 | 687.71 | 664.56 | +23.14 |
| Level 4 (High) | 0.9181 | 610.79 | 586.72 | +24.07 |
