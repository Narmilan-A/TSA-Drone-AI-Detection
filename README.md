# Tropical Soda Apple (TSA) Drone AI Detection

## Project Full Title  
**Development of Drone Detection Technology to Enhance Tropical Soda Apple (TSA) Control in Rugged High-Value Grazing Country in Northern NSW**

## Overview
This repository contains the complete code, documentation, dataset links, and instructions for detecting **Tropical Soda Apple (TSA)** infestations using UAV-acquired **RGB**, **multispectral (MS)**, and **hyperspectral (HS)** imagery combined with **machine learning (ML)** and **deep learning (DL)** models.

The project was developed as part of field campaigns across Northern New South Wales (2023–2025) and includes:

- **High-resolution drone imagery** from multiple sensors  
- **Manual ground-truth labelling** using QGIS
- **Preprocessing workflows** for orthomosaics, region of interests (ROIs), and annotations  
- **ML/DL training scripts** for RGB, MS, and HS imagery  
- **Model performance evaluations** with precision, recall, and F1-scores  
- **Setup instructions** for Colab, local GPU machines, and HPC environments

<img width="1757" height="476" alt="image" src="https://github.com/user-attachments/assets/07aae30d-5239-423d-9972-53792d432c0d" />

## Repository Structure
```
TSA-DRONE-AI-DETECTION/
│   README.md
│   tree_output.txt
│
├── docs/
│   ├── methodology.md        # Detailed methodology from technical report
│   ├── README.md              # Docs overview
│   ├── results.md             # Performance results, tables, and figures
│
├── ground_truth_labelling/
│   └── README.md              # Ground truth data preparation process
│
├── image_annotation/
│   ├── README.md
│   ├── Using_QGIS_for_Annotation.md
│
├── machine_learning_scripts/
│   ├── hs/                    # Hyperspectral model scripts
│   ├── ms/                    # Multispectral model scripts
│   ├── rgb/                   # RGB model scripts
│   └── other/                 # Utility scripts for raster/vector operations
│
├── model_performance/         # Model results, metrics, plots
│
├── sample_data/
│   └── README.md              # Example dataset for quick tests
│
└── setup_instructions/
    ├── Google Colab Setup.md
    ├── Local_Machine_Setup.md
    ├── HPC_Setup.md
    ├── environment_windows_gpu.yml
    ├── requirements.txt
    └── TensorFlow GPU.md
```

---

## Quick Start

### 1. Environment Setup
Follow the relevant guide from `setup_instructions/`:
- **Google Colab:** `Google Colab Setup.md`
- **Local GPU:** `Local_Machine_Setup.md`
- **HPC:** `HPC_Setup.md`

Or use Conda:
```bash
conda env create -f setup_instructions/environment_windows_gpu.yml
conda activate tsa-detection
```

---

### 2. Data Preparation
1. Download UAV imagery and ground truth data from the cloud link provided in `sample_data/README.md`.
2. Place raw orthomosaics and labels in the appropriate folders.
3. Follow the steps in `ground_truth_labelling/README.md` and `image_annotation/` guides.

---

### 3. Running Models
Training and inference scripts are under `machine_learning_scripts/`:

- **RGB:**
```bash
python machine_learning_scripts/rgb/tsa_unet_train_rgb_v1.py
python machine_learning_scripts/rgb/tsa_unet_predict_rgb_v1.py
```

- **Multispectral (MS):**
```bash
python machine_learning_scripts/ms/tsa_unet_train_ms_v1.py
python machine_learning_scripts/ms/tsa_unet_predict_ms_v1.py
```

- **Hyperspectral (HS):**
```bash
# Add HS training/inference commands here
```

---

## Results
Detailed performance metrics and evaluation results are available in `docs/results.md`.
<img width="1932" height="820" alt="image" src="https://github.com/user-attachments/assets/838b75a2-7570-4d0c-9df8-ae661abcaccf" />


---

## License
....
---

## Citation
Amarasingam, N., & Dehaan, R. (2025). Development of Drone Detection Technology to Enhance Tropical Soda Apple Control in Rugged High-Value Grazing Country in Northern NSW. Charles Sturt University.
