# Tropical Soda Apple (TSA) Drone AI Detection

## Project Full Title  
**Development of Drone Detection Technology to Enhance Tropical Soda Apple (TSA) Control in Rugged High-Value Grazing Country in Northern NSW**

## Overview
This repository contains the complete code, documentation, dataset links, and instructions for detecting **Tropical Soda Apple (TSA)** infestations using UAV-acquired **RGB**, **multispectral (MS)**, and **hyperspectral (HS)** imagery combined with **machine learning (ML)** and **deep learning (DL)** models.

The project was developed as part of field campaigns across Northern New South Wales (2023–2025) and includes:

- **High-resolution drone imagery** from multiple sensors  
- **Manual ground-truth labelling** using QGIS
- **ML/DL training scripts** for RGB, MS, and HS imagery  
- **Model performance evaluations** with precision, recall, and F1-scores  
- **Setup instructions** for Colab, local GPU machines, and cloud-based HPC environments

<img width="1757" height="476" alt="image" src="https://github.com/user-attachments/assets/07aae30d-5239-423d-9972-53792d432c0d" />

## Repository Structure
```
TSA-DRONE-AI-DETECTION/
│   README.md
│
├── docs/
│   ├── methodology.md        # Detailed methodology from technical report
│   ├── README.md              # Docs overview
│
├── image_annotation/
│   ├── README.md              # Ground truth data preparation process
│   ├── Using_QGIS_for_Annotation.md
│
├── machine_learning_scripts/
│   ├── README.md              # Docs overview
│   ├── hs/                    # Hyperspectral model scripts
│   ├── ms/                    # Multispectral model scripts
│   ├── rgb/                   # RGB model scripts
│   └── other/                 # Utility scripts for raster/vector operations
│
├── model_performance/         # Model results, metrics, plots
│   ├── README.md              # Docs overview
│
├── dataset Links/
│   └── README.md              # Example dataset for quick tests
│
└── setup_instructions/
    ├── Google Colab Setup.md
    ├── Local_Machine_Setup.md
    ├── Cloud_based_HPC_Setup.md
    ├── TensorFlow-GPU_Installation.yml
    ├── environment_hpc.yml
    ├── environment_local_machine.yml
    └── README.md
```

---

## Quick Start

### 1. Environment Setup
Follow the relevant guide from `setup_instructions/`:
- **Google Colab:** `Google Colab Setup.md`
- **Local GPU:** `Local_Machine_Setup.md`
- **HPC:** `HPC_Setup.md`

---

### 2. Data Preparation
1. Download drone imagery and ground truth data from the cloud link.
2. Place ROIs and labels in the appropriate folders.
3. Use ML scripts to train, validate and test for TSA detection.

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

## Citation

If you use this dataset or related work in your research, please cite:

```bibtex
@misc{tsa_drone_ai_detection,
  title={Development of Drone Detection Technology to Enhance Tropical Soda Apple Control in Rugged High-Value Grazing Country in Northern NSW},
  author={Amarasingam, N., & Dehaan, R and collaborators},
  year={2025},
  url={https://csuprod.sharepoint.com/:f:/r/sites/RSRCH-WeedDataShared/Shared%20Documents/General/TSA-Drone-AI-Detection?csf=1&web=1&e=FypsMy}
}
