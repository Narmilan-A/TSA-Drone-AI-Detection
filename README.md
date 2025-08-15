# Tropical Soda Apple (TSA) Drone AI Detection

## Overview
This repository contains the complete code, documentation, datasets, and instructions for detecting **Tropical Soda Apple (TSA)** infestations using UAV-acquired RGB, multispectral (MS), and hyperspectral (HS) imagery combined with **machine learning (ML)** and **deep learning (DL)** models.

The project was developed as part of field campaigns across Northern New South Wales (2023–2025) and includes:
- **High-resolution UAV imagery** from multiple sensors.
- **Manual ground-truth labelling** using QGIS and ArcGIS Pro.
- **Preprocessing workflows** for orthomosaics, ROIs, and annotations.
- **ML/DL training scripts** for RGB, MS, and HS imagery.
- **Model performance evaluations** with precision, recall, and F1-scores.
- **Setup instructions** for Colab, local GPU machines, and HPC environments.

<table>
  <tr>
    <td align="center">
      <img src="https://github.com/user-attachments/assets/93805d54-3d3d-4ad6-a7c4-9cef46f90f29"
           alt="TSA labeling using Fuji RGB ROI" width="400" height="300" />
    </td>
    <td align="center" valign="middle"><span style="font-size: 48px; font-weight: bold;">➜</span></td>
    <td align="center">
      <img src="https://github.com/user-attachments/assets/21aead30-71a9-452b-9aa9-933098612071"
           alt="U-Net prediction" width="400" height="300" />
    </td>
  </tr>
  <tr>
    <td align="center"><em>TSA labeling using Fuji RGB ROI</em></td>
    <td></td>
    <td align="center"><em>U-Net prediction</em></td>
  </tr>
</table>


---

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
│   ├── Activity 2.1 - Labelling Techniques.md
│   ├── Activity 2.2 - Annotation_Guidelines.md
│   ├── Activity 2.3 - Ground_Truth_Data_and_Region_of_Interest.md
│   ├── Activity 2.4 - Using_QGIS_for_Annotation.md
│   ├── Activity 2.5 - Using_ArcGIS_Pro_for_Annotation.md
│   └── Activity 2.6 - Geo-SAM_QGIS_Plugin_Installation_Guide.md
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
    ├── Activity 3.1.1 - Google Colab Setup.md
    ├── Activity 3.1.2 - Local_Machine_Setup.md
    ├── Activity 3.1.3 - QUT_HPC_Setup.md
    ├── environment_windows_gpu.yml
    ├── requirements.txt
    └── TensorFlow GPU.md
```

---

## Quick Start

### 1. Environment Setup
Follow the relevant guide from `setup_instructions/`:
- **Google Colab:** `Activity 3.1.1 - Google Colab Setup.md`
- **Local GPU:** `Activity 3.1.2 - Local_Machine_Setup.md`
- **QUT HPC:** `Activity 3.1.3 - QUT_HPC_Setup.md`

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

---

## License
MIT License

---

## Citation
Amarasingam, N., & Dehaan, R. (2025). Development of Drone Detection Technology to Enhance Tropical Soda Apple Control in Rugged High-Value Grazing Country in Northern NSW. Charles Sturt University.
