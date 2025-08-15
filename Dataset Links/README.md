# TSA Drone AI Detection Dataset

## Dataset Overview
This dataset contains UAV-acquired imagery and annotations for **Tropical Soda Apple (TSA)** detection in Northern New South Wales. It was collected during field campaigns (2023–2025) and is intended for training and evaluating **machine learning (ML)** and **deep learning (DL)** models for weed detection.  

The dataset is hosted on the MS Teams shared drive:  
[Access the TSA Dataset](https://csuprod.sharepoint.com/:f:/r/sites/RSRCH-WeedDataShared/Shared%20Documents/General/TSA-Drone-AI-Detection?csf=1&web=1&e=FypsMy)

---

## Folder Structure

```
TSA-Drone-AI-Detection/
├── seasons_for_labelling/
│   ├── oct-24/
│   │   ├── Site1
│   │   ├── Site2
│   │   ├── Site3
│   │   └── Site4
│   ├── apr-23/
│   │   ├── Site1/hs
│   │   └── Site2
│   ├── jan-25/
│   │   ├── Site1/hs
│   │   ├── Site2
│   │   ├── Site3/hs
│   │   └── Site4
│   └── nov-23/Site1
├── sensors_for_modelling/
│   ├── rgb/
│   │   ├── sony/
│   │   │   ├── mask_rois/
│   │   │   └── rgb_rois/
│   │   ├── fuji/
│   │   │   ├── mask_rois/
│   │   │   └── rgb_rois/
│   │   ├── p1/
│   │   ├── phase1/
│   │   ├── m3e/
│   │   ├── m3m-rgb/
│   │   ├── fuji_tile/
│   │   ├── p1_tile/
│   │   ├── m3e_tile/
│   │   ├── m3m-rgb_tile/
│   │   ├── phase1_tile/
│   │   └── sony_tile/
│   ├── ms/
│   │   ├── rededge-p/
│   │   ├── m3m-ms/
│   │   ├── altum/
│   │   ├── m3m-ms_tile/
│   │   ├── altum_tile/
│   │   └── rededge-p_tile/
│   └── hs/
│       └── afx/
├── scripts/
│   ├── ms/
│   ├── rgb/
│   ├── hs/
│   └── other/get_info/
└── sensors_model_performance/
    ├── rgb/
    │   ├── sony/unet_prediction_rois
    │   ├── p1/unet_prediction_rois
    │   ├── phase1/unet_prediction_rois
    │   ├── m3e/unet_prediction_rois
    │   ├── m3m-rgb/unet_prediction_rois
    │   └── fuji/unet_prediction_rois
    └── ms/
        ├── m3m-ms/
        ├── altum/
        │   ├── xgb_altum_model&outcomes_1
        │   └── unet_altum_model&outcomes
        └── rededge-p/
            └── xgb_rededge-p_model&outcomes_1
```

---

## Dataset Description

### 1. seasons_for_labelling
Contains UAV imagery organized by **season and site** for manual annotation. Some sites include **hyperspectral (hs)** folders.

### 2. sensors_for_modelling
Contains **RGB, MS, and HS data** processed into regions of interest (ROIs) for model training and testing.  

- **RGB sensors:** sony, fuji, p1, phase1, m3e, m3m-rgb  
- **MS sensors:** rededge-p, m3m-ms, altum  
- **HS sensors:** afx  
- Each sensor has:  
  - `rgb_rois` or `ms_rois` (images for model input)  
  - `mask_rois` (binary masks for training segmentation models)  
  - Subfolders: `all`, `training`, `testing`  

### 3. scripts
Contains preprocessing and analysis scripts for **RGB, MS, HS**, and utility scripts for dataset info extraction.

### 4. sensors_model_performance
Contains model predictions and evaluation outputs for **RGB and MS sensors**, organized per sensor.

---

## Usage Instructions
1. Download the dataset from the MS Teams link.  
2. Maintain the folder structure when using scripts from the TSA Drone AI Detection repository.  
3. Use `training` and `testing` folders to split data for ML/DL model development.  
4. `mask_rois` folders are used for segmentation ground-truth labels.  

---

## Citation
If you use this dataset in publications, please cite:  

