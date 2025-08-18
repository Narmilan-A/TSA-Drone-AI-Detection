# TSA Drone AI Detection Dataset

## Dataset Overview
This dataset contains drone-acquired imagery and annotations for **Tropical Soda Apple (TSA)** detection in Northern New South Wales. It was collected during field campaigns (2023–2025) and is intended for training and evaluating **machine learning (ML)** and **deep learning (DL)** models for weed detection.  

The dataset is hosted on the MS Teams shared drive:  
[Access the TSA Dataset](https://csuprod.sharepoint.com/:f:/r/sites/RSRCH-WeedDataShared/Shared%20Documents/General/TSA-Drone-AI-Detection?csf=1&web=1&e=FypsMy)

---
## Naming Conventions

### Seasons
- `oct-24` → October 2024  
- `apr-23` → April 2023  
- `jan-25` → January 2025  
- `nov-23` → November 2023  

### RGB Sensors
- `sony` → Sony  
- `fuji` → Fuji   
- `p1` → DJI Phantom 1  
- `phase1` → Phase One  
- `m3e` → DJI Mavic 3 Enterprise
- `m3m-rgb` → DJI Mavic 3 Multispectral  

### Multispectral (MS) Sensors
- `altum` → Micasense Altum  
- `rededge-p` → Micasense RedEdge-P
- `m3m-ms` → DJI Mavic 3 Multispectral

### Hyperspectral (HS) Sensors
- `afx` → Specim AFX hyperspectral  

### Tiles vs Original
- `_tile` → cropped ROI tiles for model training/testing  
- Non-tile → original ROI folders for reference or annotation
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
│   │   ├── Site1
│   │   └── Site2
│   ├── jan-25/
│   │   ├── Site1
│   │   ├── Site2
│   │   ├── Site3
│   │   └── Site4
│   └── nov-23/Site1
├── sensors_for_modelling/
│   ├── rgb/
│   │   ├── sony/
│   │   ├── fuji/
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
    │   ├── sony
    │   ├── p1
    │   ├── phase1
    │   ├── m3e
    │   ├── m3m-rgb
    │   └── fuji
    └── ms/
        ├── m3m-ms/
        ├── altum/
        └── rededge-p/
```

---

## Dataset Description

### 1. seasons_for_labelling
Contains drone imagery organised by **season and site** for manual annotation.

### 2. sensors_for_modelling
Contains **RGB, MS, and HS data** processed into regions of interest for model training and testing.  

- **RGB sensors:** sony, fuji, p1, phase1, m3e, m3m-rgb  
- **MS sensors:** rededge-p, m3m-ms, altum  
- **HS sensors:** afx  
- Each sensor has:  
  - `rgb_rois` or `ms_rois` or `hs_rois` (images for model input)  
  - `mask_rois` (binary masks for training segmentation models)  
  - Subfolders: `all`, `training`, `testing`  

### 3. scripts
Contains preprocessing and analysis scripts for **RGB, MS, HS**, and utility scripts for dataset info extraction.

### 4. sensors_model_performance
Contains model predictions and evaluation outputs for **RGB, MS, and HS sensors**, organised per sensor.

---

## Usage Instructions
1. Download the dataset from the MS Teams link.  
2. Maintain the folder structure when using scripts from the TSA Drone AI Detection repository.  
3. Use `training` and `testing` folders to split data for ML/DL model development.  
4. `mask_rois` folders are used for segmentation ground-truth labels.  

---

## Citation
```
@data{TSA-Drone-AI-Detection,
  url = {https://csuprod.sharepoint.com/:f:/r/sites/RSRCH-WeedDataShared/Shared%20Documents/General/TSA-Drone-AI-Detection?csf=1&web=1&e=FypsMy},
  author = {Amarasingam, N., & Dehaan, R and collaborators},
  publisher = {Charles Sturt University / MS Teams Shared Drive},
  title = {Development of Drone Detection Technology to Enhance Tropical Soda Apple Control in Rugged High-Value Grazing Country in Northern NSW},
  year = {2025}
}
```

