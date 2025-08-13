# Tropical Soda Apple (TSA) Detection Using UAVs and AI

## Overview
This repository contains the complete workflow for detecting **Tropical Soda Apple (TSA)** infestations in rugged grazing country of northern New South Wales, Australia, using **UAV-acquired RGB, multispectral (MS), and hyperspectral (HS) imagery** combined with **machine learning (ML) and deep learning (DL)** models.

The project integrates:
- High-resolution UAV imagery from multiple sensors.
- Orthomosaic generation and advanced preprocessing.
- Manual ground-truth labelling by weed specialists.
- ML/DL model training and evaluation (SVM, RF, XGBoost, U-Net).
- Prediction maps for operational weed management.

## Project Goals
1. Develop a generalizable AI pipeline for TSA detection across multiple seasons and sensors.
2. Compare classical ML and DL approaches for weed detection.
3. Recommend optimal UAV and sensor configurations for large-scale monitoring.
4. Enable reproducibility by sharing code, trained models, and documentation.

## Workflow Summary
1. **Data Collection**
   - UAV platforms: DJI Matrice 300, M600, Mavic 3 Multispectral.
   - Sensors: Fuji RGB, DJI P1, PhaseOne, M3E RGB, Micasense Altum, RedEdge-P, Specim AFX VNIR.
   - Sites: 11 locations across 4 seasonal campaigns (2023–2025).

2. **Preprocessing**
   - Orthomosaic generation (Agisoft Metashape, Pix4D, DJI Terra).
   - Radiometric and atmospheric correction.
   - Image alignment & ROI extraction.
   - Manual TSA labelling in QGIS.

3. **Model Training**
   - ML: SVM, Random Forest, XGBoost, KNN.
   - DL: U-Net, FCN (TensorFlow/Keras).
   - Performance metrics: Precision, Recall, F1-score.

4. **Prediction & Validation**
   - Full-site TSA prediction maps.
   - Visual validation with expert field observations.
   - Performance analysis across sensors.

## Repository Structure
```
TSA-Drone-Detection/
├── README.md
├── LICENSE
├── requirements.txt
├── setup_instructions.md
├── data/
│   ├── README.md
│   ├── sample_data/
│   └── metadata/
├── notebooks/
│   ├── 01_data_preprocessing.ipynb
│   ├── 02_model_training.ipynb
│   ├── 03_model_evaluation.ipynb
│   └── 04_prediction_visualisation.ipynb
├── src/
│   ├── preprocessing/
│   ├── training/
│   ├── evaluation/
│   └── prediction/
├── models/
│   ├── trained_weights/
│   └── README.md
└── docs/
    ├── methodology.md
    ├── results.md
    └── figures/
```

## Setup
1. Clone this repository
```bash
git clone https://github.com/yourusername/TSA-Drone-Detection.git
cd TSA-Drone-Detection
```
2. Create environment
```bash
conda create -n tsa-detection python=3.10
conda activate tsa-detection
pip install -r requirements.txt
```
3. Install GDAL system packages if needed.

## Data Access
Data is hosted externally due to size. Download orthomosaics, ROI tiles, and labels from the provided cloud link and place them under `data/`.

## Quick Start
Run notebooks in sequence to preprocess, train, evaluate, and predict.

## Results Summary
Best performance:
- Sensor: Fuji RGB / DJI P1 / PhaseOne
- Model: U-Net (Precision > 0.90, F1-score > 0.90)

## License
MIT License

## Citation
Amarasingam, N., & Dehaan, R. (2025). Development of Drone Detection Technology to Enhance Tropical Soda Apple Control in Rugged High-Value Grazing Country in Northern NSW. Charles Sturt University.
