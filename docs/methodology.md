# TSA Detection - Processing Pipeline

## Data Collection
UAV data was collected from 11 sites over 4 seasonal campaigns between 2023–2025 using DJI Matrice 300, DJI Matrice 600, and Mavic 3 Multispectral UAVs, equipped with RGB, multispectral, and hyperspectral sensors.

## Preprocessing
- Orthomosaic generation using Agisoft Metashape Pro, Pix4D, DJI Terra.
- Radiometric and atmospheric corrections.
- Alignment of orthomosaics from different sensors.
- ROI extraction based on ground truth.
- Manual labelling in QGIS.

## Model Training
ML models: SVM, Random Forest, XGBoost, KNN.
DL models: U-Net, FCN (TensorFlow/Keras).

Metrics:
- Precision
- Recall
- F1-score

## Evaluation
- Split datasets into training, validation, and testing.
- Performance comparison across sensors.
- Visual validation with experts.

## Infrastructure
- Python libraries: GDAL, XGBoost, Scikit-learn, OpenCV, Matplotlib, TensorFlow, Keras.
- HPC with NVIDIA A100 GPUs.
