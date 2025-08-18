# Machine Learning Scripts — Repository Overview

This repository contains end‑to‑end pipelines for remote‑sensing segmentation/classification, including data preparation, model training (e.g., U‑Net), and evaluation/utilities. The code is organized into four main folders to reflect different sensor types and workflows.

## Folder Structure

### `machine_learning_scripts`

Scripts:

- `machine_learning_scripts/hs/tsa_classical_ml_hs_v1/predict/__init__.py` — Python script.
- `machine_learning_scripts/hs/tsa_classical_ml_hs_v1/predict/predict_model.py` — Add project root to sys.path when running directly
- `machine_learning_scripts/hs/tsa_classical_ml_hs_v1/test/__init__.py` — Python script.
- `machine_learning_scripts/hs/tsa_classical_ml_hs_v1/test/test_model.py` — Add project root to sys.path when running directly
- `machine_learning_scripts/hs/tsa_classical_ml_hs_v1/train/__init__.py` — Python script.
- `machine_learning_scripts/hs/tsa_classical_ml_hs_v1/train/train_model.py` — train/train_model.py
- `machine_learning_scripts/hs/tsa_classical_ml_hs_v1/utils/__init__.py` — Python script.
- `machine_learning_scripts/hs/tsa_classical_ml_hs_v1/utils/data_utils.py` — Python script.
- `machine_learning_scripts/hs/tsa_classical_ml_hs_v1/utils/model_utils.py` — Python script.
- `machine_learning_scripts/hs/tsa_unet_predict_hs_v1.py` — ==============================
- `machine_learning_scripts/hs/tsa_unet_test_hs_v1.py` — ==============================
- `machine_learning_scripts/hs/tsa_unet_train_hs_v1.py` — ==============================
- `machine_learning_scripts/ms/tsa_unet_predict_ms_v1.py` — Import general python libraries
- `machine_learning_scripts/ms/tsa_unet_test_ms_v1.py` — Import general python libraries
- `machine_learning_scripts/ms/tsa_unet_train_ms_v1.py` — Import general python libraries
- `machine_learning_scripts/other/spectral_selection_m3m.py` — Python script.
- `machine_learning_scripts/other/tsa_rasterize_vector_labels_rgb.py` — Import general python libraries
- `machine_learning_scripts/rgb/tsa_unet_predict_rgb_v3.py` — Import general python libraries
- `machine_learning_scripts/rgb/tsa_unet_test_rgb_v3.py` — Import general python libraries
- `machine_learning_scripts/rgb/tsa_unet_train_rgb_v3.py` — Import general python libraries

## What’s Inside (at a glance)

- **Data I/O & Tiling**: Utilities for loading GeoTIFF/ENVI imagery with GDAL, extracting fixed‑size tiles or patches, and pairing them with masks.

- **Model Training**: U‑Net training pipelines (Keras/TensorFlow), with options for per‑band normalization, optional histogram equalization/CLAHE, overlap tiling, class weighting, and checkpointing.

- **Evaluation**: Confusion matrices, precision/recall, IoU, Mean IoU, and training history exports (.csv) with plots.

- **Prediction**: Sliding‑window inference and georeferenced output saving.

## Environments (brief)

- **Local workstation**: Python 3.10+, GDAL, NumPy, OpenCV, scikit‑image, scikit‑learn, TensorFlow/Keras, Matplotlib, Seaborn.

- **HPC (e.g., Slurm/LUMI‑like)**: Same Python deps; ensure GDAL bindings are available on compute nodes and set `CUDA_VISIBLE_DEVICES` as needed. Use `.keras` model format and `.weights.h5` for checkpoints.

- **Google Colab**: Install GDAL via `apt` and Python deps via `pip`. Mount Drive for datasets/models; prefer `.keras` format for portability.

## Data & Masks

- Expect image/mask pairs aligned by ROI/tile keys, e.g., `...roi_<n>_tile_<m>.tif` and `mask_...roi_<n>_tile_<m>.tif`.

- For multispectral/HS data, you may specify selected band indices (0‑based) or default to all bands.

- Masks can be binary (bg/tsa) or multi‑class; see training script flags for class handling.

## Outputs

- Trained models saved to `.keras`; periodic checkpoints saved as `.weights.h5`.

- Metrics and plots (PNG), and CSV logs of training history and dataset audits.

## Notes

- If you see `not a TIFF file b'\x00\x00\x00\x00'`, your copy to a network share likely produced zero‑filled files. Re‑copy with verification (e.g., `rsync --partial --inplace --checksum`).

- Histogram equalization is optional and should be applied per‑band if used.

- For imbalanced datasets (tsa ≪ bg), enable class weighting or pixel‑wise weights in the training script.
