# Model Performance – Quick Overview

This README summarises where to find evaluation outputs (reports, plots, and predictions) produced by the HS/MS/RGB training runs.

## OneDrive Access (add your link)
**Primary OneDrive folder**: [Open in OneDrive](https://csuprod.sharepoint.com/:f:/r/sites/RSRCH-WeedDataShared/Shared%20Documents/General/TSA-Drone-AI-Detection?csf=1&web=1&e=tYbznU)

---

## What’s inside (typical)
- `hs/hs_test_results/` – Validation/test confusion matrix heatmaps, classification reports, and IoU summaries.
- `hs/hs_unet_predictions/` – GIS‑ready per‑tile predictions in **GeoTIFF** and **ENVI** (`.dat/.hdr`).
- `*/tsa_unet_train_*_model&outcomes_*/` – Per‑experiment folders containing:
  - `unet_best_model.keras` (Keras v3 format) and periodic `*.weights.h5` checkpoints.
  - `unet_training_history.csv` and plots: `Accuracy.png`, `Loss.png`, `IoU.png`, etc.
  - `unet_cm_heatmap_val.png` and `unet_val_report.txt` (confusion matrix & classification report).
  - `dataset_audit.csv` showing which image/mask pairs and tiles were used.

If you keep multiple experiments, each run will create its own timestamped/config‑encoded directory under the corresponding sensor root.

---

# Model Performance & Artifacts — Overview

This README describes **what each file/folder contains** in your *model performance* dump, and how to use the artifacts to review and compare experiments. It’s written to match the current layout you shared (hyperspectral **hs**, multispectral **ms**, and RGB **rgb**).

---

## 1) Top-level layout

- `hs/` – Hyperspectral U-Net runs and predictions  
- `ms/` – Multispectral experiments  
  - `altum/` – U-Net run + an **XGBoost** baseline (`xgb_altum_model&outcomes_1/`)  
  - `m3m-ms/` – U-Net run (train/val + test artifacts)  
  - `rededge-p/` – U-Net run + an **XGBoost** baseline (`xgb_rededge-p_model&outcomes_1/`)  
- `rgb/fuji/` – RGB U-Net run (train/val + test artifacts)  

> The structure is consistent: each dataset has a U-Net run (images, CSV, text reports) and some datasets also have classical ML baselines (XGBoost) for comparison.

---

## 2) Common U‑Net artifacts (appear in most model folders)

- **`unet_save_best_model.keras` / `unet_best_model.keras`**  
  The best checkpoint saved in the Keras v3 format. Use `keras.models.load_model(...)` to load for inference.
  
- **`unet_training_history.csv`**  
  Per‑epoch training & validation metrics (loss, accuracy, precision/recall, IoU, FN/FP). Useful for plotting custom charts or performing comparisons across runs.

- **Metric plots (PNGs)**  
  - `Accuracy.png` — Train/val accuracy per epoch.  
  - `Loss.png` — Train/val loss per epoch.  
  - `Precision.png`, `Recall.png` — Class‑specific behavior over training.  
  - `IoU.png` — (when present) Intersection‑over‑Union curve(s).  
  - `FalseNegatives.png`, `FalsePositives.png` — Helpful for class imbalance diagnostics.

- **Confusion matrix & report (validation)**  
  - `unet_cm_heatmap_val.png` — Heatmap of the validation confusion matrix.  
  - `unet_val_report.txt` or `unet_model_training_&_validation_performance_report.txt` — Detailed classification report (precision, recall, F1) alongside the confusion matrix.

- **Dataset bookkeeping**  
  - `dataset_audit.csv` — Reasons any tiles were skipped, and tile counts per image/mask pair.  
  - `unet_train_val_shapes.txt` / `unet_training_and_validation samples.txt` — Input shapes used, for reproducibility and quick checks.

---

## 3) Hyperspectral (**hs**) extras

- **Testing artifacts** (`hs/hs_test_results/`)  
  - `unet_cm_heatmap_testing.png` — Confusion matrix on the **held‑out test** tiles.  
  - `unet_model_testing_performance_report.txt` — Classification report for test tiles.  
  - `unet_testing_samples.txt` — Summary (counts/shapes) of the test dataset.

- **Predictions export** (`hs/hs_unet_predictions/`)  
  - `tif/` — GeoTIFF rasters of model predictions per ROI/tile (`*_pred.tif`).  
  - `envi/` — ENVI pair (`.hdr` + `.dat`) for the same predictions.  
  These are per‑tile probability/logit/label rasters (depending on the exporter used). They’re ready for GIS inspection (QGIS/ENVI).

---

## 4) Multispectral (**ms**) notes

- **`altum/` and `rededge-p/`** include **XGBoost** baselines:  
  - `best_xgb_model.pkl` — Serialized XGB model (load with `pickle` or `joblib`).  
  - `xgb_cr&cm_validation.txt`, `xgb_cm_heatmap_validation.png` — Validation metrics.  
  - `xgb_cr&cm_testing.txt`, `xgb_cm_heatmap_testing.png` — Held‑out test metrics.

- **`m3m-ms/`** follows the standard U‑Net pattern and includes **training** (`…training_&_validation_performance_report.txt`) and **testing** (`unet_model_testing_performance_report.txt`) reports plus confusion matrices for both.

> Use these baselines to cross‑check if the deep model is over/under‑performing a simple learner on the same features.

---

## 5) RGB (**rgb/fuji**) notes

- Contains both **train/validation** and **test** reports/plots like the MS runs.  
- Also includes a `unet_prediction_rois/` folder with per‑ROI/tile predictions exported to **.tif** and **ENVI** formats for GIS evaluation.

---

## 6) How to read & compare runs

1. **Start with the PNGs** (`Accuracy.png`, `Loss.png`) to spot over/under‑fitting.  
2. **Open the confusion matrix** for val/test to see dominant error types.  
3. **Read the classification reports** for precision/recall/F1 (especially for your minority class *tsa*).  
4. **If class imbalance is severe**, track `FalseNegatives.png` to ensure recall for *tsa* is improving.  
5. **Inspect predictions** (`*_pred.tif`) overlayed in QGIS on the source imagery to validate spatial patterns.  
6. **Use `unet_training_history.csv`** to compute any custom metric or to aggregate multiple runs in a spreadsheet.

---

**Last updated:** this file describes the artifacts currently present and will remain valid as long as the layout stays consistent.

