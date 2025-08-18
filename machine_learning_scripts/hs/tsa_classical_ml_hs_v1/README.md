## Hyperspectral model training using classical machine learning

Train, test, and predict TSA vs Background from hyperspectral imagery using four models: **RandomForest, XGBoost, SVM, KNN**. Fully driven by `config/config.yaml`.

### Requirements

- Python 3.9+ (Anaconda recommended)
- Packages: `numpy`, `scikit-learn`, `xgboost`, `gdal` (or `osgeo`), `matplotlib`, `seaborn`, `pyyaml`, `joblib`

### Configure

Edit `config/config.yaml`:
- Update the paths under `data:`
- Set `model_type` to one of: `RandomForest`, `XGBoost`, `SVM`, `KNN`
- Adjust `processing.top_items_count_bands` as needed (e.g., 30)
- Tune per-model hyperparameters under their sections

### How to Run

#### Option A — Run as packages (recommended)
Open a terminal in the **project root** (the folder that contains `config/`, `utils/`, `train/`, `test/`, `predict/`) and run:

```bash
python -m train.train_model
python -m test.test_model
python -m predict.predict_model
```

#### Option B — Run scripts directly (VS Code / double-click / F5)
The scripts include a small `sys.path` fallback so you can run them directly too:

```bash
python .\train\train_model.py
python .\test\test_model.py
python .\predict\predict_model.py
```

> If you still get `ModuleNotFoundError: No module named 'utils'`, ensure you are running **from the project root**.

### Outputs

Training outputs are saved to:

```
Training_model_outcomes/<ModelType>/run_YYYYmmdd_HHMMSS/
```

Each run contains:
- `best_<model>_model.pkl` — trained model
- `training_scaler.pkl` — if normalization enabled
- `selected_bands.txt` — indices of top spectral bands
- `hyperparameters.txt` — hyperparameters used
- `training_samples.txt` — per-image TSA/BG counts + dataset shapes
- `<model>_CR_CM_report.txt` + `<model>_CM_heatmap.png`
- `predictions/` — GeoTIFF (and ENVI if enabled)

### Notes

- Masks should use: 0=unlabeled, 1=TSA, 2=Background.
- Band selection is based on mean difference between class means for each band.
- Ensure GDAL is correctly installed for Python (on Windows, try OSGeo4W or conda-forge).

### Troubleshooting

- **No module named 'utils'**: run from project root, or use `python -m`, or check `__init__.py` files exist.
- **GDAL import issues**: install via conda: `conda install -c conda-forge gdal`.
- **No training data found**: verify your `train_image_folder` and `train_mask_folder` and that mask files are named `mask_<image>.tif`.
