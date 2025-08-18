# Machine Learning Scripts — Repository Overview

This repository contains end‑to‑end pipelines for remote‑sensing segmentation, including data preparation, model training (e.g., U‑Net), and evaluation/utilities. The code is organised into four main folders to reflect different sensor types and workflows.

## Folder Structure

### `machine_learning_scripts`

```text
machine_learning_scripts/
├── hs/
│   ├── tsa_classical_ml_hs_v1/
│   │   ├── predict/
│   │   │   ├── __init__.py
│   │   │   └── predict_model.p
│   │   ├── test/
│   │   │   ├── __init__.py
│   │   │   └── test_model.p
│   │   ├── train/
│   │   │   ├── __init__.py
│   │   │   └── train_model.py
│   │   └── utils/
│   │       ├── __init__.py
│   │       ├── data_utils.py
│   │       └── model_utils.py
│   ├── tsa_unet_predict_hs_v1.py
│   ├── tsa_unet_test_hs_v1.py
│   └── tsa_unet_train_hs_v1.py
├── ms/
│   ├── tsa_unet_predict_ms_v1.py
│   ├── tsa_unet_test_ms_v1.py
│   └── tsa_unet_train_ms_v1.py
├── other/
│   ├── spectral_selection_m3m.py
│   └── tsa_rasterize_vector_labels_rgb.py
└── rgb/
    ├── tsa_unet_predict_rgb_v3.py
    ├── tsa_unet_test_rgb_v3.py
    └── tsa_unet_train_rgb_v3.py
```
