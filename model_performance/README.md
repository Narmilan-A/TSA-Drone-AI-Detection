# Model Performance Guide

This README documents how to read, compare, and report performance for the U‑Net models in this repo (HS / MS / RGB). It is **code‑free** and focuses on what the metrics mean, where to find them, and how to summarize results clearly.

## 1) Scope and Terminology

- **tsa**: target class of interest (e.g., trees/shrubs).
- **bg**: background class.
- **Unlabeled (0)**: pixels to ignore in training/eval for the HS setup with sparse masks.
- **Pixel‑level metrics**: all metrics here are computed per pixel unless otherwise stated.
- **Tiles**: fixed‑size windows cut from rasters; predictions and metrics are aggregated over tiles.

## 2) Datasets & Labels

- **Binary setup (2 classes)**: `bg=0, tsa=1` (or vice‑versa as specified per script). Metrics and confusion matrix cover both classes.
- **Sparse 3‑label masks (ignore class 0)**: masks contain `{0: unlabeled, 1: tsa, 2: bg}`. Training and reporting **ignore 0** and evaluate only classes **1 (tsa)** and **2 (bg)**.

> Tip: When reporting, always state which label scheme you used and whether class 0 was ignored.

## 3) Splits & Reproducibility

- Typical split is **train/val** (e.g., 80/20) inside the training script, and an optional **held‑out test** split for final reporting.
- Scripts set a fixed random seed for split and model init, so results are reproducible given the same data, tile parameters, and preprocessing (e.g., histogram equalization on/off).

## 4) Reported Metrics (what you will see)

**From training/validation (saved under each run folder):**
- **Loss**: cross‑entropy on the chosen label scheme.
- **BinaryAccuracy** (binary) or general accuracy (multiclass).
- **Precision/Recall for TSA**: precision answers “when the model predicts tsa, how often is it correct?”; recall answers “how much of the true tsa was found?”
- **IoU (class=tsa)**: intersection‑over‑union for the target class.
- **MeanIoU**: average IoU across evaluated classes (excludes the ignored class 0 when applicable).
- **Confusion Matrix**: counts of predicted vs true pixels. Saved as a heatmap image and printed as a table.
- **Classification Report**: precision, recall, F1 for each class plus macro/weighted averages.
- **Training Curves**: CSV (`unet_training_history.csv`) and PNG plots for loss and metrics.

**From test evaluation (optional test script):**
- Confusion matrix, classification report, and IoU per class on the held‑out test tiles.
- A plain‑text performance report saved alongside plots.

## 5) Where to Find the Outputs

Each training run creates an outcome folder like:

```
<root>/tsa_unet_train_*_model&outcomes_*   (exact name encodes tile size, overlap, lr, epochs, etc.)
```

Common files inside a run directory:

- **`unet_best_model.keras`**: best model by validation loss (Keras v3 format).
- **`log/weights.*.weights.h5`**: checkpointed weights across epochs.
- **`unet_cm_heatmap_val.png`**: validation confusion matrix heatmap.
- **`unet_val_report.txt`**: validation confusion matrix + classification report + training time.
- **`unet_training_history.csv`**: per‑epoch metrics and loss.
- **`dataset_audit.csv`**: exactly which images/masks were used and why any were skipped.

Test‑time (if you run the test script):
- **`hs_test_results/unet_cm_heatmap_testing.png`** and **`unet_model_testing_performance_report.txt`** (or similar for MS/RGB).

## 6) Interpreting the Numbers (especially with class imbalance)

When **background massively outnumbers tsa**, overall accuracy can be misleadingly high. Prioritize:

- **Recall (tsa)**: ability to capture tsa pixels (misses are costly if tsa is the target).
- **Precision (tsa)**: fraction of predicted tsa that is truly tsa (controls false alarms).
- **IoU (tsa)**: balances overlap between prediction and ground truth for the target class.

If you see high accuracy but **very low tsa recall**, the model is ignoring tsa. Consider balancing strategies (below).

## 7) Class Imbalance: What to Report and How to Address

**What to report**
- Class distribution (percentage of tsa vs bg pixels) in the *used* tiles (after filtering and tiling).
- **Precision/Recall/IoU for tsa** prominently. Include confusion matrix to show error modes.
- Macro vs weighted averages (macro is less biased by the dominant class).

**What you can adjust in training (qualitatively)**
- **Pixel weights** (e.g., up‑weight tsa in the loss). Increasing the tsa weight pushes the model to find more tsa (boosting recall), but may lower precision if set too high.
- **Tile rebalancing**: oversample tiles containing tsa.
- **Threshold tuning**: after softmax/sigmoid outputs, adjust the decision threshold for tsa if needed.
- **Augmentation**: class‑aware augmentations that preserve minority patterns.

> Always document the chosen weights/thresholds so others can reproduce the reported numbers.

## 8) Recommended Primary Table (per run)

Fill once training/validation and test are complete:

| Run ID | Data (HS/MS/RGB) | Tile Size | Bands Used | HE/CLAHE | Train/Val Split | TSA Prec | TSA Recall | TSA IoU | Mean IoU | Val Acc | Test TSA IoU | Notes |
|---|---|---:|---:|---|---|---:|---:|---:|---:|---:|---:|---|

Notes can include pixel weights, overlap %, learning rate, and any preprocessing flags.

## 9) Quality Checks You Can Cite

- **`dataset_audit.csv`** lists each image/mask pair and reasons for any skips (size mismatch, open failure, image smaller than tile, etc.).
- **Pairing rule** is by a normalized `roi_<n>_tile_<m>` key; mask filenames can have a `mask_` prefix but must share the same key.
- **Selected Bands**: if using HS selected bands, the indices must match the source rasters; mismatches can degrade performance.

## 10) Known Pitfalls to Mention in Reports

- **Invalid rasters after copying**: zero‑filled `.tif` on network shares cause “not a TIFF file b'\x00\x00\x00\x00'”. Verify with `gdalinfo` before training.
- **Histogram Equalization changes the distribution**: keep the same setting (on/off) between training and inference or expect drift.
- **Tile geometry must match model**: tile size, overlap, and band count must be consistent across train/test/predict scripts.
- **Ignored class 0**: for sparse masks, always state that unlabeled pixels were excluded from loss/metrics.

## 11) Changelog (example template)

- YY‑MM‑DD — HS, 64×64, selected‑bands N=20, HE=on (hist), lr=1e‑3, epochs=200. TSA Recall 0.62, TSA IoU 0.48, MeanIoU 0.66. Pixel weights TSA:BG = 2:1.
- YY‑MM‑DD — MS, 128×128, 4 bands + indices, HE=off, lr=1e‑3, epochs=120. TSA Recall 0.54, TSA IoU 0.41. Oversampled tsa tiles ×2.

## 12) Glossary

- **Precision (tsa)**: TP / (TP + FP) for the tsa class.
- **Recall (tsa)**: TP / (TP + FN) for the tsa class.
- **IoU (tsa)**: TP / (TP + FP + FN) for the tsa class.
- **Macro avg**: arithmetic average across classes, insensitive to class imbalance.
- **Weighted avg**: average weighted by class support (may be dominated by bg).
- **Val/Test**: validation is used for model selection; test is held‑out for final reporting.

---

**Bottom line**: emphasize **tsa recall** and **tsa IoU**, include the confusion matrix, and document any balancing strategies used. Keep preprocessing and tile configuration consistent between training and evaluation for numbers that truly compare.
