# ==============================
# U-Net TEST for HS data using selected bands (no vegetation indices)
# ==============================

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from osgeo import gdal
from keras.utils import to_categorical
from keras.models import load_model
from sklearn.metrics import confusion_matrix, classification_report
from keras.models import load_model


# -------------------------
# User Config
# -------------------------
# Root of your HS dataset
root_image_folder = r'/home/amarasi5/hpc/tsa/tsa_model_training/sensors_for_modelling/hs/afx_tile'

# Testing data folders
image_folder_path = os.path.join(root_image_folder, 'hs_rois/testing')
mask_folder_path  = os.path.join(root_image_folder, 'mask_rois/testing')

# Selected bands (0-based) — load from file or set list directly
SELECTED_BANDS_IDX_FILE = r'/home/amarasi5/hpc/tsa/tsa_model_training/sensors_for_modelling/hs/selected_bands_indices.txt'
SELECTED_BANDS_0BASED = None  # e.g., [33, 55, 77, 111, 139, ...]

# Model path (best model from training script)
best_model_path = os.path.join(
    root_image_folder,
    'tsa_unet_train_hs_model&outcomes_HS_selectedBands_tile[64]_olap[0.1]_ts[0.2]_bs[8]_ep[120]_lr[0.001]',
    'unet_best_model.keras'
)

# Tiling params (must match training)
tile_size = 64
overlap_percentage = 0.10

# Class settings
n_classes = 2
target_names = ['bg', 'tsa']  # 0: bg, 1: tsa

# Optional smoothing (keep False to match training unless you also used it there)
APPLY_GAUSSIAN = False
APPLY_MEAN = False
GAUSS_KSIZE = (3,3)
MEAN_KSIZE = (3,3)

# Output folder for test results
test_out_dir = os.path.join(root_image_folder, 'hs_test_results')
os.makedirs(test_out_dir, exist_ok=True)

# -------------------------
# Helpers
# -------------------------
def load_selected_bands_0based():
    """Load list like [0, 1, 5, ...] from file or fallback to SELECTED_BANDS_0BASED."""
    if SELECTED_BANDS_0BASED is not None:
        return sorted(list(set(int(i) for i in SELECTED_BANDS_0BASED)))
    if not os.path.exists(SELECTED_BANDS_IDX_FILE):
        print(f"[WARN] Selected-bands file not found: {SELECTED_BANDS_IDX_FILE}. Using ALL bands.")
        return None
    txt = open(SELECTED_BANDS_IDX_FILE, 'r').read().strip().strip('[]').strip()
    if not txt:
        print(f"[WARN] Selected-bands file empty: {SELECTED_BANDS_IDX_FILE}. Using ALL bands.")
        return None
    idx = [int(x.strip()) for x in txt.split(',')]
    return sorted(list(set(idx)))

def normalize_tile_minmax(tile):  # (H, W, C)
    tile = tile.astype(np.float32, copy=False)
    H, W, C = tile.shape
    out = np.zeros_like(tile, dtype=np.float32)
    for c in range(C):
        band = tile[:,:,c]
        bmin = band.min()
        bmax = band.max()
        if np.isfinite(bmin) and np.isfinite(bmax) and bmax > bmin:
            out[:,:,c] = (band - bmin) / (bmax - bmin)
        else:
            out[:,:,c] = 0.0
    return out

def read_tile_selected_bands(ds_img, x_start, y_start, size, selected_bands_0b):
    """Read tile (H,W,C) using only selected bands (0-based). If None, read all bands."""
    bands = ds_img.RasterCount
    chosen = list(range(bands)) if selected_bands_0b is None else \
             [b for b in selected_bands_0b if 0 <= b < bands]
    if not chosen:
        raise ValueError("Selected band list is empty or out of range.")
    arr = []
    for b0 in chosen:
        band = ds_img.GetRasterBand(b0+1).ReadAsArray(x_start, y_start, size, size)
        arr.append(band)
    tile = np.stack(arr, axis=0)         # (C, H, W)
    tile = np.transpose(tile, (1, 2, 0)) # (H, W, C)
    return tile

def get_image_dimensions(fp):
    ds = gdal.Open(fp, gdal.GA_ReadOnly)
    if ds is None: return None, None
    return ds.RasterXSize, ds.RasterYSize

# -------------------------
# Load model
# -------------------------
unet_model = load_model(best_model_path)
print(f"Model loaded: {best_model_path}")

# -------------------------
# Build test dataset (tiles)
# -------------------------
selected_bands_0b = load_selected_bands_0based()
overlap = int(tile_size * overlap_percentage)

img_files = [f for f in os.listdir(image_folder_path) if f.lower().endswith('.tif')]
msk_files = [f for f in os.listdir(mask_folder_path) if f.lower().endswith('.tif')]
img_files.sort()
msk_files.sort()

if len(img_files) != len(msk_files):
    print(f"[WARN] #images ({len(img_files)}) != #masks ({len(msk_files)}). Will iterate by min length.")
n_pairs = min(len(img_files), len(msk_files))
print(f"Found {n_pairs} image/mask files for testing.")

image_patches, mask_patches = [], []

for i in range(n_pairs):
    img_path = os.path.join(image_folder_path, img_files[i])
    msk_path = os.path.join(mask_folder_path,  msk_files[i])

    iw, ih = get_image_dimensions(img_path)
    mw, mh = get_image_dimensions(msk_path)
    if any(v is None for v in (iw, ih, mw, mh)):
        print(f"[SKIP] Could not open: {img_files[i]} / {msk_files[i]}")
        continue
    if (iw != mw) or (ih != mh):
        print(f"[SKIP] Size mismatch: {img_files[i]} vs {msk_files[i]}")
        continue

    ds_img = gdal.Open(img_path, gdal.GA_ReadOnly)
    ds_msk = gdal.Open(msk_path, gdal.GA_ReadOnly)
    if ds_img is None or ds_msk is None:
        print(f"[SKIP] GDAL failed: {img_files[i]} / {msk_files[i]}")
        continue

    num_tiles_x = (iw - tile_size) // (tile_size - overlap) + 1
    num_tiles_y = (ih - tile_size) // (tile_size - overlap) + 1

    for ty in range(num_tiles_y):
        for tx in range(num_tiles_x):
            x_start = tx * (tile_size - overlap)
            y_start = ty * (tile_size - overlap)

            tile = read_tile_selected_bands(ds_img, x_start, y_start, tile_size, selected_bands_0b)

            if APPLY_GAUSSIAN:
                for c in range(tile.shape[2]):
                    tile[:,:,c] = cv2.GaussianBlur(tile[:,:,c], GAUSS_KSIZE, 0)
            if APPLY_MEAN:
                for c in range(tile.shape[2]):
                    tile[:,:,c] = cv2.blur(tile[:,:,c], MEAN_KSIZE)

            tile = normalize_tile_minmax(tile)

            mask = ds_msk.GetRasterBand(1).ReadAsArray(x_start, y_start, tile_size, tile_size).astype(np.int32)

            image_patches.append(tile)
            mask_patches.append(mask)

    print(f"Processed: {img_files[i]}  |  {msk_files[i]}")

image_patches = np.array(image_patches, dtype=np.float32)  # (N,H,W,C)
mask_patches  = np.array(mask_patches, dtype=np.int32)     # (N,H,W)

print("image_patches.shape:", image_patches.shape)
print("mask_patches.shape:",  mask_patches.shape)

# Save quick shapes
with open(os.path.join(test_out_dir, 'unet_testing_samples.txt'), 'w') as f:
    f.write(f"image_patches.shape: {image_patches.shape}\n")
    f.write(f"mask_patches.shape: {mask_patches.shape}\n")

# -------------------------
# Predict & evaluate
# -------------------------
mask_cat = to_categorical(mask_patches, num_classes=n_classes)

y_pred = unet_model.predict(image_patches, verbose=1)
y_pred_classes = np.argmax(y_pred, axis=-1)
y_true_classes = np.argmax(mask_cat, axis=-1)

cm = confusion_matrix(y_true_classes.ravel(), y_pred_classes.ravel())
print("Confusion matrix:\n", cm)

plt.figure(figsize=(5,4))
sns.heatmap(cm, annot=True, fmt='d', cmap='viridis',
            xticklabels=target_names, yticklabels=target_names)
plt.title('Confusion Matrix (Test)')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.tight_layout()
plt.savefig(os.path.join(test_out_dir, 'unet_cm_heatmap_testing.png'), dpi=400)
plt.show()

cr = classification_report(y_true_classes.ravel(), y_pred_classes.ravel(), target_names=target_names)
print("\nClassification Report:\n", cr)

rep_path = os.path.join(test_out_dir, 'unet_model_testing_performance_report.txt')
with open(rep_path, 'w') as f:
    f.write("Confusion Matrix:\n")
    f.write(str(cm) + "\n\n")
    f.write("Classification Report:\n")
    f.write(cr + "\n")

# -------------------------
# IoU per class + mean IoU
# -------------------------
class_iou = []
with open(rep_path, 'a') as f:
    f.write("\nIoU Results:\n")
    for cls in range(n_classes):
        true_cls = (y_true_classes == cls)
        pred_cls = (y_pred_classes == cls)
        intersection = np.sum(true_cls & pred_cls)
        union = np.sum(true_cls) + np.sum(pred_cls) - intersection
        iou = (intersection / union) if union > 0 else 0.0
        class_iou.append(iou)
        f.write(f"IoU for class {target_names[cls]} ({cls}): {iou:.4f}\n")
        print(f"IoU for class {target_names[cls]} ({cls}): {iou:.4f}")

    mean_iou = float(np.mean(class_iou)) if len(class_iou) > 0 else 0.0
    f.write(f"Average IoU: {mean_iou:.4f}\n")
    print(f"Average IoU: {mean_iou:.4f}")

print("Test results saved to:", test_out_dir)
