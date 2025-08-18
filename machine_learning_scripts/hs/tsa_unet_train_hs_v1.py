# ==============================
# U-Net training for HS data using selected bands (no vegetation indices)
# Robust image<->mask pairing by ROI/tile key
# Optional per-band histogram equalization (skimage.exposure)
# ==============================

import os, re, csv
import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt
import seaborn as sns
from time import time
import random as python_random
import tensorflow as tf

from osgeo import gdal
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from skimage import exposure

from keras.utils import to_categorical
from keras.optimizers import Adam
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.metrics import (
    BinaryAccuracy, Precision, Recall, IoU, MeanIoU, FalseNegatives, FalsePositives
)
from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, Conv2DTranspose, concatenate, Dropout, BatchNormalization
from keras.callbacks import ModelCheckpoint

# -------------------------
# User Config
# -------------------------
root_image_folder = r'/home/amarasi5/hpc/tsa/tsa_model_training/sensors_for_modelling/hs/afx_tile'
image_folder_path = os.path.join(root_image_folder, 'hs_rois/training')
mask_folder_path  = os.path.join(root_image_folder, 'mask_rois/training')

# Selected bands (0-based indices)
SELECTED_BANDS_IDX_FILE = r'/home/amarasi5/hpc/tsa/tsa_model_training/sensors_for_modelling/hs/selected_bands_indices.txt'
SELECTED_BANDS_0BASED = None  # e.g., [33,55,77,...]; leave None to read from file; if file missing -> use ALL bands

# Tiling / training
tile_size = 64
overlap_percentage = 0.20
test_size = 0.20
learning_rate = 0.001
batch_size = 8
epochs = 200

# Size filter (optional)
min_width = 0
min_height = 0
max_width = 20000
max_height = 20000

# Classes
n_classes = 2
target_names = ['bg', 'tsa']  # 0: bg, 1: tsa

# Simple smoothing
APPLY_GAUSSIAN = True
APPLY_MEAN = False
GAUSS_KSIZE = (3,3)
MEAN_KSIZE = (3,3)

# ---- NEW: per-band histogram equalization (choose one or none)
APPLY_HE = True          # turn ON/OFF equalization
HE_METHOD = 'hist'       # 'hist' for exposure.equalize_hist, or 'clahe' for exposure.equalize_adapthist
CLAHE_CLIP_LIMIT = 0.01  # used only if HE_METHOD == 'clahe'

# Output root
config_str = f'HS_selectedBands_tile[{tile_size}]_olap[{overlap_percentage}]_ts[{test_size}]_bs[{batch_size}]_ep[{epochs}]_lr[{learning_rate}]'
root_model_folder = os.path.join(root_image_folder, f'tsa_unet_train_hs_model&outcomes_{config_str}')
os.makedirs(root_model_folder, exist_ok=True)
log_dir = os.path.join(root_model_folder, 'log')
os.makedirs(log_dir, exist_ok=True)
audit_csv = os.path.join(root_model_folder, 'dataset_audit.csv')

# -------------------------
# Helpers
# -------------------------
def load_selected_bands_0based():
    """Load 0-based band list from file, or use SELECTED_BANDS_0BASED, else None (ALL bands)."""
    if SELECTED_BANDS_0BASED is not None:
        return sorted(list(set(int(i) for i in SELECTED_BANDS_0BASED)))
    if not os.path.exists(SELECTED_BANDS_IDX_FILE):
        print(f"[WARN] No selected-bands file at: {SELECTED_BANDS_IDX_FILE}. Using ALL bands.")
        return None
    txt = open(SELECTED_BANDS_IDX_FILE, 'r').read().strip()
    txt = txt.strip('[]').strip()
    if not txt:
        print(f"[WARN] Selected-bands file empty. Using ALL bands.")
        return None
    idx = [int(x.strip()) for x in txt.split(',')]
    return sorted(list(set(idx)))

def normalize_tile_minmax(tile):  # (H,W,C) -> per-band [0,1]
    tile = tile.astype(np.float32, copy=False)
    H, W, C = tile.shape
    out = np.zeros_like(tile, dtype=np.float32)
    for c in range(C):
        band = tile[:,:,c]
        bmin, bmax = band.min(), band.max()
        if np.isfinite(bmin) and np.isfinite(bmax) and bmax > bmin:
            out[:,:,c] = (band - bmin) / (bmax - bmin)
        else:
            out[:,:,c] = 0.0
    return out

def equalize_per_band(tile, method='hist', clip_limit=0.01):
    """
    Per-band histogram equalization.
    - 'hist'  -> exposure.equalize_hist
    - 'clahe' -> exposure.equalize_adapthist (CLAHE)
    Returns float32 in [0,1].
    """
    H, W, C = tile.shape
    out = np.zeros((H, W, C), dtype=np.float32)
    for c in range(C):
        band = tile[:, :, c]
        if method == 'clahe':
            out[:, :, c] = exposure.equalize_adapthist(band, clip_limit=clip_limit).astype(np.float32)
        else:  # default 'hist'
            out[:, :, c] = exposure.equalize_hist(band).astype(np.float32)
    return out

def get_image_dimensions(file_path):
    ds = gdal.Open(file_path, gdal.GA_ReadOnly)
    if ds is None:
        return None, None
    return ds.RasterXSize, ds.RasterYSize

_key_re = re.compile(r'roi_(\d+)_tile_(\d+)', re.IGNORECASE)
def roi_tile_key(basename: str) -> str:
    """Return normalized key 'roi_<n>_tile_<m>' from a filename (mask_ prefix ignored)."""
    name = os.path.splitext(os.path.basename(basename))[0]
    if name.lower().startswith('mask_'):
        name = name[5:]
    m = _key_re.search(name)
    return f"roi_{m.group(1)}_tile_{m.group(2)}".lower() if m else name.lower()

def read_tile_selected_bands(ds_img, x_start, y_start, size, selected_bands_0b):
    """Read (H,W,C) from GDAL dataset using only selected bands (0-based). If None, read all bands."""
    bands = ds_img.RasterCount
    if selected_bands_0b is None:
        chosen = list(range(bands))
    else:
        chosen = [b for b in selected_bands_0b if 0 <= b < bands]
        if not chosen:
            raise ValueError("Selected band list is empty or out of range for this dataset.")
    arr = []
    for b0 in chosen:
        arr.append(ds_img.GetRasterBand(b0+1).ReadAsArray(x_start, y_start, size, size))
    tile = np.stack(arr, axis=0)         # (C,H,W)
    tile = np.transpose(tile, (1, 2, 0)) # (H,W,C)
    return tile

def UNet(n_classes, H, W, C):
    inputs = Input((H, W, C))
    seed_value = 22
    np.random.seed(seed_value); tf.random.set_seed(seed_value); python_random.seed(seed_value)

    c1 = Conv2D(64, (3,3), activation="relu", kernel_initializer="he_normal", padding="same")(inputs)
    c1 = BatchNormalization()(c1); c1 = Dropout(0.2)(c1)
    c1 = Conv2D(64, (3,3), activation="relu", kernel_initializer="he_normal", padding="same")(c1)
    c1 = BatchNormalization()(c1); p1 = MaxPooling2D((2,2))(c1)

    c2 = Conv2D(128, (3,3), activation="relu", kernel_initializer="he_normal", padding="same")(p1)
    c2 = BatchNormalization()(c2); c2 = Dropout(0.2)(c2)
    c2 = Conv2D(128, (3,3), activation="relu", kernel_initializer="he_normal", padding="same")(c2)
    c2 = BatchNormalization()(c2); p2 = MaxPooling2D((2,2))(c2)

    c3 = Conv2D(256, (3,3), activation="relu", kernel_initializer="he_normal", padding="same")(p2)
    c3 = BatchNormalization()(c3); c3 = Dropout(0.2)(c3)
    c3 = Conv2D(256, (3,3), activation="relu", kernel_initializer="he_normal", padding="same")(c3)
    c3 = BatchNormalization()(c3); p3 = MaxPooling2D((2,2))(c3)

    c4 = Conv2D(512, (3,3), activation="relu", kernel_initializer="he_normal", padding="same")(p3)
    c4 = BatchNormalization()(c4); c4 = Dropout(0.2)(c4)
    c4 = Conv2D(512, (3,3), activation="relu", kernel_initializer="he_normal", padding="same")(c4)
    c4 = BatchNormalization()(c4); p4 = MaxPooling2D((2,2))(c4)

    c5 = Conv2D(1024, (3,3), activation="relu", kernel_initializer="he_normal", padding="same")(p4)
    c5 = BatchNormalization()(c5); c5 = Dropout(0.2)(c5)
    c5 = Conv2D(1024, (3,3), activation="relu", kernel_initializer="he_normal", padding="same")(c5)
    c5 = BatchNormalization()(c5)

    u6 = Conv2DTranspose(512, (2,2), strides=(2,2), padding="same")(c5)
    u6 = concatenate([u6, c4]); c6 = Conv2D(512, (3,3), activation="relu", kernel_initializer="he_normal", padding="same")(u6)
    c6 = BatchNormalization()(c6); c6 = Dropout(0.2)(c6)
    c6 = Conv2D(512, (3,3), activation="relu", kernel_initializer="he_normal", padding="same")(c6); c6 = BatchNormalization()(c6)

    u7 = Conv2DTranspose(256, (2,2), strides=(2,2), padding="same")(c6)
    u7 = concatenate([u7, c3]); c7 = Conv2D(256, (3,3), activation="relu", kernel_initializer="he_normal", padding="same")(u7)
    c7 = BatchNormalization()(c7); c7 = Dropout(0.2)(c7)
    c7 = Conv2D(256, (3,3), activation="relu", kernel_initializer="he_normal", padding="same")(c7); c7 = BatchNormalization()(c7)

    u8 = Conv2DTranspose(128, (2,2), strides=(2,2), padding="same")(c7)
    u8 = concatenate([u8, c2]); c8 = Conv2D(128, (3,3), activation="relu", kernel_initializer="he_normal", padding="same")(u8)
    c8 = BatchNormalization()(c8); c8 = Dropout(0.2)(c8)
    c8 = Conv2D(128, (3,3), activation="relu", kernel_initializer="he_normal", padding="same")(c8); c8 = BatchNormalization()(c8)

    u9 = Conv2DTranspose(64, (2,2), strides=(2,2), padding="same")(c8)
    u9 = concatenate([u9, c1], axis=3); c9 = Conv2D(64, (3,3), activation="relu", kernel_initializer="he_normal", padding="same")(u9)
    c9 = BatchNormalization()(c9); c9 = Dropout(0.2)(c9)
    c9 = Conv2D(64, (3,3), activation="relu", kernel_initializer="he_normal", padding="same")(c9); c9 = BatchNormalization()(c9)

    outputs = Conv2D(n_classes, (1,1), activation="softmax")(c9)
    return Model(inputs=inputs, outputs=outputs)

# -------------------------
# Build dataset: pair by ROI/tile key
# -------------------------
selected_bands_0b = load_selected_bands_0based()
overlap = int(tile_size * overlap_percentage)
image_patches, mask_patches = [], []

img_files = [os.path.join(image_folder_path, f) for f in os.listdir(image_folder_path) if f.lower().endswith('.tif')]
msk_files = [os.path.join(mask_folder_path,  f) for f in os.listdir(mask_folder_path)  if f.lower().endswith('.tif')]

# index masks by key
mask_index = {}
for mp in msk_files:
    key = roi_tile_key(mp)
    mask_index[key] = mp

image_stems = sorted({roi_tile_key(p) for p in img_files})
print(f"Found {len(image_stems)} unique image stems for training.")

with open(audit_csv, 'w', newline='') as fcsv:
    writer = csv.writer(fcsv)
    writer.writerow(['key','image_path','mask_path','status','reason','image_w','image_h','mask_w','mask_h'])

    kept_pairs = 0
    for ip in img_files:
        key = roi_tile_key(ip)
        mp = mask_index.get(key)
        if mp is None:
            writer.writerow([key, ip, '', 'SKIP', 'no matching mask for key', '', '', '', ''])
            continue

        iw, ih = get_image_dimensions(ip)
        mw, mh = get_image_dimensions(mp)
        if any(v is None for v in (iw, ih, mw, mh)):
            writer.writerow([key, ip, mp, 'SKIP', 'gdal open failed', iw, ih, mw, mh])
            continue
        if not (min_width <= iw <= max_width and min_height <= ih <= max_height):
            writer.writerow([key, ip, mp, 'SKIP', 'image out of size bounds', iw, ih, mw, mh])
            continue
        if (iw != mw) or (ih != mh):
            print(f"[SKIP] Size mismatch: {os.path.basename(ip)} vs {os.path.basename(mp)}")
            writer.writerow([key, ip, mp, 'SKIP', 'size mismatch', iw, ih, mw, mh])
            continue
        if iw < tile_size or ih < tile_size:
            writer.writerow([key, ip, mp, 'SKIP', 'image smaller than tile_size', iw, ih, mw, mh])
            continue

        ds_img = gdal.Open(ip, gdal.GA_ReadOnly)
        ds_msk = gdal.Open(mp, gdal.GA_ReadOnly)
        if ds_img is None or ds_msk is None:
            writer.writerow([key, ip, mp, 'SKIP', 'gdal open failed (second pass)', iw, ih, mw, mh])
            continue

        num_tiles_x = (iw - tile_size) // (tile_size - overlap) + 1
        num_tiles_y = (ih - tile_size) // (tile_size - overlap) + 1
        if num_tiles_x <= 0 or num_tiles_y <= 0:
            writer.writerow([key, ip, mp, 'SKIP', 'no tiles computed', iw, ih, mw, mh])
            continue

        for ty in range(num_tiles_y):
            for tx in range(num_tiles_x):
                x_start = tx * (tile_size - overlap)
                y_start = ty * (tile_size - overlap)

                # image tile
                tile = read_tile_selected_bands(ds_img, x_start, y_start, tile_size, selected_bands_0b)

                # optional smoothing
                if APPLY_GAUSSIAN:
                    for c in range(tile.shape[2]):
                        tile[:, :, c] = cv2.GaussianBlur(tile[:, :, c], GAUSS_KSIZE, 0)
                if APPLY_MEAN:
                    for c in range(tile.shape[2]):
                        tile[:, :, c] = cv2.blur(tile[:, :, c], MEAN_KSIZE)

                # per-band equalization OR minmax
                if APPLY_HE:
                    tile = equalize_per_band(tile, method=HE_METHOD, clip_limit=CLAHE_CLIP_LIMIT)  # -> float32 [0,1]
                else:
                    tile = normalize_tile_minmax(tile)  # -> float32 [0,1]

                # mask tile
                mask = ds_msk.GetRasterBand(1).ReadAsArray(x_start, y_start, tile_size, tile_size).astype(np.int32)

                image_patches.append(tile)
                mask_patches.append(mask)

        kept_pairs += 1
        writer.writerow([key, ip, mp, 'OK', f'tiles_x={num_tiles_x}, tiles_y={num_tiles_y}', iw, ih, mw, mh])

print(f"Pairs kept: {kept_pairs}")
print(f"Tiles kept: {len(image_patches)}")

image_patches = np.array(image_patches, dtype=np.float32)  # (N,H,W,C)
mask_patches  = np.array(mask_patches, dtype=np.int32)     # (N,H,W)

print("image_patches.shape:", image_patches.shape)
print("mask_patches.shape:", mask_patches.shape)

if image_patches.size == 0:
    raise RuntimeError(
        "No usable tiles were created. Check 'dataset_audit.csv' for reasons. "
        "Most common cause is mis-paired files; this script matches by roi/tile key."
    )

# categorical masks
mask_patches_cat = to_categorical(mask_patches, num_classes=n_classes)

# Train/Val split
X_train, X_test, y_train, y_test = train_test_split(
    image_patches, mask_patches_cat, test_size=test_size, random_state=22
)

# Save shapes
with open(os.path.join(root_model_folder, 'unet_train_val_shapes.txt'), 'w') as f:
    f.write(f"X_train: {X_train.shape}\nX_test: {X_test.shape}\n")
    f.write(f"y_train: {y_train.shape}\ny_test: {y_test.shape}\n")
    f.write(f"Image H,W,C: {X_train.shape[1:]}\n")
    f.write(f"APPLY_HE={APPLY_HE}, HE_METHOD={HE_METHOD}\n")

# -------------------------
# Build & compile model
# -------------------------
H, W, C = X_train.shape[1], X_train.shape[2], X_train.shape[3]
model = UNet(n_classes, H, W, C)
model.compile(
    optimizer=Adam(learning_rate=learning_rate),
    loss=BinaryCrossentropy(),
    metrics=[
        BinaryAccuracy(name='binary_accuracy'),
        Precision(class_id=1, name='precision'),
        Recall(class_id=1, name='recall'),
        IoU(num_classes=2, target_class_ids=[1], name='iou'),
        MeanIoU(num_classes=2, name='mean_iou'),
        FalseNegatives(name='false_negatives'),
        FalsePositives(name='false_positives')
    ]
)

# -------------------------
# Train
# -------------------------
best_model_path = os.path.join(root_model_folder, 'unet_best_model.keras')               # Keras v3 model format
weights_path    = os.path.join(log_dir, "weights.{epoch:02d}-{val_loss:.4f}.weights.h5")# weights-only format

checkpoint_best    = ModelCheckpoint(best_model_path, save_best_only=True, monitor='val_loss', mode='min')
checkpoint_weights = ModelCheckpoint(weights_path, save_weights_only=True, save_freq='epoch', verbose=1)

start_time = time()
history = model.fit(
    X_train, y_train,
    batch_size=batch_size,
    epochs=epochs,
    verbose=1,
    validation_data=(X_test, y_test),
    callbacks=[checkpoint_best, checkpoint_weights],
    shuffle=True
)
training_time = time() - start_time
print(f"Training time: {training_time:.2f} s")

# -------------------------
# Evaluation
# -------------------------
y_pred = model.predict(X_test, verbose=1)
y_pred_classes = np.argmax(y_pred, axis=-1)
y_test_classes = np.argmax(y_test, axis=-1)

cm = confusion_matrix(y_test_classes.ravel(), y_pred_classes.ravel())
print(cm)

plt.figure()
sns.heatmap(cm, annot=True, fmt='d', cmap='viridis', xticklabels=target_names, yticklabels=target_names)
plt.title('Confusion Matrix (val)')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.savefig(os.path.join(root_model_folder, 'unet_cm_heatmap_val.png'), bbox_inches='tight', dpi=400)
plt.show()

cr = classification_report(y_test_classes.ravel(), y_pred_classes.ravel(), target_names=target_names)
print(cr)
with open(os.path.join(root_model_folder, 'unet_val_report.txt'), 'w') as f:
    f.write(f"Training Time (s): {training_time:.2f}\n\n")
    f.write("Confusion Matrix:\n")
    f.write(str(cm) + "\n\n")
    f.write("Classification Report:\n")
    f.write(cr + "\n")

# -------------------------
# History plots
# -------------------------
hist = pd.DataFrame(history.history)
hist.to_csv(os.path.join(root_model_folder, 'unet_training_history.csv'), index=False)

def plot_metric(metric_name, title=None, ylabel=None, filename=None):
    if metric_name not in hist.columns or f"val_{metric_name}" not in hist.columns:
        return
    plt.figure(figsize=(10,6))
    plt.plot(hist[metric_name], label='Train')
    plt.plot(hist[f"val_{metric_name}"], label='Val')
    plt.title(title or metric_name)
    plt.xlabel('Epoch'); plt.ylabel(ylabel or metric_name)
    plt.legend()
    if filename:
        plt.savefig(os.path.join(root_model_folder, filename), bbox_inches='tight', dpi=300)
    plt.show()

plot_metric('binary_accuracy', 'Model Accuracy', 'Accuracy', 'Accuracy.png')
plot_metric('loss', 'Model Loss', 'Loss', 'Loss.png')
plot_metric('precision', 'Model Precision (class=1)', 'Precision', 'Precision.png')
plot_metric('recall', 'Model Recall (class=1)', 'Recall', 'Recall.png')
plot_metric('iou', 'IoU (class=1)', 'IoU', 'IoU.png')
plot_metric('mean_iou', 'Mean IoU', 'MeanIoU.png')
plot_metric('false_negatives', 'False Negatives', '#', 'FalseNegatives.png')
plot_metric('false_positives', 'False Positives', '#', 'FalsePositives.png')

print("Done. Outputs saved to:", root_model_folder)
print("Audit CSV:", audit_csv)
