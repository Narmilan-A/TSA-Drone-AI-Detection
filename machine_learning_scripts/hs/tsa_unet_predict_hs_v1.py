# ==============================
# U-Net PREDICTION for HS data using selected bands (no vegetation indices)
# ==============================

import os
import numpy as np
import cv2
from empatches import EMPatches
from osgeo import gdal
from osgeo.gdalconst import GDT_Byte
from keras.models import load_model
from keras.models import load_model


# -------------------------
# User Config
# -------------------------
# Root of your HS dataset
root_image_folder = r'/home/amarasi5/hpc/tsa/tsa_model_training/sensors_for_modelling/hs/afx_tile'

# Folder of HS images to predict on
input_img_folder = os.path.join(root_image_folder, 'hs_rois/testing')

# Selected bands (0-based) — load from file or set list directly
SELECTED_BANDS_IDX_FILE = r'/home/amarasi5/hpc/tsa/tsa_model_training/sensors_for_modelling/hs/selected_bands_indices.txt'
SELECTED_BANDS_0BASED = None  # e.g., [33, 55, 77, ...]; leave None to read from file

# Trained model path (must match training config)
best_model_path = os.path.join(
    root_image_folder,
    'tsa_unet_train_hs_model&outcomes_HS_selectedBands_tile[64]_olap[0.1]_ts[0.2]_bs[8]_ep[120]_lr[0.001]',
    'unet_best_model.keras'
)

# Patch/merge settings (must match training’s input size)
PATCH_SIZE = 64
EMP_OVERLAP = 0.30  # EMPatches fractional overlap; independent of training overlap used for tiling dataset

# Optional smoothing (only enable if also used during training)
APPLY_GAUSSIAN = False
APPLY_MEAN = False
GAUSS_KSIZE = (3,3)
MEAN_KSIZE = (3,3)

# Classes: 0=bg, 1=tsa
n_classes = 2

# Output folder
prediction_folder = os.path.join(root_image_folder, 'hs_unet_predictions')
os.makedirs(prediction_folder, exist_ok=True)

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

def read_full_selected_bands(ds_img, selected_bands_0b):
    """Read the full scene (H, W, C) using only selected bands (0-based). If None, read all bands."""
    bands = ds_img.RasterCount
    chosen = list(range(bands)) if selected_bands_0b is None else \
             [b for b in selected_bands_0b if 0 <= b < bands]
    if not chosen:
        raise ValueError("Selected band list is empty or out of range.")
    arr = []
    for b0 in chosen:
        arr.append(ds_img.GetRasterBand(b0+1).ReadAsArray())
    cube = np.stack(arr, axis=0)         # (C, H, W)
    cube = np.transpose(cube, (1, 2, 0)) # (H, W, C)
    return cube

def map_labels_to_colors(prediction_2d):
    """0=bg -> black; 1=tsa -> red."""
    h, w = prediction_2d.shape
    out = np.zeros((h, w, 3), dtype=np.uint8)
    out[prediction_2d == 1] = [255, 0, 0]  # TSA in red
    # bg stays black
    return out

# -------------------------
# Load model & bands
# -------------------------
unet_model = load_model(best_model_path)
print(f"Model loaded: {best_model_path}")
selected_bands_0b = load_selected_bands_0based()
print("Selected bands (0-based):", selected_bands_0b if selected_bands_0b is not None else "ALL")

# -------------------------
# Predict over all images
# -------------------------
img_files = [f for f in os.listdir(input_img_folder) if f.lower().endswith('.tif')]
img_files.sort()

total_files = len(img_files)
ignored_files = 0
print(f"Found {total_files} HS ROI(s) to predict.")

for i, fname in enumerate(img_files):
    img_path = os.path.join(input_img_folder, fname)
    ds = gdal.Open(img_path, gdal.GA_ReadOnly)
    if ds is None:
        print(f"[SKIP] GDAL failed to open: {fname}")
        ignored_files += 1
        continue

    # Read the selected bands for the whole scene
    img = read_full_selected_bands(ds, selected_bands_0b)  # (H, W, C)
    H, W, C = img.shape

    if H < PATCH_SIZE or W < PATCH_SIZE:
        print(f"[SKIP] Smaller than patch ({H}x{W}): {fname}")
        ignored_files += 1
        continue

    # Optional smoothing (band-wise)
    if APPLY_GAUSSIAN:
        for c in range(C):
            img[:,:,c] = cv2.GaussianBlur(img[:,:,c], GAUSS_KSIZE, 0)
    if APPLY_MEAN:
        for c in range(C):
            img[:,:,c] = cv2.blur(img[:,:,c], MEAN_KSIZE)

    # EMPatches extraction
    emp = EMPatches()
    patches, indices = emp.extract_patches(img, patchsize=PATCH_SIZE, overlap=EMP_OVERLAP)

    # Make sure each patch is exactly PATCH_SIZE and normalized bandwise
    proc_patches = []
    for p in patches:
        ph, pw, pc = p.shape
        # pad if slightly smaller (edge cases)
        if ph < PATCH_SIZE or pw < PATCH_SIZE:
            tmp = np.zeros((PATCH_SIZE, PATCH_SIZE, pc), dtype=p.dtype)
            tmp[:ph, :pw, :] = p
            p = tmp
        # per-patch min-max normalize
        p = normalize_tile_minmax(p)
        proc_patches.append(p)

    proc_patches = np.asarray(proc_patches, dtype=np.float32)

    # Predict and merge
    preds = unet_model.predict(proc_patches, verbose=0)  # (N, 256, 256, n_classes)
    merged = emp.merge_patches(preds, indices, mode='min')  # (H, W, n_classes)

    pred_classes = np.argmax(merged, axis=-1).astype(np.uint8)  # (H, W)

    # Geo info
    geotransform = ds.GetGeoTransform()
    projection = ds.GetProjection()

    # ===== Save single-band ENVI (class IDs) =====
    envi_dir = os.path.join(prediction_folder, 'envi')
    os.makedirs(envi_dir, exist_ok=True)
    out_envi = os.path.join(envi_dir, os.path.splitext(fname)[0] + '_pred.dat')

    driver_envi = gdal.GetDriverByName('ENVI')
    ds_envi = driver_envi.Create(out_envi, W, H, 1, gdal.GDT_Byte)
    ds_envi.GetRasterBand(1).WriteArray(pred_classes)
    ds_envi.SetGeoTransform(geotransform)
    ds_envi.SetProjection(projection)
    ds_envi = None
    print(f"[ENVI] Saved: {out_envi}")

    # ===== Save color GeoTIFF (RGB) =====
    tif_dir = os.path.join(prediction_folder, 'tif')
    os.makedirs(tif_dir, exist_ok=True)
    out_tif = os.path.join(tif_dir, os.path.splitext(fname)[0] + '_pred.tif')

    color = map_labels_to_colors(pred_classes)  # (H, W, 3)
    driver_tif = gdal.GetDriverByName('GTiff')
    ds_tif = driver_tif.Create(out_tif, W, H, 3, GDT_Byte)
    ds_tif.GetRasterBand(1).WriteArray(color[:,:,0])
    ds_tif.GetRasterBand(2).WriteArray(color[:,:,1])
    ds_tif.GetRasterBand(3).WriteArray(color[:,:,2])
    ds_tif.SetGeoTransform(geotransform)
    ds_tif.SetProjection(projection)
    ds_tif = None
    print(f"[TIFF] Saved: {out_tif}")

print(f"Total HS ROIs: {total_files}")
print(f"Ignored HS ROIs: {ignored_files}")
print("All predictions saved to:", prediction_folder)
