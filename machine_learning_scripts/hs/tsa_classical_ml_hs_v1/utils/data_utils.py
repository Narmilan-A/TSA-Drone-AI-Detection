import os, glob
import numpy as np
from osgeo import gdal
from sklearn.utils import resample

def list_pairs(img_dir, msk_dir):
    imgs = sorted(glob.glob(os.path.join(img_dir, "*.tif")))
    pairs = []
    for ip in imgs:
        name = os.path.basename(ip)
        mp = os.path.join(msk_dir, f"mask_{name}")
        if os.path.exists(mp):
            pairs.append((ip, mp))
    return pairs

def compute_top_bands(pairs, class_pair=(1,2), top_k=10):
    if not pairs:
        raise RuntimeError("No image/mask pairs provided to compute_top_bands.")
    first_img = gdal.Open(pairs[0][0])
    if first_img is None:
        raise RuntimeError(f"Failed to open image: {pairs[0][0]}")
    num_bands = first_img.RasterCount
    sums = {cid: np.zeros(num_bands, dtype=np.float64) for cid in class_pair}
    counts = {cid: np.zeros(num_bands, dtype=np.int64) for cid in class_pair}

    for img_path, msk_path in pairs:
        ds = gdal.Open(img_path)
        ms = gdal.Open(msk_path)
        if ds is None or ms is None:
            continue
        arr = np.array([ds.GetRasterBand(i+1).ReadAsArray() for i in range(ds.RasterCount)])
        msk = ms.GetRasterBand(1).ReadAsArray()
        for b in range(num_bands):
            for cid in class_pair:
                pix = arr[b][msk==cid]
                if pix.size>0:
                    sums[cid][b] += pix.sum()
                    counts[cid][b] += pix.size

    means = {cid: np.divide(sums[cid], np.maximum(counts[cid],1), dtype=np.float64) for cid in class_pair}
    diffs = np.abs(means[class_pair[0]] - means[class_pair[1]])

    if top_k > diffs.shape[0]:
        print(f"[WARN] Requested top_k={top_k} > available bands={diffs.shape[0]}. Using {diffs.shape[0]}.")
        top_k = diffs.shape[0]
    selected_bands = np.argsort(diffs)[::-1][:top_k].tolist()
    return selected_bands, int(num_bands)

def build_dataset(pairs, selected_bands):
    X_list, y_list = [], []
    sample_details = []
    for img_path, msk_path in pairs:
        ds = gdal.Open(img_path)
        ms = gdal.Open(msk_path)
        if ds is None or ms is None:
            continue
        img = np.array([ds.GetRasterBand(i+1).ReadAsArray() for i in range(ds.RasterCount)])
        msk = ms.GetRasterBand(1).ReadAsArray()
        X_full = img.transpose(1,2,0).reshape(-1, img.shape[0])[:, selected_bands]
        y_full = msk.flatten()
        idx = np.where((y_full==1)|(y_full==2))[0]
        if idx.size==0:
            continue
        X = X_full[idx]
        y = np.where(y_full[idx]==1,0,1)
        X_list.append(X)
        y_list.append(y)
        sample_details.append({
            "image": os.path.basename(img_path),
            "mask": os.path.basename(msk_path),
            "TSA_pixels": int(np.sum(y==0)),
            "BG_pixels": int(np.sum(y==1)),
            "channels": int(X.shape[1])
        })
    if not X_list:
        raise RuntimeError("No labeled pixels found! Check masks are 1/2 for TSA/BG.")
    return np.concatenate(X_list, axis=0), np.concatenate(y_list, axis=0), sample_details

def balance_classes(X, y, seed=42):
    idx0 = np.where(y==0)[0]
    idx1 = np.where(y==1)[0]
    if idx0.size==0 or idx1.size==0:
        return X, y
    if idx0.size<idx1.size:
        idx0 = resample(idx0, replace=True, n_samples=idx1.size, random_state=seed)
    else:
        idx1 = resample(idx1, replace=True, n_samples=idx0.size, random_state=seed)
    all_idx = np.concatenate([idx0, idx1])
    np.random.seed(seed)
    np.random.shuffle(all_idx)
    return X[all_idx], y[all_idx]
