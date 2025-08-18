# Add project root to sys.path when running directly
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import os, glob, yaml, joblib
import numpy as np
from osgeo import gdal
from utils import model_utils

def load_config():
    here = os.path.dirname(os.path.abspath(__file__))
    cfg_path = os.path.join(os.path.dirname(here), "config", "config.yaml")
    with open(cfg_path,"r") as f:
        return yaml.safe_load(f)

def main():
    cfg = load_config()
    data_cfg = cfg["data"]
    model_type = cfg["model_type"]

    run_folder = model_utils.get_latest_run_folder(data_cfg["output_folder"], model_type)
    print(f"Using run folder: {run_folder}")

    model_path = os.path.join(run_folder, f"best_{model_type.lower()}_model.pkl")
    clf = joblib.load(model_path)
    scaler_path = os.path.join(run_folder, "training_scaler.pkl")
    scaler = joblib.load(scaler_path) if os.path.exists(scaler_path) else None
    with open(os.path.join(run_folder, "selected_bands.txt"), "r") as f:
        selected_bands = eval(f.read())

    img_dir = data_cfg["predict_image_folder"]
    imgs = sorted(glob.glob(os.path.join(img_dir, "*.tif")))
    if not imgs:
        raise RuntimeError("No images found in predict_image_folder.")

    pred_dir = os.path.join(run_folder, "predictions")
    os.makedirs(pred_dir, exist_ok=True)

    for ip in imgs:
        print(f"Predicting {os.path.basename(ip)} ...")
        ds = gdal.Open(ip)
        if ds is None:
            print(f"[WARN] Failed to open {ip}; skipping.")
            continue
        img = np.array([ds.GetRasterBand(i+1).ReadAsArray() for i in range(ds.RasterCount)])
        H, W = img.shape[1], img.shape[2]
        X_full = img.transpose(1,2,0).reshape(-1, img.shape[0])[:, selected_bands]
        if scaler is not None:
            X_full = scaler.transform(X_full)
        y_pred01 = clf.predict(X_full)  # 0=TSA, 1=BG
        out_map = np.where(y_pred01==0, 1, 2).reshape(H, W).astype(np.uint8)  # 1=TSA, 2=BG

        base = os.path.splitext(os.path.basename(ip))[0]
        tif_out = os.path.join(pred_dir, f"{base}_{model_type}_pred.tif")
        dat_out = os.path.join(pred_dir, f"{base}_{model_type}_pred.dat")

        # Always write GeoTIFF; ENVI optional per YAML
        model_utils.write_palette_geotiff(tif_out, out_map, ds)
        if cfg[model_type]["prediction"].get("save_envi", True):
            model_utils.write_envi(dat_out, out_map, ds)
        print(f"Saved: {tif_out} {'and '+dat_out if cfg[model_type]['prediction'].get('save_envi', True) else ''}")

if __name__ == "__main__":
    import numpy as np
    main()
