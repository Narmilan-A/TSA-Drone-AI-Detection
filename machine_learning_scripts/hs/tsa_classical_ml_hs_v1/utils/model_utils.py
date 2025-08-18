import os, joblib, datetime
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from osgeo import gdal

def create_run_folder(base_folder, model_name):
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_folder = os.path.join(base_folder, model_name, f"run_{timestamp}")
    os.makedirs(run_folder, exist_ok=True)
    return run_folder

def get_latest_run_folder(base_folder, model_name):
    model_base = os.path.join(base_folder, model_name)
    if not os.path.isdir(model_base):
        raise RuntimeError(f"No model folder at {model_base}")
    subs = [os.path.join(model_base, d) for d in os.listdir(model_base) if d.startswith("run_")]
    if not subs:
        raise RuntimeError(f"No run_* folders inside {model_base}")
    return max(subs, key=os.path.getmtime)

def save_hyperparams(cfg, run_folder, model_name):
    hp_file = os.path.join(run_folder, "hyperparameters.txt")
    with open(hp_file, "w") as f:
        f.write(f"Hyperparameters for {model_name}\n\n")
        model_cfg = cfg.get(model_name, {})
        training_cfg = model_cfg.get("training", {})
        for k,v in training_cfg.items():
            f.write(f"{k}: {v}\n")
    print(f"Hyperparameters saved to {hp_file}")

def save_sample_details(details, run_folder, X_train=None, X_test=None, y_train=None, y_test=None, selected_bands=None):
    sample_file = os.path.join(run_folder, "training_samples.txt")
    with open(sample_file, "w") as f:
        f.write("Training samples details:\n\n")
        for d in details:
            f.write(f"Image: {d['image']}, Mask: {d['mask']}, TSA pixels: {d['TSA_pixels']}, BG pixels: {d['BG_pixels']}, Channels: {d.get('channels','?')}\n")
        if X_train is not None:
            f.write("\n--- Dataset Summary ---\n")
            f.write(f"X_train shape: {getattr(X_train,'shape',X_train)}\n")
            f.write(f"y_train shape: {getattr(y_train,'shape',y_train)}\n")
            if X_test is not None:
                f.write(f"X_test shape: {getattr(X_test,'shape',X_test)}\n")
                f.write(f"y_test shape: {getattr(y_test,'shape',y_test)}\n")
            if selected_bands is not None:
                f.write(f"Selected bands (indices): {selected_bands}\n")
                try:
                    f.write(f"Number of channels used: {X_train.shape[1]}\n")
                except Exception:
                    pass
    print(f"Sample details saved to {sample_file}")

def save_report_and_cm(cm, cr, out_dir, prefix):
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir,f"{prefix}_CR_CM_report.txt"),"w") as f:
        f.write("--- Classification Report ---\n\n")
        f.write(cr)
        f.write("\n\n--- Confusion Matrix ---\n\n")
        f.write(str(cm))
    plt.figure(figsize=(7,6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['TSA','BG'], yticklabels=['TSA','BG'])
    plt.xlabel("Predicted"); plt.ylabel("True")
    plt.title(f"{prefix} - Confusion Matrix")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir,f"{prefix}_CM_heatmap.png"),dpi=300)
    plt.close()

def write_palette_geotiff(out_path, arr, template_ds):
    driver = gdal.GetDriverByName("GTiff")
    H, W = arr.shape
    ds = driver.Create(out_path, W, H, 1, gdal.GDT_Byte, options=["COMPRESS=LZW"])
    ds.SetGeoTransform(template_ds.GetGeoTransform())
    ds.SetProjection(template_ds.GetProjection())
    band = ds.GetRasterBand(1)
    ct = gdal.ColorTable()
    ct.SetColorEntry(0, (0,0,0,0))
    ct.SetColorEntry(1, (255,0,0,255))  # TSA
    ct.SetColorEntry(2, (0,255,0,255))  # BG
    band.SetColorTable(ct)
    band.WriteArray(arr)
    band.SetNoDataValue(0)
    band.FlushCache()
    ds = None

def write_envi(out_path_dat, arr, template_ds):
    driver = gdal.GetDriverByName("ENVI")
    H, W = arr.shape
    ds = driver.Create(out_path_dat, W, H, 1, gdal.GDT_Byte)
    ds.SetGeoTransform(template_ds.GetGeoTransform())
    ds.SetProjection(template_ds.GetProjection())
    ds.GetRasterBand(1).WriteArray(arr)
    ds.GetRasterBand(1).SetNoDataValue(0)
    ds.FlushCache()
    ds = None
