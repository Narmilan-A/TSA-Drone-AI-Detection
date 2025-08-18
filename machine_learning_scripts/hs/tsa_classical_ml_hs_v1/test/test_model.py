# Add project root to sys.path when running directly
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import os, glob, yaml, joblib
from utils import data_utils, model_utils
from sklearn.metrics import confusion_matrix, classification_report

def load_config():
    here = os.path.dirname(os.path.abspath(__file__))
    cfg_path = os.path.join(os.path.dirname(here), "config", "config.yaml")
    with open(cfg_path,"r") as f:
        return yaml.safe_load(f)

def main():
    cfg = load_config()
    data_cfg = cfg["data"]
    model_type = cfg["model_type"]

    # Locate latest run folder for this model
    run_folder = model_utils.get_latest_run_folder(data_cfg["output_folder"], model_type)
    print(f"Using run folder: {run_folder}")

    # Load assets
    model_path = glob.glob(os.path.join(run_folder, f"best_{model_type.lower()}_model.pkl"))[0]
    clf = joblib.load(model_path)
    scaler_path = os.path.join(run_folder, "training_scaler.pkl")
    scaler = joblib.load(scaler_path) if os.path.exists(scaler_path) else None
    with open(os.path.join(run_folder, "selected_bands.txt"), "r") as f:
        selected_bands = eval(f.read())

    # Build test dataset
    pairs = data_utils.list_pairs(data_cfg["test_image_folder"], data_cfg["test_mask_folder"])
    X_test, y_test, sample_details = data_utils.build_dataset(pairs, selected_bands)
    if scaler is not None:
        X_test = scaler.transform(X_test)

    # Predict and evaluate
    y_pred = clf.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    cr = classification_report(y_test, y_pred, target_names=['TSA','Background'])
    print("Confusion Matrix:\n", cm)
    print("\nClassification Report:\n", cr)
    model_utils.save_report_and_cm(cm, cr, run_folder, f"{model_type.lower()}_test")

    # Save sample summary
    sample_file = os.path.join(run_folder, f"{model_type.lower()}_test_samples.txt")
    with open(sample_file, "w") as f:
        f.write("Test samples details:\n\n")
        for d in sample_details:
            f.write(f"Image: {d['image']}, Mask: {d['mask']}, TSA pixels: {d['TSA_pixels']}, BG pixels: {d['BG_pixels']}, Channels: {d['channels']}\n")
        f.write(f"\nTotal test pixels: {X_test.shape[0]}\n")
    print(f"Saved test sample details to {sample_file}")

if __name__ == "__main__":
    main()
