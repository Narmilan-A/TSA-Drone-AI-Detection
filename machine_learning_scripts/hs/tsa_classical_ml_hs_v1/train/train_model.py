# train/train_model.py
# Add project root to sys.path when running directly (VS Code / script)
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import os, yaml, joblib, csv
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
import xgboost as xgb
import matplotlib.pyplot as plt

from utils import data_utils, model_utils


# -------------------- helpers --------------------

def load_config():
    here = os.path.dirname(os.path.abspath(__file__))
    cfg_path = os.path.join(os.path.dirname(here), "config", "config.yaml")
    with open(cfg_path, "r") as f:
        return yaml.safe_load(f)

def build_model(model_type, cfg, seed):
    mt = model_type.lower()
    if mt == "randomforest":
        p = cfg["RandomForest"]["training"]
        return RandomForestClassifier(
            n_estimators=p["n_estimators"],
            max_depth=p.get("max_depth", None),
            min_samples_split=p.get("min_samples_split", 2),
            min_samples_leaf=p.get("min_samples_leaf", 1),
            bootstrap=p.get("bootstrap", True),
            class_weight=p.get("class_weight", None),
            n_jobs=-1,
            random_state=seed
        )
    elif mt == "xgboost":
        p = cfg["XGBoost"]["training"]
        return xgb.XGBClassifier(
            objective="binary:logistic",
            n_estimators=p["n_estimators"],
            learning_rate=p["learning_rate"],
            max_depth=p["max_depth"],
            subsample=p["subsample"],
            colsample_bytree=p["colsample_bytree"],
            reg_lambda=p.get("reg_lambda", 1.0),
            reg_alpha=p.get("reg_alpha", 0.0),
            n_jobs=-1,
            random_state=seed
        )
    elif mt == "svm":
        p = cfg["SVM"]["training"]
        return SVC(
            C=p.get("C", 1.0),
            kernel=p.get("kernel", "rbf"),
            gamma=p.get("gamma", "scale"),
            probability=p.get("probability", False),
            random_state=seed
        )
    elif mt == "knn":
        p = cfg["KNN"]["training"]
        return KNeighborsClassifier(
            n_neighbors=p.get("n_neighbors", 7),
            weights=p.get("weights", "distance"),
            algorithm=p.get("algorithm", "auto"),
            leaf_size=p.get("leaf_size", 30),
            p=p.get("p", 2)
        )
    else:
        raise ValueError(f"Unsupported model_type: {model_type}")

def _safe_feature_names(orig_names, n):
    if orig_names is None or len(orig_names) != n:
        return [f"f{i}" for i in range(n)]
    return [str(x) for x in orig_names]

def _save_feature_importances(importances, feature_names, out_csv_path):
    with open(out_csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["feature", "importance"])
        for name, imp in zip(feature_names, importances):
            w.writerow([name, float(imp)])

def _plot_feature_importances(importances, feature_names, out_png_path, title):
    idx = np.argsort(importances)[::-1]
    sorted_imps = np.array(importances)[idx]
    sorted_names = np.array(feature_names)[idx]
    top_k = min(20, len(sorted_imps))

    plt.figure(figsize=(8, 6))
    plt.barh(range(top_k), sorted_imps[:top_k][::-1])
    plt.yticks(range(top_k), sorted_names[:top_k][::-1])
    plt.xlabel("Importance")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_png_path, dpi=200)
    plt.close()

def _evaluate_and_save(clf, X_test, y_test, out_dir, label, target_names):
    y_pred = clf.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    cr = classification_report(y_test, y_pred, target_names=target_names)
    print(f"[{label}] Confusion Matrix:\n", cm)
    print(f"\n[{label}] Classification Report:\n", cr)
    model_utils.save_report_and_cm(cm, cr, out_dir, label)

def _feature_selection_and_retrain(current_model_type, cfg, seed, X_bal, y_bal, run_folder, orig_feature_names):
    """
    Runs ONLY for the currently selected model if it's RandomForest or XGBoost:
      - compute feature importances on full features
      - save CSV/PNG in run_folder/feature_selection/
      - select top_k (default 10 or feature_selection.top_k_features)
      - retrain the same model on those features
      - evaluate and save
    """
    mt = current_model_type.lower()
    if mt not in ["randomforest", "xgboost"]:
        return

    fs_cfg = cfg.get("feature_selection", {})
    top_k = int(fs_cfg.get("top_k_features", 10))
    out_sub = fs_cfg.get("output_subfolder", "feature_selection")
    save_plots = bool(fs_cfg.get("save_plots", True))
    save_csv = bool(fs_cfg.get("save_csv", True))

    out_dir = os.path.join(run_folder, out_sub)
    os.makedirs(out_dir, exist_ok=True)

    print(f"=== Feature Importance & Top-{top_k} Retraining for {mt} (current model only) ===")

    # Separate split for FS
    test_size = cfg[current_model_type]["training"].get("test_size", 0.25)
    X_train, X_test, y_train, y_test = train_test_split(
        X_bal, y_bal, test_size=test_size, random_state=seed
    )

    # Normalize if configured
    scaler_full = None
    if cfg["processing"].get("normalize", True):
        scaler_full = StandardScaler(with_mean=False, with_std=False)
        X_train = scaler_full.fit_transform(X_train)
        X_test  = scaler_full.transform(X_test)
        joblib.dump(scaler_full, os.path.join(out_dir, f"scaler_{mt}_full.pkl"))

    # Train current model on full features
    clf_full = build_model(current_model_type, cfg, seed)
    print(f"Training {mt} (full feature set) for importance...")
    clf_full.fit(X_train, y_train)
    joblib.dump(clf_full, os.path.join(out_dir, f"best_{mt}_model_full.pkl"))
    _evaluate_and_save(clf_full, X_test, y_test, out_dir, f"{mt}_full", ["TSA", "Background"])

    # Get importances
    importances = getattr(clf_full, "feature_importances_", None)
    if importances is None:
        print(f"Warning: {mt} does not expose feature_importances_. Skipping FS.")
        return

    feature_names = _safe_feature_names(orig_feature_names, len(importances))

    if save_csv:
        _save_feature_importances(importances, feature_names, os.path.join(out_dir, f"feature_importance_{mt}.csv"))
    if save_plots:
        _plot_feature_importances(importances, feature_names, os.path.join(out_dir, f"feature_importance_{mt}.png"),
                                  title=f"{mt.upper()} Feature Importance")

    # Top-k selection
    k = min(top_k, len(importances))
    top_idx = np.argsort(importances)[::-1][:k]
    top_names = [feature_names[i] for i in top_idx]
    with open(os.path.join(out_dir, f"top_{k}_features_{mt}.txt"), "w") as f:
        f.write("\n".join(top_names))
    print(f"{mt}: Selected top-{k} features:", top_names)

    # Retrain on top-k
    X_bal_top = X_bal[:, top_idx]
    X_train_t, X_test_t, y_train_t, y_test_t = train_test_split(
        X_bal_top, y_bal, test_size=test_size, random_state=seed
    )

    scaler_top = None
    if cfg["processing"].get("normalize", True):
        scaler_top = StandardScaler(with_mean=False, with_std=False)
        X_train_t = scaler_top.fit_transform(X_train_t)
        X_test_t  = scaler_top.transform(X_test_t)
        joblib.dump(scaler_top, os.path.join(out_dir, f"scaler_{mt}_top{k}.pkl"))

    clf_top = build_model(current_model_type, cfg, seed)
    print(f"Retraining {mt} on top-{k} features...")
    clf_top.fit(X_train_t, y_train_t)
    joblib.dump(clf_top, os.path.join(out_dir, f"best_{mt}_model_top{k}.pkl"))
    _evaluate_and_save(clf_top, X_test_t, y_test_t, out_dir, f"{mt}_top{k}", ["TSA", "Background"])


# -------------------- main --------------------

def main():
    cfg = load_config()
    data_cfg = cfg["data"]
    proc_cfg = cfg["processing"]
    model_type = cfg["model_type"]
    seed = proc_cfg["random_seed"]

    # Create run folder
    run_folder = model_utils.create_run_folder(data_cfg["output_folder"], model_type)
    model_utils.save_hyperparams(cfg, run_folder, model_type)

    # Load training pairs
    train_pairs = data_utils.list_pairs(
        data_cfg["train_image_folder"], data_cfg["train_mask_folder"]
    )
    if not train_pairs:
        raise RuntimeError("No training data found.")

    # First-stage band pre-filter
    top_k_bands = proc_cfg["top_items_count_bands"]
    top_bands, num_bands = data_utils.compute_top_bands(
        train_pairs, class_pair=tuple(proc_cfg["class_pairs"][0]), top_k=top_k_bands
    )
    with open(os.path.join(run_folder, "selected_bands.txt"), "w") as f:
        f.write(str(top_bands))
    print(f"Selected top {top_k_bands} bands (of {num_bands}): {top_bands}")

    # Build dataset
    X, y, sample_details = data_utils.build_dataset(train_pairs, top_bands)
    X_bal, y_bal = data_utils.balance_classes(X, y, seed=seed)

    # Single run according to YAML-selected model
    test_size = cfg[model_type]["training"].get("test_size", 0.25)
    X_train, X_test, y_train, y_test = train_test_split(
        X_bal, y_bal, test_size=test_size, random_state=seed
    )

    scaler = None
    if proc_cfg.get("normalize", True):
        scaler = StandardScaler(with_mean=False, with_std=False)
        X_train = scaler.fit_transform(X_train)
        X_test  = scaler.transform(X_test)
        joblib.dump(scaler, os.path.join(run_folder, "training_scaler.pkl"))

    model_utils.save_sample_details(
        sample_details, run_folder, X_train, X_test, y_train, y_test, top_bands
    )

    clf = build_model(model_type, cfg, seed)
    print(f"Training {model_type}...")
    clf.fit(X_train, y_train)
    joblib.dump(clf, os.path.join(run_folder, f"best_{model_type.lower()}_model.pkl"))
    print(f"{model_type} training complete. Model saved to {run_folder}")

    # Evaluate
    y_pred = clf.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    cr = classification_report(y_test, y_pred, target_names=["TSA", "Background"])
    print("Confusion Matrix:\n", cm)
    print("\nClassification Report:\n", cr)
    model_utils.save_report_and_cm(cm, cr, run_folder, model_type.lower())

    # Feature selection ONLY for current model (RF/XGB)
    feature_names = [str(b) for b in top_bands]
    _feature_selection_and_retrain(
        current_model_type=model_type,
        cfg=cfg,
        seed=seed,
        X_bal=X_bal,
        y_bal=y_bal,
        run_folder=run_folder,
        orig_feature_names=feature_names,
    )

if __name__ == "__main__":
    main()
