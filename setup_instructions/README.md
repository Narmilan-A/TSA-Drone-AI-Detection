# TSA Model Training — Environment Setup Overview

This project can be run in three typical environments. Below is a quick intro to each, what they’re best for, and what to expect.

## 1) Linux / cloud-based HPC cluster
- **Best for:** Large training runs, multi-GPU nodes, long jobs via schedulers (SLURM, PBS).
- **What you get:** Stable CUDA/cuDNN stack, fast I/O on shared storage, reproducible Conda modules.
- **Notes:** Paths point to shared folders; GDAL needs system libs available; use the `.keras` model format. Copy tiles with verification to avoid zero-filled rasters on network shares.

## 2) Google Colab
- **Best for:** Quick experiments, sanity checks, demos with a free/spot GPU.
- **What you get:** Disposable notebook environment; simple data mounting from Drive; easy plotting.
- **Notes:** Session storage is ephemeral; upload or mount only the tiles you need; adjust paths to `/content/...`. Keep models small enough to download/save between sessions.

## 3) Local machine (desktop/laptop)
- **Best for:** Iteration on code, small datasets, offline development.
- **What you get:** Full control of packages and drivers; convenient debugging and visualization.
- **Notes:** Ensure GDAL is installed consistently with Python. CPU is fine for tests; for GPU, match TensorFlow/CUDA versions carefully. Use the same folder layout as on HPC for drop‑in script compatibility.

---
