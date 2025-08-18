# 🚀 Linux HPC Workflow Guide

This guide provides a step-by-step workflow for connecting to, setting up, and running jobs on a **Linux-based High-Performance Computing (HPC)** system.  
It is general enough to apply to most university or research HPC clusters.

---

## 📑 Table of Contents
1. [Connecting to the HPC](#connecting-to-the-hpc)
2. [Mounting HPC Storage](#mounting-hpc-storage)
3. [Conda Environment Setup](#conda-environment-setup)
4. [GPU and CUDA Setup](#gpu-and-cuda-setup)
5. [Submitting Jobs](#submitting-jobs)
6. [References & Links](#references--links)

---

## Connecting to the HPC

Replace `<username>` and `<hostname.domain>` with your HPC credentials.

### Method 1 – SSH (Linux / macOS / Windows Terminal)
```bash
ssh <username>@<hostname.domain>
```

### Method 2 – PuTTY (Windows only)
1. Install [PuTTY](https://www.putty.org/).  
2. Enter the hostname:
   ```
   <username>@<hostname.domain>
   ```
3. Click **Open**, then login with your password or SSH key.

### Method 3 – VS Code Remote SSH
1. Install the **Remote – SSH** extension in VS Code.  
2. Open the terminal in VS Code and connect with:
   ```bash
   ssh <username>@<hostname.domain>
   ```

---

## Mounting HPC Storage

On Linux or macOS, use SSHFS:

```bash
sshfs <username>@<hostname.domain>:/home/<username> ~/hpc_home
```

On Windows, use:
- [WinFsp](https://github.com/winfsp/winfsp/releases) + [SSHFS-Win](https://github.com/winfsp/sshfs-win).

---

## Conda Environment Setup

### Step 1 – Install Miniconda
```bash
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh -b -p $HOME/miniconda3
```

### Step 2 – Initialize Conda
```bash
$HOME/miniconda3/bin/conda init
echo 'export PATH="$HOME/miniconda3/bin:$PATH"' >> ~/.bashrc
source ~/.bashrc
```

### Step 3 – Create and Activate Environment
```bash
conda create -n tsa_env python=3.10 -y
conda activate tsa_env
```

### Step 4 – Install Libraries
```bash
# TensorFlow (GPU version if CUDA available)
pip install tensorflow[and-cuda]

# Common libraries
pip install opencv-python opencv-python-headless matplotlib seaborn scikit-image xgboost==1.7.5
conda install -c conda-forge gdal
```

### Step 5 – Create from YAML (Optional)
```bash
conda env create -f environment.yml -n myenv
conda activate myenv
```

---

## GPU and CUDA Setup

Check available CUDA modules:
```bash
module avail cuda
```

Load the appropriate version:
```bash
module load CUDA/12.2.0
```

Verify GPU availability:
```bash
python3 -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
```

---

## Submitting Jobs

### Interactive CPU Job
```bash
qsub -I -S /bin/bash -l select=1:ncpus=2:mem=8GB -l walltime=04:00:00
```

### Interactive GPU Job
```bash
qsub -I -S /bin/bash -l select=1:ncpus=4:mem=32g:ngpus=1:gpu_id=A100 -l walltime=12:00:00
```

---

## References & Links
- [Conda Environments](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html)
- [WinFsp](https://github.com/winfsp/winfsp/releases)
- [SSHFS-Win](https://github.com/winfsp/sshfs-win)
- [Miniforge](https://github.com/conda-forge/miniforge)
