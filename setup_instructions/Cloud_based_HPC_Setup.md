# Working with Cloud-based High Performance Computing (Linux) – Workflow

## 1. Connecting to an HPC Host
You can connect to a Linux-based HPC system in several ways. Replace `<username>`, `<hostname>`, and `<domain>` with the details provided by your HPC provider.  

- **Method 1 – PuTTY (Windows only)**  
  1. Install [PuTTY](https://www.putty.org/).  
  2. In the *Host Name* field, enter:  
     ```
     <username>@<hostname>.<domain>
     ```  
  3. Click **Open**, then enter your password.  

- **Method 2 – Command line (Linux/macOS/Windows Terminal)**  
  ```bash
  ssh <username>@<hostname>.<domain>
  ```  
  Then enter your password (or use SSH keys if configured).  

- **Method 3 – Visual Studio Code Remote-SSH**  
  1. Install the **Remote – SSH** extension.  
  2. Open VS Code terminal and run:  
     ```bash
     ssh <username>@<hostname>.<domain>
     ```

---

## 2. Mounting your HPC Home Folder
Some HPC systems allow you to mount your home or project directory as a network drive (Windows/macOS/Linux). Options include:  
- [SSHFS](https://github.com/libfuse/sshfs) (Linux/macOS)  
- [SSHFS-Win](https://github.com/winfsp/sshfs-win) (Windows users)  
- [WinFsp](https://github.com/winfsp/winfsp/releases)  

---

## 3. Conda Setup
### Step 1: Download Miniconda
```bash
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
```

### Step 2: Install Miniconda
```bash
bash Miniconda3-latest-Linux-x86_64.sh -b -p $HOME/miniconda3
```

### Step 3: Initialise Conda
```bash
$HOME/miniconda3/bin/conda init
```

### Step 4: Update `.bashrc`
```bash
echo 'export PATH="$HOME/miniconda3/bin:$PATH"' >> ~/.bashrc
source ~/.bashrc
```

### Step 5: Create and Activate an Environment
```bash
conda create --name tsa_env python=3.10 -y
conda activate tsa_env
```

### Step 6: Install Libraries
- TensorFlow (GPU):  
  ```bash
  pip install tensorflow[and-cuda]
  ```
- TensorFlow (CPU):  
  ```bash
  pip install tensorflow
  ```
- OpenCV:  
  ```bash
  pip install opencv-python opencv-python-headless
  ```
- Matplotlib & Seaborn:  
  ```bash
  pip install matplotlib seaborn
  ```
- scikit-image:  
  ```bash
  pip install scikit-image
  ```
- GDAL:  
  ```bash
  conda install -c conda-forge gdal
  ```
- XGBoost:  
  ```bash
  pip install xgboost==1.7.5
  ```

### Step 7: Using `environment.yml` (Optional)
```bash
conda env create -f environment.yml
conda activate myenv
```

---

## 4. Submitting Jobs on HPC

### CPU-only interactive job
```bash
qsub -I -S /bin/bash -l select=1:ncpus=1:mem=4GB -l walltime=12:00:00
```

### GPU interactive jobs (examples)
```bash
# Request A100
qsub -I -S /bin/bash -l select=1:ncpus=4:mem=32g:ngpus=1:gpu_id=A100 -l walltime=12:00:00

# Request H100
qsub -I -S /bin/bash -l select=1:ncpus=4:mem=32g:ngpus=1:gpu_id=H100 -l walltime=12:00:00
```

### Batch job script (example)
```bash
#!/bin/bash -l
#PBS -N My_GPU_Job
#PBS -l select=1:ncpus=4:ngpus=1:mem=32g:gpu_id=A100
#PBS -l walltime=12:00:00

module load cuda
conda activate myenv
python myscript.py
```

---

## 5. Useful HPC Commands
- `qsub` – submit jobs  
- `qdel` – cancel jobs  
- `qstat` – check job status  
- `qjobs` – list your jobs  
- `pbsnodes` – check available nodes  

---

## 6. Check System Info
Check hostname:
```python
import socket
print(socket.gethostname())
```

Check IP address:
```python
import socket
remote_host = "<hostname>"
print(socket.gethostbyname(remote_host))
```

Check GPUs:
```bash
nvidia-smi -L
```

Check CPUs:
```bash
lscpu | grep "^CPU(s):"
```

---

# Linux Terminal Commands (Quick Reference)

### Directory Navigation
```bash
pwd       # current directory
ls -l     # list files with details
cd ..     # go up one level
cd ~      # home directory
```

### File Management
```bash
mkdir mydir          # create directory
rm file.txt          # remove file
rm -rf mydir         # remove folder
cp file1 file2       # copy file
mv old new           # move/rename
```

### Viewing Files
```bash
cat file.txt         # full contents
head -n 5 file.txt   # first 5 lines
tail -n 5 file.txt   # last 5 lines
more file.txt        # paginated view
```
