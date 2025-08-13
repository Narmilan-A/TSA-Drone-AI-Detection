# Working with QUT's HPC - Workflow

## Connecting to HPC
   - Method 1 - Install Putty and set aqua@qut.edu.au as host name.
   - Method 2 - In the windows terminal, type
   ```
   ssh <username>@aqua.qut.edu.au
   ``` 
   - Then enter password.
   - Method 3 - Open VS Code --> Terminal --> follow Method 2.

## Mounting your HPC Home Folder
- [Create local network drive and mount with HPC](https://qutvirtual4.qut.edu.au/group/research-students/conducting-research/specialty-research-facilities/advanced-research-computing-storage/supercomputing/using-hpc-filesystems#h2-0)

## Conda setup
### Step 1: Download Miniconda
```
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
```
### Step 2: Run the installer and specify a valid home directory
```
bash Miniconda3-latest-Linux-x86_64.sh -b -p $HOME/miniconda3
```
### Step 3: Initialise Conda

```
$HOME/miniconda3/bin/conda init
```

### Step 4: Update .bashrc
```
echo 'export PATH="$HOME/miniconda3/bin:$PATH"' >> ~/.bashrc
source ~/.bashrc
```

### Step 5: Conda create and activate

#### Step 5.1: Load Conda if not already loaded
```
source $HOME/miniconda3/bin/activate
```

```
conda init
```
#### Step 5.2: Create and Activate the Environment

```
conda create --name myenv python=3.10 -y
```
```
conda activate myenv
```

#### Step 5.3: Installation of required  Libraries

##### Install TensorFlow

###### For GPU users
```
pip install tensorflow[and-cuda]
```
###### For CPU users
```
pip install tensorflow
```

##### Load the CUDA Module

###### If your HPC uses module for environment management (common), run:
```
module avail cuda
```
###### Then load the correct version:

```
module load CUDA/12.8.0
```
###### Re-check GPU Access in Python
```
python3 -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
```

#### Install other required libraries

###### OpenCV
```
pip install opencv-python opencv-python-headless
```
###### Matplotlib
```
pip install matplotlib
```
###### Seaborn
```
pip install seaborn
```
###### scikit-image
```
pip install scikit-image
```
###### GDAL
```
conda install -c conda-forge gdal
```
###### XGBoost
```
pip install xgboost==1.7.5
```

#### Note: Create the Environment using existing libraries  (Optional)
Open your terminal or command prompt, navigate to the directory containing your environment.yml file, and execute the following command
```
conda env create -f environment.yml
```

This command reads the environment.yml file and creates the environment as specified. If the environment.yml file includes a name field, that name will be used for the environment. Otherwise, you can specify a name using the -n or --name option:
```
conda env create -f environment.yml -n myenv
```

Activate the Environment: After creation, activate the new environment 
```
conda activate myenv
```

## 6. Submitting Jobs on GPU Nodes (Example code)

CPU-Only Interactive Jobs
You can request an interactive CPU-only session using the following command:
   ```
   qsub -I -S /bin/bash -l select=1:ncpus=1:mem=4GB -l walltime=12:00:00
   ``` 

## ✅ GPU Update on the HPC

### Item  
Good news — the **NVIDIA A100 GPU nodes** have been migrated from **Lyra** to **Aqua**, increasing the number of GPUs available on the HPC.

> **Note:** If you are running interactive or batch jobs and **do not specify** a particular GPU type, the scheduler will allocate **any available GPU** (A100 or H100).

---

### ℹ️ Background

When running batch or interactive jobs on the HPC, you can choose the type of GPU you want to use:

- **H100s** are newer and faster.
- **A100s** are powerful GPUs recently migrated from Lyra to Aqua.

---

## 🛠️ Instructions

To request a **specific GPU model**, set the `gpu_id` resource to either `A100` or `H100`.

---

### 💻 Interactive Jobs

**A100:**
```bash
qsub -I -S /bin/bash -l select=1:ncpus=4:mem=32g:ngpus=1:gpu_id=A100 -l walltime=12:00:00
```

**H100:**
```bash
qsub -I -S /bin/bash -l select=1:ncpus=4:mem=32g:ngpus=1:gpu_id=H100 -l walltime=12:00:00
```

```bash
qsub -I -S /bin/bash -l select=1:ncpus=32:mem=128g:ngpus=1:gpu_id=H100 -l walltime=24:00:00
```

---

### 📦 Batch Jobs

**A100:**
```bash
#!/bin/bash -l
#PBS -N My_A100_Job
#PBS -l select=1:ncpus=4:ngpus=1:mem=32g:gpu_id=A100
```

**H100:**
```bash
#!/bin/bash -l
#PBS -N My_H100_Job
#PBS -l select=1:ncpus=4:ngpus=1:mem=32g:gpu_id=H100
```

---

## 7. List of Useful Commands
   - `qsub`.
   - `qdel`.
   - `qstat`.
   - `pbsusage`.
   - `qjobs`.
   - `pbsnodes`.

## 8. Useful Website Links
   - [QUT HPC Aqua](https://docs.eres.qut.edu.au/major-changes-early-adopters).
   - [QUT's HPC Facilities](https://wiki.qut.edu.au/pages/viewpage.action?spaceKey=cyphy&title=Working+with+QUT%27s+HPC+%28High+Performance+Computing%29+Facilities)
   - [Map a Network Drive in Windows](https://qutvirtual4.qut.edu.au/group/research-students/conducting-research/specialty-research-facilities/advanced-research-computing-storage/supercomputing/using-hpc-filesystems#h2-0)
   - [SSHFS-Win GitHub Repository](https://github.com/winfsp/sshfs-win)
   - [WinFsp Releases](https://github.com/winfsp/winfsp/releases/tag/v1.12.22339)
   - [Miniforge GitHub Repository](https://github.com/conda-forge/miniforge)
   - [Managing Environments with Conda](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html#creating-an-environment-with-commands)

## 9. Check Host Name
```python
import socket
hostname = socket.gethostname()
print(hostname)
```

## 10. Check IP address
```python
import socket
remote_host = "lyra01"
try:
    remote_ip = socket.gethostbyname(remote_host)
    print(f"The IP address of {remote_host} is {remote_ip}")
except socket.gaierror:
    print(f"Could not get the IP address of {remote_host}")
```

## 11. Check GPU and CPU number
```python
import paramiko
```

### Create an SSH client
```python
client = paramiko.SSHClient()
client.load_system_host_keys()
client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
```

### Connect to the remote server
```python
client.connect('10.13.0.10', username='n10837647', password='NMilan!@2023')
```

### Run the nvidia-smi command on the remote server to check the number of GPUs
```python
stdin, stdout, stderr = client.exec_command('nvidia-smi -L')
num_gpus = len(stdout.readlines())
print(f"Number of GPUs: {num_gpus}")
```

### Run the lscpu command on the remote server to check the number of CPUs
```python
stdin, stdout, stderr = client.exec_command('lscpu | grep "^CPU(s):"')
output = stdout.read().decode()
num_cpus = output.strip().split()[1]
print(f"Number of CPUs: {num_cpus}")
```

### Close the SSH client
```python
client.close()
```
# Linux Terminal Commands

## Basic Linux Commands
This document provides a curated list of essential Linux commands for navigating the High-Performance Computing (HPC) environment. These commands will help you manage files, directories, and perform basic system operations.

## Get Info About Commands
Most Linux commands have manual pages with detailed instructions. Use the following command to access them:

```sh
man <command_name>
```

## Navigating the Directory Structure
### Current Directory
Get the present working directory:
```sh
pwd
```
## Navigating the exact Folder

```
cd ~/hpc/tsa/tsa_model_training/scripts/rgb
```
### List Files
List files in the current directory:
```sh
ls
```
List a specific file within a directory:
```sh
ls a.out
```
List files with details (permissions, ownership, etc.):
```sh
ls -l
```

### Change Directory
Change into a directory:
```sh
cd public_html
```
Move one level up:
```sh
cd ..
```
Navigate using absolute path:
```sh
cd ~/public_html/
```
Return to home directory:
```sh
cd ~
```

## Managing Directories and Files
### Create a Directory
Create a new directory:
```sh
mkdir testdir
```
Create nested directories:
```sh
mkdir -p testdir/test
```

### Remove Files and Directories
Remove a file:
```sh
rm testfile
```
Remove a directory and its contents (with prompt):
```sh
rm -ri testdir
```
Forcefully remove a directory and its contents:
```sh
rm -rf testdir
```
Remove an empty directory:
```sh
rmdir NextDirectoryDown/
```

## Copy and Move Files
### Copy Files
Copy a file:
```sh
cp <sourcefile> <destinationfile>
```
Copy a directory recursively:
```sh
cp -r <sourcedir> <destinationdir>
```

### Move and Rename Files
Move a file to a new directory:
```sh
mv file1 ./destinationdir/
```
Rename a file:
```sh
mv file1 file2
```

## Viewing File Contents
### Display Full File Content
```sh
cat daysofweek.txt
```

### View File End (Tail)
```sh
tail -n 3 animals
```
Monitor file updates interactively:
```sh
tail -f logfile.log
```

### View File Start (Head)
```sh
head -n 2 animals
```

### Paginated File Viewing
View file page by page:
```sh
more filename.txt
```
