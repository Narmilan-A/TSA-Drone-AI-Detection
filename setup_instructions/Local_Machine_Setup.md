# Setting Up a Local Development Environment

This guide helps you set up a Python development environment on your local machine using **Anaconda Distribution (or Miniconda), Visual Studio Code (VS Code), and Python Standard Library**.

---

## 📌 Prerequisites
Ensure you have the following installed:

1. **Anaconda Distribution or Miniconda** – [Download Here](https://www.anaconda.com/download)
2. **Visual Studio Code (VS Code)** – [Download Here](https://code.visualstudio.com/)
3. **Python Standard Library** – Included in Anaconda/Miniconda

---

## 🛠️ Installing Anaconda or Miniconda

### **Download and Install**
Choose and download the appropriate installer for your system:

- **Windows / Mac / Linux**: [Anaconda Download](https://www.anaconda.com/download)
- **Miniconda (lightweight alternative)**: [Miniconda Download](https://docs.conda.io/en/latest/miniconda.html)

### **Installation Steps**:
1. Run the installer and follow the default installation settings.
2. Restart your terminal or command prompt after installation.

### **Verify Installation**
Run the following command to check if Conda is installed:
```bash
conda --version
```

---

## 🖥️ Installing Visual Studio Code (VS Code)

- Download VS Code from [here](https://code.visualstudio.com/).
- Install the **Python Extension**: Open VS Code, go to `Extensions` (Ctrl+Shift+X), and search for "Python".

---

## 🌱 Setting Up a Conda Environment

### **Create a New Conda Environment**:
```bash
conda create -n tsa_env python=3.10
```

### **Activate the Environment**:
```bash
conda activate tsa_env
```

### **Verify Available Environments**:
```bash
conda info --envs
```

### **List All Installed Libraries in the Environment**:
```bash
conda list
```

---

## 📦 Installing Python and Required Libraries

### **Install Essential Libraries**:
```bash
conda install numpy pandas matplotlib scikit-learn
```

### **Install Additional Dependencies Using Pip**:
```bash
pip install seaborn opencv-python requests
```

### **Check Installed Packages in the Active Environment**:
```bash
conda list
```

### **Remove an Installed Package**:
```bash
conda remove package_name
```

---

## 🛠️ Setting Up VS Code with Conda Environment

To use the Conda environment in VS Code:
1. Open VS Code.
2. Press `Ctrl+Shift+P`, then search for `Python: Select Interpreter`.
3. Choose the Conda environment (`myenv`).

---

## 🔄 Managing Conda Environments

### **List All Available Conda Environments**:
```bash
conda env list
```

### **Deactivate the Current Environment**:
```bash
conda deactivate
```

### **Delete a Conda Environment**:
```bash
conda remove --name myenv --all
```

---

## ✅ Verifying the Setup
Run the following in your terminal to check installation:
```bash
python -c "import numpy, pandas, matplotlib; print('Setup Successful!')"
```

---

## 📚 Additional Resources
- **Conda Cheat Sheet**: [View Here](https://docs.conda.io/projects/conda/en/latest/user-guide/cheatsheet.html)
- **Python Documentation**: [Read More](https://docs.python.org/3/)
- **VS Code Python Guide**: [Setup Guide](https://code.visualstudio.com/docs/python/python-tutorial)
