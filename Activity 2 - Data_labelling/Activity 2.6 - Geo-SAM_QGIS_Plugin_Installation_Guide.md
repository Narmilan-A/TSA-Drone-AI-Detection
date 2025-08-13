# Geo-SAM QGIS Plugin Installation Guide

## Introduction

Geo-SAM is a QGIS plugin designed to help users segment, delineate, or label landforms efficiently when working with large-size geospatial raster images. It leverages the power of the **Segment Anything Model (SAM)**, a foundation AI model with great capabilities. However, the SAM model is very large and can be slow to process images, even on modern GPUs. Geo-SAM optimizes this process for practical use within QGIS.

---

## Install Python Dependencies - For Windows Users

1. Open the **OSGeo4W Shell** (`OsGeo4WShell`) as **Administrator** from the Start menu. This is a dedicated shell for QGIS.

2. If your PC has an NVIDIA GPU, to accelerate encoding you need to:
   - Download and install the **CUDA Toolkit** first.
   - Then install the GPU-enabled PyTorch version using the following command (example uses CUDA 12.4):

   ```bash
   pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
   ```

3. Install the other necessary dependencies:

   ```bash
   pip3 install torchgeo segment-anything
   ```

---

## Install the Geo-SAM Plugin

### Locate the QGIS Plugin Folder

1. In QGIS, navigate to:

   ```
   Settings > User Profiles > Open active profile folder
   ```

2. This will open the profile directory. Inside, look for a `python` folder.  
   - Inside the `python` folder, there should be a `plugins` folder.  
   - If the `plugins` folder does not exist, create it.

3. Example plugin folder paths:

   - **Windows:**  
     `%APPDATA%\QGIS\QGIS3\profiles\default\python\plugins`

---

### Download Plugin and Configure Folder

#### 1. Download the Plugin

You have two options to download the plugin:

- **Manually download the plugin ZIP file**  
- **Clone the repository using git**

##### From ZIP file

- Download the compressed source code from the GitHub releases page:  
  [coolzhao/Geo-SAM](https://github.com/coolzhao/Geo-SAM/releases)
- Unzip the downloaded file.
- Rename the extracted folder to `Geo-SAM`.

##### Update Plugin

- You can also download the `*_code_update.zip` file for the latest code updates.  
- This file contains only the code, reducing file size and download time.

#### 2. Configure the Plugin Folder

- Place the entire `Geo-SAM` folder inside the `plugins` directory.
- The directory structure should look like this:

```
python
└── plugins
    └── Geo-SAM
       ├── checkpoint
       ├── docs
       ├── ...
       ├── tools
       └── ui
```

> **Warning:**  
> Make sure the folder is named exactly `Geo-SAM`.  
> Avoid names like `Geo-SAM-v1.3.1` or `Geo-SAM-v1.3-rc`.  
> Also, beware of nested folders like `Geo-SAM-v1.3.1/Geo-SAM/...` after unzipping—move the inner `Geo-SAM` folder directly into the `plugins` folder.  
> For details, see [coolzhao/Geo-SAM#22](https://github.com/coolzhao/Geo-SAM/issues/22).

---

### Activate the Geo-SAM Plugin

1. Restart QGIS.
2. Go to:

   ```
   Plugins > Manage and Install Plugins
   ```

3. Under **Installed**, find the **Geo SAM** plugin.
4. Check it to activate.

---

You are now ready to use the Geo-SAM plugin within QGIS to segment and label your geospatial raster images efficiently with GPU acceleration support.

---

## Useful Links

- **Video Tutorial:** [Segment Anything in QGIS with the Geo-SAM Plugin](https://www.youtube.com/watch?v=GSKmK7qERUw&ab_channel=HansvanderKwast)
- **Geo-SAM Official GitHub:** [https://github.com/coolzhao/Geo-SAM](https://github.com/coolzhao/Geo-SAM)
- **Geo-SAM Plugin Download:** [https://github.com/coolzhao/Geo-SAM/releases](https://github.com/coolzhao/Geo-SAM/releases)
- **Installation Guide:** [https://geo-sam.readthedocs.io/en/latest/installation.html](https://geo-sam.readthedocs.io/en/latest/installation.html)
- **Usage of Geo-SAM Tools:** [https://geo-sam.readthedocs.io/en/latest/Usage/index.html](https://geo-sam.readthedocs.io/en/latest/Usage/index.html)
- **Download SAM Checkpoints:** [https://geo-sam.readthedocs.io/en/latest/Usage/encoding.html](https://geo-sam.readthedocs.io/en/latest/Usage/encoding.html)


