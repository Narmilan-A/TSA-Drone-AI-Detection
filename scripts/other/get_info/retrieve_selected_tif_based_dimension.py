# Import general python libraries
import numpy as np
import os
import pandas as pd
import cv2
import matplotlib.pyplot as plt
import seaborn as sns
from skimage import exposure
from scipy.ndimage import convolve
from time import time
import random
import random as python_random
import tensorflow as tf
from PIL import Image

# Import the GDAL module from the osgeo package
from osgeo import gdal

# Import necessary functions from scikit-learn
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report

# Import necessary functions and classes from Keras
from keras.utils import to_categorical
from keras.optimizers import Adam
from keras.losses import categorical_crossentropy
from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, Conv2DTranspose, concatenate, Dropout
from keras.callbacks import ModelCheckpoint
#----------------------------------------------------------------------#
def post_idx_calc(index, normalise):
    # Replace nan with zero and inf with finite numbers
    idx = np.nan_to_num(index)
    if normalise:
        return cv2.normalize(
            idx, None, alpha=0.0, beta=1.0, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    else:
        return idx

# Define function to calculate vegetation indices
def calculate_veg_indices(input_img):
# Extract the all channels from the input image
    RedEdge = input_img[:, :, 3]
    nir = input_img[:, :, 4]
    red = input_img[:, :, 2]
    green = input_img[:, :, 1]
    blue = input_img[:, :, 0]

# Define all the vegetation indices
    # Calculate vegetation indices
    ndvi = post_idx_calc((nir - red) / (nir + red),normalise=True)
    gndvi = post_idx_calc((nir - green) / (nir + green),normalise=True)
    ndre = post_idx_calc((nir - RedEdge) / (nir + RedEdge),normalise=True)
    gci = post_idx_calc((nir)/(green) - 1,normalise=True)
    msavi = post_idx_calc(((2 * nir) + 1 -(np.sqrt(np.power((2 * nir + 1), 2) - 8*(nir - red))))/2,normalise=True)
    exg = post_idx_calc(((2*green)-red-blue)/(red+green+blue),normalise=True)
    sri = post_idx_calc((nir / red),normalise=True)
    arvi = post_idx_calc((nir - (2*red - blue)) / (nir + (2*red - blue)),normalise=True)
    lci = post_idx_calc((nir - RedEdge) / (nir + red),normalise=True)
    hrfi = post_idx_calc((red - blue) / (green + blue),normalise=True)
    dvi = post_idx_calc((nir - red),normalise=True)
    rvi = post_idx_calc((nir)/(red),normalise=True)
    tvi = post_idx_calc((60*(nir - green)) - (100 * (red - green)),normalise=True)
    gdvi = post_idx_calc((nir - green),normalise=True)
    ngrdi = post_idx_calc((green - red) / (green + red),normalise=True)
    grvi = post_idx_calc((red - green) / (red + green),normalise=True)
    rgi = post_idx_calc((red / green),normalise=True)
    endvi = post_idx_calc(((nir + green) - (2 * blue)) / ((nir + green) + (2 * blue)),normalise=True)
    sri = post_idx_calc((nir / red),normalise=True)

    #veg_indices = np.stack((ndvi,ndre,hrfi,gndvi,gci,msavi,exg,sri,arvi,lci, dvi, rvi, tvi, gdvi, ngrdi, grvi, rgi, endvi, sri), axis=2)
    veg_indices = np.stack((ndvi,msavi,exg,arvi, dvi, tvi, gdvi, ngrdi, grvi, endvi), axis=2)
    return veg_indices
#----------------------------------------------------------------------#
# Define a 7x7 low-pass averaging kernel
kernel_size = 3
kernel = np.ones((kernel_size, kernel_size)) / (kernel_size**2)

# Define a function to apply Gaussian blur to an image
def apply_gaussian_blur(img):
    return cv2.GaussianBlur(img, (7,7), 0)
#----------------------------------------------------------------------#
# Define the tile size and overlap percentage
tile_size = 128
overlap = int(tile_size * 0.2)
#----------------------------------------------------------------------#
# Define the root directory with input images and respective masks
root_data_folder = r'F:/scc_final_worksblp_classification/image_mask_rois/training'
root_image_folder = r'F:/scc_final_works/blp_classification/image_mask_rois/training'
root_model_folder =os.path.join(root_data_folder, 'model&outcomes')
# Check if the "model&outcomes" folder exists, and create it if it doesn't
if not os.path.exists(root_model_folder):
    os.makedirs(root_model_folder)
#----------------------------------------------------------------------#
# Store the tiled images and masks
image_patches = []
mask_patches = []

# Import necessary functions from GDAL
from osgeo import gdal

# Define a function to get the width and height of an image using GDAL
def get_image_dimensions(file_path):
    ds = gdal.Open(file_path)
    if ds is not None:
        width = ds.RasterXSize
        height = ds.RasterYSize
        return width, height
    return None, None

# Specify the folder paths for images and masks
image_folder_path = os.path.join(root_image_folder, 'msi_rois')
mask_folder_path = os.path.join(root_image_folder, 'msi_mask_rois')

# Minimum width and height for filtering
min_width = 1000
min_height = 1000

# Filter image and mask files based on dimensions
filtered_image_files = []
filtered_mask_files = []

input_img_folder = os.path.join(root_image_folder, 'msi_rois')
input_mask_folder = os.path.join(root_image_folder, 'msi_mask_rois')

img_files = [file for file in os.listdir(input_img_folder) if file.endswith(".tif")]
mask_files = [file for file in os.listdir(input_mask_folder) if file.endswith(".tif")]

# Iterate through the image files
for img_file in img_files:
    img_path = os.path.join(image_folder_path, img_file)
    img_width, img_height = get_image_dimensions(img_path)
    
    if img_width is not None and img_height is not None:
        if img_width >= min_width and img_height >= min_height:
            filtered_image_files.append(img_path)

# Iterate through the mask files
for mask_file in mask_files:
    mask_path = os.path.join(mask_folder_path, mask_file)
    mask_width, mask_height = get_image_dimensions(mask_path)
    
    if mask_width is not None and mask_height is not None:
        if mask_width >= min_width and mask_height >= min_height:
            filtered_mask_files.append(mask_path)

# Print the number of filtered image and mask files
print(f"Number of filtered image files: {len(filtered_image_files)}")
print(f"Number of filtered mask files: {len(filtered_mask_files)}")


# Sort the filtered files to ensure consistent ordering
filtered_image_files.sort()
filtered_mask_files.sort()

for i in range(len(filtered_image_files)):
    img_file = os.path.basename(filtered_image_files[i])  # Get the file name without the path
    mask_file = os.path.basename(filtered_mask_files[i])  # Get the file name without the path
    
    ds_img = gdal.Open(filtered_image_files[i])
    ds_mask = gdal.Open(filtered_mask_files[i])
    width = ds_img.RasterXSize
    height = ds_img.RasterYSize

    # Calculate the number of tiles in the image
    num_tiles_x = (width - tile_size) // (tile_size - overlap) + 1
    num_tiles_y = (height - tile_size) // (tile_size - overlap) + 1

    for y in range(num_tiles_y):
        for x in range(num_tiles_x):
            # Calculate the tile coordinates
            x_start = x * (tile_size - overlap)
            y_start = y * (tile_size - overlap)
            x_end = x_start + tile_size
            y_end = y_start + tile_size

            # Extract the image tile
            input_bands = 5  # Number of input bands
            input_img = np.array([ds_img.GetRasterBand(j + 1).ReadAsArray(x_start, y_start, tile_size, tile_size) for j in range(input_bands)])
            input_img = np.transpose(input_img, (1, 2, 0))
            input_img = exposure.equalize_hist(input_img)
            
            # veg_indices = calculate_veg_indices(input_img)
            # input_img = np.concatenate((input_img, veg_indices), axis=2)

            #input_img = np.delete(input_img, [0,2], axis=2)

            # for c in range(input_img.shape[2]):
            #     input_img[:, :, c] = convolve(input_img[:, :, c], kernel)

            # input_img = apply_gaussian_blur(input_img)

            input_mask = ds_mask.GetRasterBand(1).ReadAsArray(x_start, y_start, tile_size, tile_size).astype(int)

            # Check if the tile size matches the desired size
            if input_img.shape[:2] == (tile_size, tile_size) and input_mask.shape == (tile_size, tile_size):
                image_patches.append(input_img)
                mask_patches.append(input_mask)

    print(f"Processed image: {img_file} --> Processed mask: {mask_file}")

# Convert the lists to NumPy arrays
image_patches = np.array(image_patches)
mask_patches = np.array(mask_patches)

# Print the shape of the arrays
print("image_patches.shape: {}".format(image_patches.shape))
print("mask_patches.shape: {}".format(mask_patches.shape))

#######################################################
