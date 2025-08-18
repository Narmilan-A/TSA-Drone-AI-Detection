import numpy as np
import matplotlib.pyplot as plt
from osgeo import gdal
import cv2
import os

# Folder paths
roi_folder = 'F:/tsa_model_training/sensors_for_modelling/ms/m3m-ms_tile/ms_rois/training'
mask_folder = 'F:/tsa_model_training/sensors_for_modelling/ms/m3m-ms_tile/mask_rois/training'
# Class names
class_names = {0: 'bg', 1: 'tsa'}


# Define band and index names
band_names = ['green', 'red', 'red_edge', 'nir']
index_names =['ndvi','ndre','gndvi','gci','msavi','sri','lci', 'dvi', 'rvi', 'tvi', 'gdvi', 'ngrdi', 'grvi', 'rgi','osavi','gosavi','ndi','reci','ndwi']


def post_idx_calc(index, normalise):
    # Replace nan with zero and inf with finite numbers
    idx = np.nan_to_num(index)
    if normalise:
        return cv2.normalize(
            idx, None, alpha=0.0, beta=1.0, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    else:
        return idx

# Function to calculate spectral values for a single ROI and mask
def calculate_spectral_values(image_path, mask_path):
    # Open the image and mask
    image_ds = gdal.Open(image_path)
    mask_ds = gdal.Open(mask_path)

    # Convert image and mask to numpy arrays
    image = np.array([image_ds.GetRasterBand(i + 1).ReadAsArray() for i in range(image_ds.RasterCount)])
    mask = mask_ds.GetRasterBand(1).ReadAsArray()

    # Extract bands
    green = image[0]
    red = image[1]
    RedEdge = image[2]
    nir = image[3]

    # Calculate vegetation indices
    ndvi = post_idx_calc((nir - red) / (nir + red),normalise=True)
    gndvi = post_idx_calc((nir - green) / (nir + green),normalise=True)
    ndre = post_idx_calc((nir - RedEdge) / (nir + RedEdge),normalise=True)
    gci = post_idx_calc((nir)/(green) - 1,normalise=True)
    msavi = post_idx_calc(((2 * nir) + 1 -(np.sqrt(np.power((2 * nir + 1), 2) - 8*(nir - red))))/2,normalise=True)
    sri = post_idx_calc((nir / red),normalise=True)
    lci = post_idx_calc((nir - RedEdge) / (nir + red),normalise=True)
    dvi = post_idx_calc((nir - red),normalise=True)
    rvi = post_idx_calc((nir)/(red),normalise=True)
    tvi = post_idx_calc((60*(nir - green)) - (100 * (red - green)),normalise=True)
    gdvi = post_idx_calc((nir - green),normalise=True)
    ngrdi = post_idx_calc((green - red) / (green + red),normalise=True)
    grvi = post_idx_calc((red - green) / (red + green),normalise=True)
    rgi = post_idx_calc((red / green),normalise=True)
    osavi= post_idx_calc((1.16 * (nir - red)) / (nir + red + 0.16),normalise=True)
    gosavi=post_idx_calc((nir - green) / (nir + green + 0.16),normalise=True)
    ndi=post_idx_calc((green - red) / (green + red),normalise=True)
    reci= post_idx_calc((nir) / (RedEdge) - 1,normalise=True)
    ndwi= post_idx_calc((green - nir) / (green + nir),normalise=True)

    # Stack the indices as new bands
    indices_stacked = np.stack([ndvi, ndre, gndvi, gci, msavi, sri, lci, dvi, rvi, tvi, gdvi, ngrdi, grvi, rgi, osavi, gosavi, ndi, reci, ndwi], axis=0)

    # Add the indices bands to the original image
    image = np.concatenate((image, indices_stacked), axis=0)

    # Initialize empty arrays to store spectral values
    spectral_values = {class_id: [] for class_id in class_names}

    # Iterate through each band and compute mean spectral values for each class
    for band_idx in range(image.shape[0]):
        for class_id in class_names:
            class_pixels = image[band_idx][mask == class_id]
            mean_spectral_value = np.mean(class_pixels)
            spectral_values[class_id].append(mean_spectral_value)

    return spectral_values

# Initialize dictionaries to store cumulative spectral values for all ROIs
all_spectral_values = {class_id: [] for class_id in range(len(class_names))}

# Iterate through all ROI and mask files in the folders
for root, dirs, files in os.walk(roi_folder):
    for file in files:
        if file.endswith(".tif"):
            roi_name = file
            mask_name = "mask_" + file

            print(f"Processing ROI: {roi_name} --> Mask: {mask_name}")

            roi_path = os.path.join(root, roi_name)
            mask_path = os.path.join(mask_folder, mask_name)

            # Calculate spectral values for the current ROI and mask
            spectral_values = calculate_spectral_values(roi_path, mask_path)

            # Update the cumulative spectral values
            for class_id in spectral_values:
                all_spectral_values[class_id].append(spectral_values[class_id])

# Calculate the average spectral values for each class
average_spectral_values = {}
for class_id in all_spectral_values:
    class_data = np.array(all_spectral_values[class_id])
    average_spectral_values[class_id] = np.mean(class_data, axis=0)


# Plot only the bands
plt.figure(figsize=(10, 6))
x_labels = band_names
x_positions = np.arange(1, len(x_labels) + 1)
for class_id in class_names:
    plt.plot(x_positions, average_spectral_values[class_id][:len(band_names)], label=f'Class {class_names[class_id]}')
plt.xlabel('Band')
plt.ylabel('Mean Spectral Value')
plt.title('Spectral Signature Curves - Bands Only')
plt.xticks(x_positions, x_labels, rotation=45)
plt.legend()
plt.grid(True)
plt.savefig(os.path.join('F:/tsa_model_training/sensors_for_modelling/ms/m3m-ms_tile/Spectral Signature Curves - Bands Only.png'), bbox_inches='tight')

plt.tight_layout()
plt.show()

# Plot only the indices
plt.figure(figsize=(10, 6))
x_labels = index_names
x_positions = np.arange(1, len(x_labels) + 1)
for class_id in class_names:
    plt.plot(x_positions, average_spectral_values[class_id][len(band_names):len(band_names) + len(index_names)], label=f'Class {class_names[class_id]}')
plt.xlabel('Index')
plt.ylabel('Mean Spectral Value')
plt.title('Spectral Signature Curves - Indices Only')
plt.xticks(x_positions, x_labels, rotation=45)
plt.legend()
plt.grid(True)
plt.savefig(os.path.join('F:/tsa_model_training/sensors_for_modelling/ms/m3m-ms_tile/Spectral Signature Curves - Indices Only.png'), bbox_inches='tight')

plt.tight_layout()
plt.show()


# Plot combination of bands and indices
plt.figure(figsize=(10, 6))
x_labels = index_names + band_names
x_positions = np.arange(1, len(x_labels) + 1)
for class_id in class_names:
    plt.plot(x_positions, average_spectral_values[class_id][:len(index_names) + len(band_names)], label=f'Class {class_names[class_id]}')
plt.xlabel('Band / Index')
plt.ylabel('Mean Spectral Value')
plt.title('Spectral Signature Curves - Bands and Indices')
plt.xticks(x_positions, x_labels, rotation=45)
plt.legend()
plt.grid(True)
plt.savefig(os.path.join('F:/tsa_model_training/sensors_for_modelling/ms/m3m-ms_tile/Spectral Signature Curves - Bands and Indices.png'), bbox_inches='tight')

plt.tight_layout()
plt.show()


# Define the class IDs to compare
class_one = 0
class_another = 1

# Define the number of top items to display for bands and indices
top_items_count_bands = 4
top_items_count_indices = 5

# Calculate the absolute differences between class curves for each band and index
band_differences = {}
index_differences = {}

for band_idx in range(len(band_names)):
    diff = np.abs(average_spectral_values[class_one][band_idx] - average_spectral_values[class_another][band_idx])
    band_differences[band_names[band_idx]] = diff

for index_idx, index_name in enumerate(index_names):
    diff = np.abs(average_spectral_values[class_one][len(band_names) + index_idx] -
                   average_spectral_values[class_another][len(band_names) + index_idx])
    index_differences[index_name] = diff

# Find the top items with the highest differences
top_band_differences = sorted(band_differences.items(), key=lambda x: x[1], reverse=True)[:top_items_count_bands]
top_bands = [band[0] for band in top_band_differences]

top_index_differences = sorted(index_differences.items(), key=lambda x: x[1], reverse=True)[:top_items_count_indices]
top_indices = [index[0] for index in top_index_differences]

# Save print statements to a text file
output_file = "F:/tsa_model_training/sensors_for_modelling/ms/m3m-ms_tile/spectral_difference_results.txt"
with open(output_file, "w") as f:
    f.write(f"Top {top_items_count_bands} bands with the most significant spectral difference between classes {class_one} and {class_another}:\n")
    for band in top_bands:
        f.write(band + "\n")

    f.write("\n")

    f.write(f"Top {top_items_count_indices} indices with the most significant spectral difference between classes {class_one} and {class_another}:\n")
    for index in top_indices:
        f.write(index + "\n")

# Print the results
print(f"Top {top_items_count_bands} bands with the most significant spectral difference between classes {class_one} and {class_another}:")
for band in top_bands:
    print(band)

print(f"\nTop {top_items_count_indices} indices with the most significant spectral difference between classes {class_one} and {class_another}:")
for index in top_indices:
    print(index)

# Print a confirmation message
print(f"Results saved to {output_file}")

# Close the datasets
image_ds = None
mask_ds = None

# Plot only the bands as dots
plt.figure(figsize=(10, 6))
x_labels = band_names
x_positions = np.arange(1, len(x_labels) + 1)
for class_id in class_names:
    plt.scatter(x_positions, average_spectral_values[class_id][:len(band_names)], label=f'Class {class_names[class_id]}', marker='o')
plt.xlabel('Band')
plt.ylabel('Mean Spectral Value')
plt.title('Spectral Signature Curves - Bands Only')
plt.xticks(x_positions, x_labels, rotation=45)
plt.legend()
plt.grid(True)
plt.savefig(os.path.join('F:/tsa_model_training/sensors_for_modelling/ms/m3m-ms_tile/Spectral Signature Curves - Bands Only_2.png'), bbox_inches='tight')
plt.tight_layout()
plt.show()