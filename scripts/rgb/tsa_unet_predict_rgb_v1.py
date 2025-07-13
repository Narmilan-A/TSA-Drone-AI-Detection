# Import general python libraries
import numpy as np
import os
import cv2
from skimage import exposure
from scipy.ndimage import convolve
from empatches import EMPatches

# Import the GDAL module from the osgeo package
from osgeo import gdal
from osgeo.gdalconst import GDT_Byte

# Import necessary functions and classes from Keras
from keras.models import load_model
#----------------------------------------------------------------------#
# Set the parameters to control the operations
apply_veg_indices = False 
apply_gaussian = False
apply_mean = False
apply_convolution=False
delete_bands = False  # now set False, since we'll explicitly load only 3 bands

# Specify the bands to be deleted
deleted_bands = [3] if delete_bands else None  # No longer used, but kept for completeness

# Minimum width and height for filtering
min_width = 0
min_height = 0
max_width = 20000
max_height = 20000

tile_size = 256           # Set patch size to 256 to match model input
overlap_percentage = 0.2
test_size=0.20

learning_rate=0.001
batch_size=8
epochs=3

n_classes = 2
target_names = ['bg','tsa']
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
    # Extract all channels from the input image
    red = input_img[:, :, 0]
    green = input_img[:, :, 1]
    blue = input_img[:, :, 2]

    # Calculate vegetation indices
    exg = ((2*green)-red-blue)/(red+green+blue)
    hrfi = (red - blue) / (green + blue)
    ngrdi = (green - red) / (green + red)
    grvi = (red - green) / (red + green)
    rgi = (red / green)
    exr= ((1.4 * red) - green) / (red + green + blue)
    exgr= (((2 * green) - red - blue) / (red + green + blue)) - (((1.4 * red) - green) / (red + green + blue))
    ndi=(green - red) / (green + red)
    gcc= green / (red + green + blue)

    #veg_indices = np.stack((exg,hrfi,ngrdi,grvi,rgi,exr,exgr,ndi,gcc), axis=2)
    veg_indices = np.stack((exg,hrfi,ngrdi,grvi), axis=2)

    return veg_indices
#----------------------------------------------------------------------#
# Define a 3x3 low-pass averaging kernel
kernel_size = 3
kernel = np.ones((kernel_size, kernel_size)) / (kernel_size**2)

# Define a function to apply Gaussian blur to an image
def apply_gaussian_blur(img):
    return cv2.GaussianBlur(img, (3,3), 0)

# Function to apply mean filter in a 3x3 window
def apply_mean_filter(img):
    return cv2.blur(img, (3,3))

#----------------------------------------------------------------------#
# Function to map labels to colors
def map_labels_to_colors(prediction):
    color_mapping = {
        1: [255, 0, 0],    # Red for class 1
        2: [0, 0, 255],    # Blue for class 2
        3: [0, 128, 0],    # Green for class 3
    }
    colored_image = np.zeros((prediction.shape[0], prediction.shape[1], 3), dtype=np.uint8)
    for label, color in color_mapping.items():
        mask = prediction == label
        colored_image[mask] = color
    return colored_image
#----------------------------------------------------------------------#
# Define the root directory with input images and respective masks
root_image_folder = r'/home/amarasi5/hpc/tsa/tsa_model_training/sensors_for_modelling/rgb/p1'

# Count the number of vegetation indices only when apply_veg_indices is True
num_veg_indices = calculate_veg_indices(np.zeros((1, 1, 27))).shape[2] if apply_veg_indices else 0

# Define a configuration string based on the parameter values
config_str = (
    f'tile_[{tile_size}]_o.lap_[{overlap_percentage}]_t.size_[{test_size}]_'
    f'b.size_[{batch_size}]_epochs_[{epochs}]_vis_[{apply_veg_indices}]_num_vi_[{num_veg_indices}]_'
    f'gau_[{apply_gaussian}]_mean_[{apply_mean}]_con_[{apply_convolution}]_'
    f'd._bands_{deleted_bands}_l.rate_[{learning_rate}]'
) if delete_bands else (
    f'tile_[{tile_size}]_o.lap_[{overlap_percentage}]_t.size_[{test_size}]_'
    f'b.size_[{batch_size}]_epochs_[{epochs}]_vis_[{apply_veg_indices}]_num_vi_[{num_veg_indices}]_'
    f'gau_[{apply_gaussian}]_mean_[{apply_mean}]_con_[{apply_convolution}]_del_band[false]_l.rate_[{learning_rate}]'
)

root_model_folder = os.path.join(root_image_folder, f'tsa_unet_train_rgb_model&outcomes_{config_str}')

# Check if the "model&outcomes" folder exists, and create it if it doesn't
if not os.path.exists(root_model_folder):
    os.makedirs(root_model_folder)
#----------------------------------------------------------------------#
# Load multispectral images
images = []
input_img_folder = os.path.join(root_image_folder, 'rgb_rois/testing')

# Retrieve all image files
img_files = [file for file in os.listdir(input_img_folder) if file.endswith(".tif")]
#----------------------------------------------------------------------#
# Load unet model
unet_model = load_model(os.path.join(root_model_folder,'unet_save_best_model.hdf5'))
print("Model loaded")
#----------------------------------------------------------------------#
# Prediction
patch_size = 256     # Updated patch size to 256 to match model input shape
total_files = len(img_files)
ignored_files = 0

for i in range(len(img_files)):
    img_file = os.path.join(input_img_folder, img_files[i])
    img_ds = gdal.Open(img_file)

    # Explicitly load only 3 bands (RGB) to avoid extra alpha band or others
    input_bands = 3
    input_img = np.array([img_ds.GetRasterBand(b+1).ReadAsArray() for b in range(input_bands)], dtype=np.float32)
    input_img = np.transpose(input_img, (1, 2, 0))  # Transpose to (height, width, channels)

    # Apply histogram equalization per channel if needed
    for c in range(input_img.shape[2]):
        input_img[:, :, c] = exposure.equalize_hist(input_img[:, :, c])

    if apply_veg_indices:
        veg_indices = calculate_veg_indices(input_img)
        input_img = np.concatenate((input_img, veg_indices), axis=2)
    
    # No need to delete bands since we only loaded 3 bands
    # if delete_bands:
    #     input_img = np.delete(input_img, deleted_bands, axis=2)

    if apply_convolution:
        for c in range(input_img.shape[2]):
            input_img[:, :, c] = convolve(input_img[:, :, c], kernel)
    
    if apply_gaussian:
        input_img = apply_gaussian_blur(input_img)

    if apply_mean:
        input_img = apply_mean_filter(input_img)

    # Check the image size
    if input_img.shape[0] < patch_size or input_img.shape[1] < patch_size:
        ignored_files += 1
        continue
#----------------------------------------------------------------------#
    # Extract patches using EMPatches
    if input_img.shape[0] >= patch_size and input_img.shape[1] >= patch_size:
        emp = EMPatches()
        img_patches, indices = emp.extract_patches(input_img, patchsize=patch_size, overlap=0.3)

        # Resize patches if needed (should be 256x256)
        resized_patches = []
        for patch in img_patches:
            ph, pw, _ = patch.shape
            if ph < patch_size or pw < patch_size:
                resized_patch = np.zeros((patch_size, patch_size, patch.shape[2]), dtype=np.float32)
                resized_patch[:ph, :pw, :] = patch
            else:
                resized_patch = patch[:patch_size, :patch_size, :]
            resized_patches.append(resized_patch)

        img_patches_processed = unet_model.predict(np.array(resized_patches))
        merged_img_patches_processed = emp.merge_patches(img_patches_processed, indices, mode='min')

        # Retrieve geo information
        geotransform = img_ds.GetGeoTransform()
        projection = img_ds.GetProjection()

        # Reshape the predicted image to 2D
        pred_image = np.argmax(merged_img_patches_processed, axis=-1)

        # Create a new .dat and .hdr file for the predicted image
        driver = gdal.GetDriverByName('ENVI')
        pred_image_file = os.path.splitext(img_files[i])[0] + '_pred.dat'

        # Define the path to the "prediction" folder
        prediction_folder = os.path.join(root_model_folder, 'unet_prediction_rois')

        if not os.path.exists(prediction_folder):
            os.makedirs(prediction_folder)

        pred_image_path = os.path.join(prediction_folder, pred_image_file)

        # Create the output GDAL dataset
        pred_image_ds = driver.Create(pred_image_path, pred_image.shape[1], pred_image.shape[0], 1, gdal.GDT_Byte)

        # Write the predicted image data to the new file
        pred_image_ds.GetRasterBand(1).WriteArray(pred_image)

        # Add spatial reference information
        pred_image_ds.SetGeoTransform(geotransform)
        pred_image_ds.SetProjection(projection)

        # Close the files
        pred_image_ds = None

        print(f"Prediction.dat saved for image {img_files[i]}")
#----------------------------------------------------------------------#
        # Map labels to colors for the .tif file
        colored_prediction = map_labels_to_colors(pred_image)

        tif_file_directory = os.path.join(root_model_folder, 'unet_prediction_rois')
        if not os.path.exists(tif_file_directory):
            os.makedirs(tif_file_directory)

        driver = gdal.GetDriverByName('GTiff')
        pred_image_file = os.path.splitext(img_files[i])[0] + '_pred.tif'
        tif_file_path = os.path.join(tif_file_directory, pred_image_file)

        pred_image_ds = driver.Create(tif_file_path, pred_image.shape[1], pred_image.shape[0], 3, GDT_Byte)
        pred_image_ds.GetRasterBand(1).WriteArray(colored_prediction[:, :, 0])
        pred_image_ds.GetRasterBand(2).WriteArray(colored_prediction[:, :, 1])
        pred_image_ds.GetRasterBand(3).WriteArray(colored_prediction[:, :, 2])
        pred_image_ds.SetGeoTransform(geotransform)
        pred_image_ds.SetProjection(projection)
        pred_image_ds = None
        print(f"prediction.tif saved for image {img_files[i]}")

print(f"Total RGB ROIs: {total_files}")
print(f"Ignored RGB ROIs: {ignored_files}")
print("All predictions saved.")