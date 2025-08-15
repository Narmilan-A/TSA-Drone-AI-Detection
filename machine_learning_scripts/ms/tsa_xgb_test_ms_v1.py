# Import general python libraries
import numpy as np
import matplotlib.pyplot as plt
import joblib
from skimage import exposure
import seaborn as sns
import os
import cv2
from scipy.ndimage import convolve

# Import the GDAL module from the osgeo package
from osgeo import gdal

# Import necessary functions from scikit-learn
from sklearn.metrics import confusion_matrix, classification_report
#----------------------------------------------------------------------#
# Define function to calculate vegetation indices
def calculate_veg_indices(input_img):
# Extract the all channels from the input image
    # RedEdge = input_img[:, :, 3]
    # nir = input_img[:, :, 4]
    # red = input_img[:, :, 2]
    # green = input_img[:, :, 1]
    # blue = input_img[:, :, 0]

    RedEdge = input_img[:, :, 2]
    nir = input_img[:, :, 3]
    red = input_img[:, :, 1]
    green = input_img[:, :, 0]

    # Calculate vegetation indices
    ndvi = (nir - red) / (nir + red)
    gndvi = (nir - green) / (nir + green)
    ndre = (nir - RedEdge) / (nir + RedEdge)
    gci = (nir)/(green) - 1
    msavi = ((2 * nir) + 1 -(np.sqrt(np.power((2 * nir + 1), 2) - 8*(nir - red))))/2
    #exg = ((2*green)-red-blue)/(red+green+blue)
    sri = (nir / red)
    #arvi = (nir - (2*red - blue)) / (nir + (2*red - blue))
    lci = (nir - RedEdge) / (nir + red)
    #hrfi = (red - blue) / (green + blue)
    dvi = (nir - red)
    rvi = (nir)/(red)
    tvi = (60*(nir - green)) - (100 * (red - green))
    gdvi = (nir - green)
    ngrdi = (green - red) / (green + red)
    grvi = (red - green) / (red + green)
    rgi = (red / green)
    #endvi = ((nir + green) - (2 * blue)) / ((nir + green) + (2 * blue))
   # evi=(2.5 * (nir - red)) / (nir + (6 * red) - (7.5 * blue) + 1)
   # sipi= (nir - blue) / (nir - red)
    osavi= (1.16 * (nir - red)) / (nir + red + 0.16)
    gosavi=(nir - green) / (nir + green + 0.16)
   # exr= ((1.4 * red) - green) / (red + green + blue)
   # exgr= (((2 * green) - red - blue) / (red + green + blue)) - (((1.4 * red) - green) / (red + green + blue))
    ndi=(green - red) / (green + red)
   # gcc= green / (red + green + blue)
    reci= (nir) / (RedEdge) - 1
    ndwi= (green - nir) / (green + nir)

    #veg_indices = np.stack((ndvi,ndre,hrfi,gndvi,gci,msavi,exg,sri,arvi,lci, dvi, rvi, tvi, gdvi, ngrdi, grvi, rgi, endvi, evi,sipi,osavi,gosavi,exr,exgr,ndi,gcc,reci,ndwi), axis=2)
    veg_indices = np.stack((ndvi, msavi, ngrdi, grvi, osavi, ), axis=2)

    return veg_indices
#----------------------------------------------------------------------#
# Define a 7x7 low-pass averaging kernel
kernel_size = 7
kernel = np.ones((kernel_size, kernel_size)) / (kernel_size**2)

# Define a function to apply Gaussian blur to an image
def apply_gaussian_blur(img):
    return cv2.GaussianBlur(img, (7,7), 0)
#----------------------------------------------------------------------#
root_data_folder = r'/home/amarasi5/hpc/tsa/tsa_model_training/sensors_for_modelling/ms/m3m-ms_tile'

root_model_folder =os.path.join(root_data_folder, 'xgb_m3m-ms_model&outcomes_1')
# Check if the "model&outcomes" folder exists, and create it if it doesn't
if not os.path.exists(root_model_folder):
    os.makedirs(root_model_folder)
#----------------------------------------------------------------------#
# Store the images and masks
input_imgs = []
input_masks = []

input_img_folder = os.path.join(root_data_folder, 'ms_rois/testing')
input_mask_folder = os.path.join(root_data_folder, 'mask_rois/testing')


# Retrieve all image and mask files
img_files = [file for file in os.listdir(input_img_folder) if file.endswith(".tif")]
mask_files = [file for file in os.listdir(input_mask_folder) if file.endswith(".tif")]

# Sort the files to ensure consistent ordering
img_files.sort()
mask_files.sort()

# Loop over the files and extract all the images and masks
for i in range(len(img_files)):
    img_file = os.path.join(input_img_folder, img_files[i])
    mask_file = os.path.join(input_mask_folder, mask_files[i])
    ds_img = gdal.Open(img_file)
    ds_mask = gdal.Open(mask_file)
    input_img = np.array([ds_img.GetRasterBand(j + 1).ReadAsArray() for j in range(4)])
    input_img = np.transpose(input_img, (1, 2, 0))
    input_img = exposure.equalize_hist(input_img)

    veg_indices = calculate_veg_indices(input_img)
    input_img = np.concatenate((input_img, veg_indices), axis=2)

    # for c in range(input_img.shape[2]):
    #     input_img[:, :, c] = convolve(input_img[:, :, c], kernel)

    #input_img = apply_gaussian_blur(input_img)

    print(f"Processed image: {img_files[i]} --> Processed mask: {mask_files[i]}")
    
    input_mask = ds_mask.GetRasterBand(1).ReadAsArray().astype(int)
    input_imgs.append(input_img)
    input_masks.append(input_mask)

# Print the number of input images and masks
print(f"Number of input images: {len(input_imgs)}")
print(f"Number of input masks: {len(input_masks)}")
#----------------------------------------------------------------------#
# Preprocess the data for model training
X = []
y = []
for i in range(len(input_imgs)):
    # Filtering unlabelled data
    gt_mask = ((input_masks[i] > -1))
    # Filter unlabelled data from the source image and store their values in the 'X' features variable
    x_array = input_imgs[i][gt_mask, :]
    # Select only labelled data from the labelled image and store their values in the 'y' labels variable
    y_array = input_masks[i][gt_mask]
    # Covert to array format
    x_array = np.asarray(x_array)
    y_array = np.asarray(y_array)
    X.append(x_array)
    y.append(y_array)

# Concatenate the arrays
X_train = np.concatenate(X)
y_train = np.concatenate(y)

print('"X" matrix size: {sz}'.format(sz=X_train.shape))
print('"y" array  size: {sz}'.format(sz=y_train.shape))
#----------------------------------------------------------------------#
# # Separate and count the number of pixels of each classes
# Background = np.where(y_train == 0)[0]
# Pandanus = np.where(y_train == 1)[0]

# n_samples_Background=len(Background)
# n_samples_Pandanus=len(Pandanus)

# print("Pixel numbers for Background :"f"{n_samples_Background}")
# print("Pixel numbers for Pandanus :"f"{n_samples_Pandanus}")
# print()

# # data balancing
# # Downsample the Non_Vegetation
# Background_downsampled = resample(Background, replace=True, n_samples=len(Pandanus), random_state=42)

# print("Pixel numbers for Background_downsampled:"f"{len(Background_downsampled)}")
# print("Pixel numbers for Pandanus :"f"{len(Pandanus)}")

# # Combine the three classes into a single dataset
# X = np.concatenate((X_train[Background_downsampled], X_train[Pandanus]), axis=0)
# y = np.concatenate((y_train[Background_downsampled], y_train[Pandanus]), axis=0)

# # Shuffle the combined dataset
# np.random.seed(42)
# shuffle_idx = np.random.permutation(len(X))
# X_train = X[shuffle_idx]
# y_train = y[shuffle_idx]
#----------------------------------------------------------------------#
# Load the saved model
model_file_path = os.path.join(root_model_folder, 'best_xgb_model.pkl')
best_xgb_model = joblib.load(model_file_path)
#------------------------------------------------------------------#
models = ['XGB']
classifiers = [best_xgb_model]
labels = ['bg','tsa']

all_cr = ''
all_cm = []

# Loop through each model
for i, model in enumerate(models):
    if model == 'XGB':
        print(f'Calculating metrics for {model} ...')
        
        # Predicting X_test using the current model
        y_pred = classifiers[i].predict(X_train)
        
        # Calculate the confusion matrix for the current model
        cm = confusion_matrix(y_train, y_pred)
        print(f'Confusion matrix of {model}:')
        print(cm)
        
        # Save the confusion matrix to the all_cm list
        all_cm.append(cm)
        
        # Develop classification report only using class id=1,2,3,4
        cr = classification_report(y_train, y_pred, target_names=labels)
        print(f'Classification report of {model}:')
        print(cr)
        
        # Append the classification report to the all_cr string
        all_cr += f'Classification report of {model}:\n{cr}\n\n'

# Save the classification report and confusion matrix in a text file
file_path = os.path.join(root_model_folder, 'xgb_cr&cm_validation.txt')
with open(file_path, 'w') as f:
    # Write the classification report to the file
    f.write('--- Classification Report ---\n\n')
    f.write(all_cr)
    
    # Write the confusion matrix to the file
    f.write('\n\n--- Confusion Matrix ---\n\n')
    for i, model in enumerate(models):
        f.write(f'Confusion matrix of {model}:\n{all_cm[i]}\n\n')

# Save the figure of confusion matrix heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(all_cm[0], annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title(f'Confusion Matrix of {models[0]}')
file_path = os.path.join(root_model_folder, 'xgb_cm_heatmap_validation.png')
plt.savefig(file_path, bbox_inches='tight', dpi=300)
plt.show()

print("Completed")
#-------------------------------------****************************-----------------------------------------------#
#-------------------------------------****************************-----------------------------------------------#
#-------------------------------------****************************-----------------------------------------------#