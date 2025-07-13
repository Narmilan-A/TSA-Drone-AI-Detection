import os

root_folder = r'E:/scc_final_works/pandanus_classication/msi_mask_rois'

# Function to count .tif files in a folder
def count_tif_files(folder_path):
    tif_count = 0
    for filename in os.listdir(folder_path):
        if filename.endswith('.tif'):
            tif_count += 1
    return tif_count

# Initialize total count
total_tif_count = 0

# Iterate through subfolders and count .tif files
for subfolder in os.listdir(root_folder):
    subfolder_path = os.path.join(root_folder, subfolder)
    if os.path.isdir(subfolder_path):
        msi_rois_folder = os.path.join(subfolder_path, 'msi_rois')
        if os.path.exists(msi_rois_folder) and os.path.isdir(msi_rois_folder):
            tif_count = count_tif_files(msi_rois_folder)
            print(f"{subfolder}' : {tif_count} .tif files")
            total_tif_count += tif_count

print(f"Total number of .tif files in all subfolders: {total_tif_count}")