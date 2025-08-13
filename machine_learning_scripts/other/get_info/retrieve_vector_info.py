# Import general python libraries
import os
import csv
from osgeo import gdal
#----------------------------------------------------------------------#
def get_shp_info(directory):
    shp_files = [file for file in os.listdir(directory) if file.endswith('.shp')]   
    data_list = []
    for shp_file in shp_files:
        filepath = os.path.join(directory, shp_file)
        dataset = ogr.Open(filepath)

        if dataset is not None:
            layer = dataset.GetLayer()
            layer_name = layer.GetLayerDefn().GetName()
            spatial_ref = layer.GetSpatialRef()
            crs = spatial_ref.ExportToWkt()

            id_values = []
            for feature in layer:
                feature_id = feature.GetFID()
                id_values.append(feature_id)

            data_list.append({
                'File Name': shp_file,
                'ID Values': id_values,
                'CRS': crs
            })
            dataset = None  # Close the dataset   
    return data_list
#----------------------------------------------------------------------#
def save_to_csv(data_list, output_filename):
    with open(output_filename, mode='w', newline='') as file:
        fieldnames = ['File Name', 'ID Values', 'CRS']
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for data in data_list:
            writer.writerow(data)
#----------------------------------------------------------------------#
if __name__ == "__main__":
    directory_path = "F:/scc_work/msi_msk_rois/gt_rois"  # Change this to your actual directory path
    output_csv_filename = "F:/scc_work/msi_msk_rois/vector_info.csv"

    shp_data = get_shp_info(directory_path)
    save_to_csv(shp_data, output_csv_filename)
    print("Data saved to CSV successfully!")
#-------------------------xxxxxx---------------------------------------#