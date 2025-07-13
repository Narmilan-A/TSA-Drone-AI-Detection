# Import general python libraries
import os
import csv
from osgeo import gdal
#----------------------------------------------------------------------#
def get_tif_info(directory):
    tif_files = [file for file in os.listdir(directory) if file.endswith('.tif')] 
    data_list = []
    for tif_file in tif_files:
        filepath = os.path.join(directory, tif_file)
        dataset = gdal.Open(filepath, gdal.GA_ReadOnly)

        if dataset is not None:
            width = dataset.RasterXSize
            height = dataset.RasterYSize
            band_count = dataset.RasterCount

            geotransform = dataset.GetGeoTransform()
            cell_size_x = geotransform[1]
            cell_size_y = geotransform[5]

            projection = dataset.GetProjection()
            spatial_ref = gdal.osr.SpatialReference()
            spatial_ref.ImportFromWkt(projection)
            crs = spatial_ref.ExportToPrettyWkt()
            data_list.append({
                'File Name': tif_file,
                'Width': width,
                'Height': height,
                'Band Count': band_count,
                'Cell Size X': cell_size_x,
                'Cell Size Y': cell_size_y,
                'CRS': crs
            })
            dataset = None  # Close the dataset       
    return data_list
#----------------------------------------------------------------------#
def save_to_csv(data_list, output_filename):
    with open(output_filename, mode='w', newline='') as file:
        fieldnames = ['File Name', 'Width', 'Height', 'Band Count', 'Cell Size X', 'Cell Size Y', 'CRS']
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for data in data_list:
            writer.writerow(data)
#----------------------------------------------------------------------#
if __name__ == "__main__":
    directory_path = "F:/scc_final_submission/tile_orthomosaic/bokarina1"  # Change this to your actual directory path
    output_csv_filename = "F:/scc_final_submission/tile_orthomosaic/bokarina1/rgb_orthomosaic_info.csv"
    tif_data = get_tif_info(directory_path)
    save_to_csv(tif_data, output_csv_filename)
    print("Data saved to CSV successfully!")
#-------------------------xxxxxx---------------------------------------#