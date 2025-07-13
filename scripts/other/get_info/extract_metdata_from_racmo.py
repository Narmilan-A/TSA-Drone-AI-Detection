# from osgeo import gdal
# import csv

# # Path to your GeoTIFF file
# tif_file = "C:/Users/narmilan/Downloads/snowmelt_reprojected_cropped_0.tif"

# # Path where you want to save the CSV
# csv_output = r"C:/Users/narmilan/Downloads/meta_time.csv"

# # Open the raster file using GDAL
# dataset = gdal.Open(tif_file)

# # Open a CSV file to write the metadata
# with open(csv_output, mode='w', newline='') as csv_file:
#     fieldnames = ['band', 'netcdf_dim_time']
    
#     writer = csv.DictWriter(csv_file, fieldnames=fieldnames)

#     # Write the header row
#     writer.writeheader()

#     # Iterate over each band in the raster dataset
#     for i in range(1, dataset.RasterCount + 1):
#         band = dataset.GetRasterBand(i)
        
#         # Get metadata from the band
#         metadata = band.GetMetadata()

#         # Retrieve the 'NETCDF_DIM_time' from the metadata
#         netcdf_dim_time = metadata.get('NETCDF_DIM_time', 'N/A')
        
#         # Prepare the band metadata
#         band_metadata = {
#             'band': i,
# from osgeo import gdal
# import csv
# from datetime import datetime, timedelta

# # Path to your GeoTIFF file
# tif_file = "C:/Users/narmilan/Downloads/snowmelt_reprojected_cropped_0.tif"

# # Path where you want to save the CSV
# csv_output = r"C:/Users/narmilan/Downloads/meta_time_with_date.csv"

# # Base date (1950-01-01)
# base_date = datetime(1950, 1, 1)

# # Open the raster file using GDAL
# dataset = gdal.Open(tif_file)

# # Open a CSV file to write the metadata
# with open(csv_output, mode='w', newline='') as csv_file:
#     fieldnames = ['band', 'netcdf_dim_time', 'date']
    
#     writer = csv.DictWriter(csv_file, fieldnames=fieldnames)

#     # Write the header row
#     writer.writeheader()

#     # Iterate over each band in the raster dataset
#     for i in range(1, dataset.RasterCount + 1):
#         band = dataset.GetRasterBand(i)
        
#         # Get metadata from the band
#         metadata = band.GetMetadata()

#         # Retrieve the 'NETCDF_DIM_time' from the metadata
#         netcdf_dim_time = metadata.get('NETCDF_DIM_time', 'N/A')
        
#         # Check if 'NETCDF_DIM_time' is a valid number (can handle decimals)
#         try:
#             if '.' in netcdf_dim_time:
#                 # Split the decimal and integer part
#                 days, decimal = map(float, netcdf_dim_time.split('.'))
#                 days = int(days)  # Integer part is the number of days
#                 # Calculate the actual date by adding the days to the base date
#                 actual_date = base_date + timedelta(days=days)
#                 formatted_date = actual_date.strftime('%Y-%m-%d')  # Only date (no time)
#             else:
#                 # For whole numbers, just convert days to date
#                 days_since_base = int(netcdf_dim_time)
#                 actual_date = base_date + timedelta(days=days_since_base)
#                 formatted_date = actual_date.strftime('%Y-%m-%d')  # Only date (no time)
#         except Exception as e:
#             formatted_date = 'N/A'
#             print(f"Error processing band {i}: {e}")

#         # Prepare the band metadata
#         band_metadata = {
#             'band': i,
#             'netcdf_dim_time': netcdf_dim_time,
#             'date': formatted_date
#         }
        
#         # Write the band metadata to the CSV
#         writer.writerow(band_metadata)

# print(f"NETCDF_DIM_time and actual date metadata exported successfully to {csv_output}.")


from osgeo import gdal
import csv
from datetime import datetime, timedelta

# Path to your GeoTIFF file
tif_file = "C:/Users/narmilan/Downloads/snowmelt_reprojected_cropped_0.tif"

# Path where you want to save the CSV
csv_output = r"C:/Users/narmilan/Downloads/meta_time_with_statistics.csv"

# Base date (1950-01-01)
base_date = datetime(1950, 1, 1)

# Open the raster file using GDAL
dataset = gdal.Open(tif_file)

# Open a CSV file to write the metadata
with open(csv_output, mode='w', newline='') as csv_file:
    fieldnames = ['band', 'netcdf_dim_time', 'date', 'stat_max', 'stat_mean', 'stat_min']
    
    writer = csv.DictWriter(csv_file, fieldnames=fieldnames)

    # Write the header row
    writer.writeheader()

    # Iterate over each band in the raster dataset
    for i in range(1, dataset.RasterCount + 1):
        band = dataset.GetRasterBand(i)
        
        # Get metadata from the band
        metadata = band.GetMetadata()

        # Retrieve the 'NETCDF_DIM_time' from the metadata
        netcdf_dim_time = metadata.get('NETCDF_DIM_time', 'N/A')

        # Retrieve the statistical values from the metadata
        stat_max = metadata.get('STATISTICS_MAXIMUM', 'N/A')
        stat_mean = metadata.get('STATISTICS_MEAN', 'N/A')
        stat_min = metadata.get('STATISTICS_MINIMUM', 'N/A')
        
        # Check if 'NETCDF_DIM_time' is a valid number (can handle decimals)
        try:
            if '.' in netcdf_dim_time:
                # Split the decimal and integer part
                days, decimal = map(float, netcdf_dim_time.split('.'))
                days = int(days)  # Integer part is the number of days
                # Calculate the actual date by adding the days to the base date
                actual_date = base_date + timedelta(days=days)
                formatted_date = actual_date.strftime('%Y-%m-%d')  # Only date (no time)
            else:
                # For whole numbers, just convert days to date
                days_since_base = int(netcdf_dim_time)
                actual_date = base_date + timedelta(days=days_since_base)
                formatted_date = actual_date.strftime('%Y-%m-%d')  # Only date (no time)
        except Exception as e:
            formatted_date = 'N/A'
            print(f"Error processing band {i}: {e}")

        # Prepare the band metadata
        band_metadata = {
            'band': i,
            'netcdf_dim_time': netcdf_dim_time,
            'date': formatted_date,
            'stat_max': stat_max,
            'stat_mean': stat_mean,
            'stat_min': stat_min
        }
        
        # Write the band metadata to the CSV
        writer.writerow(band_metadata)

print(f"NETCDF_DIM_time, statistics, and actual date metadata exported successfully to {csv_output}.")

