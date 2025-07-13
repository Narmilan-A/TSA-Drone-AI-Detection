from osgeo import gdal, osr

# Load the orthomosaic GeoTIFF
dataset = gdal.Open('D:/SAEF_Project/processed/robbos/rgb/rgb_orthomosaic_utm49s_modified.tif')

# Get the spatial reference system of the orthomosaic (CRS)
utm_srs = osr.SpatialReference()
utm_srs.ImportFromEPSG(32749)  # EPSG:32749 corresponds to WGS 84 / UTM zone 49S

# Get the georeferencing information
geotransform = dataset.GetGeoTransform()

# Get the coordinates for the four corners of the image
# (left, top), (right, top), (left, bottom), (right, bottom)
top_left = (geotransform[0], geotransform[3])
top_right = (geotransform[0] + geotransform[1] * dataset.RasterXSize, geotransform[3])
bottom_left = (geotransform[0], geotransform[3] + geotransform[5] * dataset.RasterYSize)
bottom_right = (geotransform[0] + geotransform[1] * dataset.RasterXSize, geotransform[3] + geotransform[5] * dataset.RasterYSize)

# Create the coordinate transformation to convert UTM to WGS 84 (Lat/Lon)
wgs84_srs = osr.SpatialReference()
wgs84_srs.ImportFromEPSG(4326)  # WGS 84 (Latitude/Longitude)

# Transform the UTM coordinates to WGS 84
transform = osr.CoordinateTransformation(utm_srs, wgs84_srs)

# Function to convert UTM to Lat/Lon
def utm_to_latlon(x, y):
    lon, lat, _ = transform.TransformPoint(x, y)
    return lat, lon

# Convert each corner to lat/lon
north_lat, north_lon = utm_to_latlon(top_left[0], top_left[1])
south_lat, south_lon = utm_to_latlon(bottom_left[0], bottom_left[1])
east_lat, east_lon = utm_to_latlon(top_right[0], top_right[1])
west_lat, west_lon = utm_to_latlon(bottom_right[0], bottom_right[1])

# Print out the results in decimal degrees
print(f"North: {north_lat}, {north_lon}")
print(f"South: {south_lat}, {south_lon}")
print(f"East: {east_lat}, {east_lon}")
print(f"West: {west_lat}, {west_lon}")
