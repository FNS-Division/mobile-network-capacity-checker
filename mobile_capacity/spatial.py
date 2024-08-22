import math
import numpy as np
import geopandas as gpd
from shapely.geometry import Polygon, box
from scipy.spatial import Voronoi
from rasterstats import zonal_stats
import pandas as pd
import rasterio


def meters_to_degrees_latitude(meters, latitude):
    """
    Calculates the number of meters in one degree of longitude at the given latitude

    Parameters:
    - meters (int): Distance in meters.
    - latitude (float): Latitude in degrees.

    Returns:
    - dist_in_degrees (float): Distance in degrees.

    Note:
    """
    lat_in_rad = latitude * \
        (math.pi / 180)  # latitude in radians, pi / 180 to convert degrees to radians
    # cosine value at latitude (adjacent side / hypotenuse = radius at lat /
    # Earth radius)
    cos_at_lat = math.cos(lat_in_rad)
    radius_at_lat = cos_at_lat * 6378137  # Radius of the Earth: 6378137 meters
    meters_per_degree = 2 * math.pi * radius_at_lat / \
        360  # meters in one degree at latitude
    # degrees in the distance at the given latitude
    dist_in_degrees = meters / meters_per_degree
    return dist_in_degrees


def generate_voronoi_polygons(points, ids, bounding_box):
    """
    Calculates Voronoi polygon areas

    Parameters:
    - points (geodataframe): Cellsites data containing cell site latitudes and longitudes.
    - ids (int): Minimum radius around cell site location for population calculation, meters.
    - bounding_box (int): Maximum radius around cell site location for population calculation, meters.

    Returns:
    - polygons (geodataframe): Cellsites data containing buffer areas around cell site locations.

    Note:
    """
    # Extract bounds from bounding box
    min_x, min_y, max_x, max_y = bounding_box
    extra_points = [
        (min_x - 1, min_y - 1),
        (min_x - 1, max_y + 1),
        (max_x + 1, min_y - 1),
        (max_x + 1, max_y + 1)
    ]
    extended_points = np.vstack([points, extra_points])

    # Compute Voronoi tessellation
    vor = Voronoi(
        extended_points,
        furthest_site=False,
        incremental=False,
        qhull_options=None)

    # Create Voronoi polygons
    polygons = []
    for point_idx, region_index in enumerate(vor.point_region[:len(points)]):
        vertices = vor.regions[region_index]
        if -1 not in vertices:  # -1 indicates vertex outside the Voronoi diagram
            polygon = Polygon([vor.vertices[i] for i in vertices])
            polygons.append((polygon, ids[point_idx]))

    return polygons


def create_voronoi_cells(cellsite_gdf, area_gdf=None):
    """
    Creates Voronoi cells for a given set of cell sites and optionally clips them to a specified area.

    Parameters:
    - cellsite_gdf (GeoDataFrame): GeoDataFrame containing the cell sites with geometries.
    - area_gdf (GeoDataFrame, optional): GeoDataFrame containing the boundary of the area to clip the Voronoi cells to.
      If None, the bounding box of the cell sites will be used.

    Returns:
    - GeoDataFrame: A GeoDataFrame containing the original cell sites with an additional column
    for the Voronoi polygons, optionally clipped to the specified area.
    """
    # Extract point data for Voronoi function
    points = np.array([(point.x, point.y) for point in cellsite_gdf.geometry])
    # Extract cellsite ids to assign to Voronoi polygons
    ids = cellsite_gdf['ict_id'].values

    if area_gdf is None:
        # Use the bounding box of the cell sites if no area is provided
        bounding_box = cellsite_gdf.total_bounds
    else:
        bounding_box = area_gdf.geometry.total_bounds

    # Generate Voronoi polygons
    voronoi_polygons_with_ids = generate_voronoi_polygons(points, ids, bounding_box)

    # Create a new GeoDataFrame from the Voronoi polygons
    voronoi_cellsites = gpd.GeoDataFrame(
        [{'geometry': poly, 'ict_id': id} for poly, id in voronoi_polygons_with_ids],
        crs=cellsite_gdf.crs
    )

    if area_gdf is not None:
        # Clip the Voronoi GeoDataFrame with the area if provided
        voronoi_cellsites = gpd.clip(voronoi_cellsites, area_gdf.geometry.item())
    else:
        # If no area is provided, clip to the bounding box of the cell sites
        bbox = box(*bounding_box)
        voronoi_cellsites = gpd.clip(voronoi_cellsites, bbox)

    # Merge the Voronoi polygons with the original cellsites
    buffer_cellsites = cellsite_gdf.merge(
        voronoi_cellsites,
        on='ict_id',
        how="left",
        suffixes=('', '_voronoi')
    )
    buffer_cellsites = buffer_cellsites.rename(columns={'geometry_voronoi': 'voronoi_polygons'})

    return buffer_cellsites


def get_population_sum_raster(geometry, rasterpath):
    """
    Calculates the population sum within a given geometry using a population density raster.

    Parameters:
    - geometry (shapely.geometry.Polygon): The geometry within which to calculate the population sum.

    Returns:
    - float: The population sum within the given geometry, or None if an error occurs.
    """
    try:
        stats = zonal_stats(
            geometry,
            rasterpath,
            stats="sum")
        return stats[0]['sum']
    except ValueError as ve:
        print(f"ValueError occurred (get_population_sum): {ve}")
        return None
    except Exception as e:
        print(f"An error occurred (get_population_sum): {e}")
        return None


def vectorized_population_sum(buffer_cellsites, population_data, radius):
    """
    Calculates the population sum within clipped ring geometries for all cell sites at once using GeoPandas.

    Parameters:
    - buffer_cellsites (GeoDataFrame): A GeoDataFrame containing the clipped ring geometries.
    - population_data (GeoDataFrame): A GeoDataFrame containing point geometries and a 'population' column.
    - radius (int): The radius for which to calculate the population sum.

    Returns:
    - Series: The population sum within each clipped ring geometry.
    """
    # Ensure CRS match
    if buffer_cellsites.crs != population_data.crs:
        population_data = population_data.to_crs(buffer_cellsites.crs)

    # Clipped ring column name
    clring_column = f'clring_{radius}'

    # Create a temporary GeoDataFrame with only the clipped ring geometries
    temp_gdf = gpd.GeoDataFrame(geometry=buffer_cellsites[clring_column], crs=buffer_cellsites.crs)

    # Perform spatial join
    joined = gpd.sjoin(population_data, temp_gdf, how='inner', predicate='within')

    # Group by the index of buffer_cellsites and sum the population
    population_sums = joined.groupby(joined.index_right)['population'].sum()

    # Reindex to ensure we have a value for all original cell sites, filling missing values with 0
    return population_sums.reindex(buffer_cellsites.index, fill_value=0)


def process_tif(input_file, drop_nodata=True):
    # Read the input raster data
    with rasterio.open(input_file) as src:
        # Get the pixel values as a 2D array
        band = src.read(1)

        transform = src.transform
        pixel_size_x = transform.a
        pixel_size_y = transform.e

        # Get the coordinates for each pixel
        x_coords, y_coords = np.meshgrid(
            np.linspace(src.bounds.left + pixel_size_x / 2, src.bounds.right - pixel_size_x / 2, src.width),
            np.linspace(src.bounds.top + pixel_size_y / 2, src.bounds.bottom - pixel_size_y / 2, src.height)
        )

        # Extract the pixel values, longitude, and latitude arrays from the tif file
        if drop_nodata:
            nodata_value = src.nodata
            nodata_mask = band != nodata_value
            pixel_values = np.extract(nodata_mask, band)
            lons = np.extract(nodata_mask, x_coords)
            lats = np.extract(nodata_mask, y_coords)
        else:
            pixel_values = band.flatten()
            lons = x_coords.flatten()
            lats = y_coords.flatten()

        # Flatten the arrays and combine them into a DataFrame
        data = pd.DataFrame({
            'lon': lons,
            'lat': lats,
            'pixel_value': pixel_values
        })

    return data


def get_tif_xsize(file_path):
    """
    Get the pixel size (resolution) of a GeoTIFF file using rasterio.

    Args:
    file_path (str): Path to the GeoTIFF file.

    Returns:
    tuple: (xsize, ysize) where xsize is the pixel width and ysize is the pixel height.

    Raises:
    ValueError: If unable to open the file or extract the transform.
    """
    try:
        with rasterio.open(file_path) as src:
            # Get the affine transform
            transform = src.transform
            # Extract the pixel sizes
            xsize = abs(transform.a)  # Pixel width
        return xsize
    except rasterio.errors.RasterioIOError:
        raise ValueError('Unable to open the tif file!')
    except Exception as e:
        raise ValueError(f'Error processing the tif file: {str(e)}')
