import math
import numpy as np
import geopandas as gpd
from shapely.geometry import Polygon
from scipy.spatial import Voronoi
from rasterstats import zonal_stats

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


def create_voronoi_cells(cellsite_gdf, area_gdf):
    """
    Creates Voronoi cells for a given set of cell sites and clips them to a specified area.

    Parameters:
    - cellsite_gdf (GeoDataFrame): GeoDataFrame containing the cell sites with geometries.
    - area_gdf (GeoDataFrame): GeoDataFrame containing the boundary of the area to clip the Voronoi cells to.

    Returns:
    - GeoDataFrame: A GeoDataFrame containing the original cell sites with an additional column
    for the Voronoi polygons clipped to the specified area.
    """
    # Extract point data for Voronoi function
    points = np.array([(point.x, point.y)
                        for point in cellsite_gdf.geometry])
    # Extract cellsite ids to assign to Voronoi polygons
    ids = cellsite_gdf['ict_id'].values
    bounding_box = area_gdf.geometry.total_bounds
    # Generate Voronoi polygons
    voronoi_polygons_with_ids = generate_voronoi_polygons(
        points, ids, bounding_box)
    # Create a new GeoDataFrame from the Voronoi polygons
    voronoi_cellsites = gpd.GeoDataFrame(
        [{'geometry': poly, 'ict_id': id}
            for poly, id in voronoi_polygons_with_ids],
        crs=cellsite_gdf.crs
    )
    # Clip the Voronoi GeoDataFrame with the area
    clipped_voronoi_cellsites = gpd.clip(
        voronoi_cellsites, area_gdf.geometry.item())
    # Merge the clipped Voronoi polygons with the original cellsites
    buffer_cellsites = cellsite_gdf.merge(
        clipped_voronoi_cellsites,
        on='ict_id',
        how="left",
        suffixes=(
            '',
            '_voronoi'))
    buffer_cellsites = buffer_cellsites.rename(
        columns={'geometry_voronoi': 'voronoi_polygons'})
    return buffer_cellsites


def get_population_sum(geometry, rasterpath):
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