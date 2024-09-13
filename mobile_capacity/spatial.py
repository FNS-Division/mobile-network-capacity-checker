import math
import os
from functools import partial
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import geopandas as gpd
from shapely.geometry import Polygon, box
from shapely import wkt
from shapely.ops import transform
from scipy.spatial import Voronoi
from rasterstats import zonal_stats
import rasterio
from rasterio.plot import show as rio_show
import contextily as cx
from contextily import providers


def haversine_(lats, lons, R=6371e3, upper_tri=False):
    """
    Calculate the great circle distance between two points
    on the earth (specified in decimal degrees) using the
    Haversine formula.

    Parameters
    ----------
    lats, lons: array-like
        Arrays of latitudes and longitudes of the two points.
        Each array should have shape (2,) where the first element
        is the latitude and the second element is the longitude.
    upper_tri : bool, optional
        If True, returns the distance matrix in upper triangular form.
        Default is False.
    R : float, optional
        Radius of the earth in meters. Default is 6371000.0 m.

    Returns
    -------
    ndarray
        The distance matrix between the points in meters.
        If `upper_tri` is True, returns the upper triangular form of the matrix.

    """

    if not len(lats) == len(lons):
        raise ValueError("The length of 'lats' and 'lons' must be equal.")

    # Convert latitudes and longitudes to radians
    lat_rads = np.radians(lats)
    lon_rads = np.radians(lons)

    # Compute pairwise haversine distances using broadcasting
    dlat = lat_rads[:, np.newaxis] - lat_rads[np.newaxis, :]
    dlon = lon_rads[:, np.newaxis] - lon_rads[np.newaxis, :]
    a = np.sin(dlat / 2) ** 2 + np.cos(lat_rads[:, np.newaxis]) * \
        np.cos(lat_rads[np.newaxis, :]) * np.sin(dlon / 2) ** 2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    distances = R * c

    if len(lats) == 2:
        distances = distances[0, 1]
    elif upper_tri:
        i_upper = np.triu_indices(distances.shape[0], k=1)
        distances = distances[i_upper]

    return distances


def calculate_haversine_for_pair(lat1, lon1, lat2, lon2, R=6371e3):
    """
    Calculate the haversine distance between two pairs of latitude and longitude.

    Parameters:
    - lat1 (float): Latitude of the first point.
    - lon1 (float): Longitude of the first point.
    - lat2 (float): Latitude of the second point.
    - lon2 (float): Longitude of the second point.
    - R (float): Radius of the Earth.

    Returns:
    float: Haversine distance between the two points.
    """

    lat1, lon1, lat2, lon2 = np.radians([lat1, lon1, lat2, lon2])

    # Haversine formula
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    c = 2 * np.arcsin(np.sqrt(a))
    distance = R * c

    return distance


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


def spherical_to_cartesian(lat, lon, alt, R=6371e3):
    """
    Convert geographical coordinates (latitude, longitude, altitude) to Cartesian coordinates (x, y, z).

    Args:
        lat (float): Latitude in decimal degrees.
        lon (float): Longitude in decimal degrees.
        alt (float): Altitude in meters above the Earth's surface.
        R (float): Earth's radius in meters

    Returns:
        tuple: Cartesian coordinates (x, y, z) in meters.
    """

    r = R + alt
    lat_rad = np.radians(lat)
    lon_rad = np.radians(lon)

    x = r * np.cos(lat_rad) * np.cos(lon_rad)
    y = r * np.cos(lat_rad) * np.sin(lon_rad)
    z = r * np.sin(lat_rad)

    return x, y, z


def line_of_sight_distance_with_altitude(lat1, lon1, alt1, lat2, lon2, alt2, R=6371e3):
    """
    Calculate the line of sight distance between two points with latitude, longitude, and altitude information.

    Args:
        lat1 (float): Latitude of point 1 in decimal degrees.
        lon1 (float): Longitude of point 1 in decimal degrees.
        alt1 (float): Altitude of point 1 in meters above the Earth's surface.
        lat2 (float): Latitude of point 2 in decimal degrees.
        lon2 (float): Longitude of point 2 in decimal degrees.
        alt2 (float): Altitude of point 2 in meters above the Earth's surface.
        R (float): Earth's radius in meters

    Returns:
        float: The line of sight distance between two points in meters.

    """

    x1, y1, z1 = spherical_to_cartesian(lat1, lon1, alt1, R)
    x2, y2, z2 = spherical_to_cartesian(lat2, lon2, alt2, R)

    distance = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2 + (z2 - z1) ** 2)
    return distance


def to_geodataframe(data, lat_col=None, lon_col=None, crs='EPSG:4326', inplace=False):
    """
    Converts a DataFrame containing latitude and longitude columns into a GeoDataFrame.

    Parameters:
    - data : pandas.DataFrame
        The DataFrame containing the latitude and longitude columns.
    - lat_col : str, optional
        The name of the column containing latitude values. If not provided, the method will try to find it.
    - lon_col : str, optional
        The name of the column containing longitude values. If not provided, the method will try to find it.
    - rename_georeference_columns : bool, optional
        Whether to rename the georeference columns if they have different names. Default is True.
    - crs : str, optional
        The coordinate reference system (CRS) of the GeoDataFrame. Default is 'EPSG:4326'.
    - inplace : bool, optional
        Whether to perform the operation in-place. Default is False.

    Returns:
    - geopandas.GeoDataFrame
        A GeoDataFrame with the same columns as the input DataFrame and a new geometry column containing
        the Point objects created from the latitude and longitude columns.
    """

    if not isinstance(data, pd.DataFrame):
        raise TypeError("The 'data' parameter must be a pandas DataFrame.")

    if not inplace:
        data = data.copy()

    if 'geometry' not in data:
        # Extract the latitude and longitude values from the input data
        latitudes = data[lat_col]
        longitudes = data[lon_col]

        # Create a new GeoDataFrame with the input data and geometry column
        geometry = gpd.points_from_xy(longitudes, latitudes)
        gdf = gpd.GeoDataFrame(data, geometry=geometry, crs=crs)
    else:
        # Convert 'geometry' column from WKT format to GeoSeries
        data['geometry'] = data['geometry'].apply(wkt.loads)
        gdf = gpd.GeoDataFrame(data, crs=crs)

    return gdf if not inplace else None


def buffer_gdf_in_meters(gdf, buffer_distance_meters, cap_style=1, inplace=False):
    """
    Buffers a GeoDataFrame with a given buffer distance in meters.

    Parameters:
    - gdf : geopandas.GeoDataFrame
        The GeoDataFrame to be buffered.
    - buffer_distance_meters : float
        The buffer distance in meters.
    - cap_style : int, optional
        The style of caps. 1 (round), 2 (flat), 3 (square). Default is 1.
    - inplace : bool, optional
        Whether to perform the operation in-place. Default is False.

    Returns:
    - geopandas.GeoDataFrame
        The buffered GeoDataFrame.
    """

    if not inplace:
        gdf = gdf.copy()

    input_crs = gdf.crs

    # create a custom UTM CRS based on the calculated UTM zone
    utm_crs = gdf.estimate_utm_crs()

    # transform your GeoDataFrame to the custom UTM CRS:
    gdf_projected = gdf.to_crs(utm_crs)

    # create the buffer in meters:
    gdf["geometry"] = gdf_projected['geometry'].buffer(buffer_distance_meters, cap_style=cap_style)

    # transform the buffer geometry back to input crs
    gdf['geometry'] = gdf.geometry.to_crs(input_crs)

    return gdf if not inplace else None


def calculate_distance_to_horizon(observer_height, R=6371.0, k=0):
    """
    Calculate the maximum distance to the horizon for an observer.

    Args:
        observer_height (float): The height of the observer in meters measured from the surface of the globe.
        R (float, optional): The radius of the Earth in kilometers. Default value is 6371.0 km.
        k (float, optional): The refraction coefficient; standard at sea level k = 0.170

    Returns:
        float: Distance to the horizon in meters.
    """

    # calculate refracted radius of the earth
    R_ = R / (1 - k)

    # convert earth radius to meters
    R_ = R_ * 1000

    # Calculate the maximum distance to the horizon using the Pythagorean theorem
    # d^2 = 2*R*h where d is the distance to the horizon, R is the radius of the Earth, and h is the height of the observer.
    distance_to_horizon = np.sqrt((2 * R_ * observer_height) + (observer_height ** 2))

    return distance_to_horizon


def sum_of_horizon_distances(first_observer_height, second_observer_height):
    """
    Calculate the sum of the distances to the horizons of two observers.

    Args:
        first_observer_height (float): The height of the first observer in meters.
        second_observer_height (float): The height of the second observer in meters.

    Returns:
        float: The sum of the distances to the horizons in meters.
    """
    distance_to_horizon_1 = calculate_distance_to_horizon(first_observer_height)
    distance_to_horizon_2 = calculate_distance_to_horizon(second_observer_height)
    total_horizon_distance = distance_to_horizon_1 + distance_to_horizon_2

    return total_horizon_distance


def calculate_curvature_drop(distance_from_observer, R=6371.0, k=0):
    """
    Calculate the curvature drop for a given distance from the observer.

    Args:
        distance_from_observer (float): The distance from the observer to the object in meters.
        R (float, optional): The radius of the Earth in kilometers. Default value is 6371.0 km.
        k (float, optional): The refraction coefficient; standard at sea level k = 0.170

    Returns:
        float: The curvature drop in meters.
    """

    # calculate refracted radius of the earth
    R_ = R / (1 - k)

    # convert earth radius to meters
    R_ = R_ * 1000

    curvature_drop = distance_from_observer ** 2 / (2 * R_)

    return curvature_drop


def calculate_hidden_height(observer_height, distance_from_observer, R=6371.0, k=0):
    """
    Calculate the hidden height of an object below the observer's line of sight.

    Args:
        observer_height (float): The height of the observer in meters measured from the surface of the globe.
        distance_from_observer (float): The distance from the observer to the object in meters.
        R (float, optional): The radius of the Earth in kilometers. Default value is 6371.0 km.
        k (float, optional): The refraction coefficient; standard at sea level k = 0.170

    Returns:
        float: The hidden height of the object in meters.
    """

    # calculate observer horizon in meters
    distance_to_horizon = calculate_distance_to_horizon(observer_height, R, k)

    if distance_from_observer <= distance_to_horizon:
        hidden_height = 0
    else:
        hidden_height = calculate_curvature_drop(distance_from_observer - distance_to_horizon, R, k)

    return hidden_height


def adjust_elevation(observer_height, distance_from_observer, R=6371.0, k=0):
    """
    Adjust the elevation based on the curvature of the Earth.

    Args:
        observer_height (float): The height of the observer in meters measured from the surface of the globe.
        distance_from_observer (float): The distance from the observer to the target in meters.
        R (float, optional): The radius of the Earth in kilometers. Default value is 6371.0 km.
        k (float, optional): The refraction coefficient; standard at sea level k = 0.170

    Returns:
        float: The curvature correction in meters.
    """

    # calculate observer horizon in meters
    distance_to_horizon = calculate_distance_to_horizon(observer_height, R, k)

    if distance_from_observer <= distance_to_horizon:
        curvature_correction = calculate_curvature_drop(distance_from_observer, R, k)
    else:
        curvature_correction = - calculate_curvature_drop(distance_from_observer - distance_to_horizon, R, k)

    return curvature_correction


def calculate_azimuth(lat1, lon1, lat2, lon2):
    """
    Calculates the azimuth angle between two geographic points between true north and antenna main beam direction measured clockwise.
    Args:
        lat1: latitude of the first antenna in decimal degrees
        lon1: longitude of the first antenna in decimal degrees
        lat2: latitude of the second antenna in decimal degrees
        lon2: longitude of the second antenna in decimal degrees
    Returns:
        The azimuth angle between the two antennas in decimal degrees
    """
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dLon = lon2 - lon1
    y = np.sin(dLon) * np.cos(lat2)
    x = np.cos(lat1) * np.sin(lat2) - np.sin(lat1) * np.cos(lat2) * np.cos(dLon)
    brng = np.arctan2(y, x)

    return np.round((np.degrees(brng) + 360) % 360, 2)


def add_padding_to_bounds(bounds, padding_ratio=0.05):
    """
    Add padding to the bounding box to avoid cropping.

    Parameters:
    - bounds: (minx, miny, maxx, maxy) tuple representing the bounding box
    - padding_ratio: Ratio of the padding to add around the bounds

    Returns:
    - Padded bounding box (minx, miny, maxx, maxy)
    """
    minx, miny, maxx, maxy = bounds
    width = maxx - minx
    height = maxy - miny
    padding_x = width * padding_ratio
    padding_y = height * padding_ratio
    return (minx - padding_x, miny - padding_y, maxx + padding_x, maxy + padding_y)


def plot_layers(mobilecapacity, poi_sufcapch_result, buffer_areas, mobile_coverage_path=None, show_basemap=True, figsize=(10, 10), output_file=None):
    crs = "EPSG:4326"

    # Prepare data for POIs and Cell Towers
    cell_sites = gpd.GeoDataFrame(mobilecapacity.cellsites.data, geometry=gpd.points_from_xy(mobilecapacity.cellsites.data.lon, mobilecapacity.cellsites.data.lat), crs=crs)
    pois = gpd.GeoDataFrame(mobilecapacity.poi.data, geometry=gpd.points_from_xy(mobilecapacity.poi.data.lon, mobilecapacity.poi.data.lat), crs=crs)
    pois = pois.merge(poi_sufcapch_result[['sufcapch']], left_on='poi_id', right_index=True)

    mc = None
    if mobile_coverage_path and os.path.exists(mobile_coverage_path):
        mc = gpd.read_file(mobile_coverage_path, crs=crs)
    elif mobile_coverage_path:
        print(f"Mobile coverage file not found at {mobile_coverage_path}")

    # Convert CRS to EPSG:3857 (Web Mercator)
    for df in [cell_sites, pois, mc]:
        if df is not None and not df.empty:
            df.to_crs(epsg=3857, inplace=True)

    # Ensure buffer_areas has a CRS set and convert CRS for all geometry columns
    if buffer_areas is not None and not buffer_areas.empty:
        if buffer_areas.crs is None:
            buffer_areas.set_crs(crs, inplace=True)
        buffer_areas.to_crs(epsg=3857, inplace=True)

    # Plot setup
    fig, ax = plt.subplots(figsize=figsize)

    # Get bounds from POIs only
    if pois.empty:
        print("POIs dataframe is empty.")
        return

    minx, miny, maxx, maxy = pois.total_bounds
    minx, miny, maxx, maxy = add_padding_to_bounds((minx, miny, maxx, maxy), padding_ratio=0.05)

    # Plot mobile coverage if available
    if mc is not None:
        mc.plot(ax=ax, alpha=0.3, legend=True)

    # Plot cell towers
    cell_sites.plot(ax=ax, color='yellow', markersize=50, marker='o', label='Cell towers', edgecolor='black')

    # Plot POIs with sufficient and insufficient capacity
    sufficient_capacity = pois[pois['sufcapch']]
    insufficient_capacity = pois[~pois['sufcapch']]
    sufficient_capacity.plot(ax=ax, color='green', markersize=30, marker='^', label="POIs with sufficient capacity", edgecolor='black')
    insufficient_capacity.plot(ax=ax, color='red', markersize=30, marker='^', label="POIs with insufficient capacity", edgecolor='black')

    # Plot buffer areas based on different radii
    for radius in range(mobilecapacity.min_radius, mobilecapacity.max_radius + 1, mobilecapacity.radius_step):
        if f'clring_{radius}' in buffer_areas.columns:
            geometry = buffer_areas[f'clring_{radius}'].set_crs(epsg=4326)
            geometry = geometry.to_crs(epsg=3857)
            geometry.plot(ax=ax, color='blue', edgecolor='lightgrey', alpha=0.5)

    # Add basemap if required
    if show_basemap:
        try:
            cx.add_basemap(ax, crs=pois.crs, source=cx.providers.CartoDB.Positron)
        except Exception as e:
            print(f"Failed to add basemap: {e}")
            print("Skipping basemap...")

    # Set plot limits
    ax.set_xlim([minx, maxx])
    ax.set_ylim([miny, maxy])
    ax.axis('off')
    ax.set_title('Cell Sites, POIs, Buffers, and Mobile Coverage', fontsize=16, pad=20)
    ax.legend(loc='upper left', bbox_to_anchor=(0, 1), fontsize=12)
    plt.tight_layout()

    # Save or show the plot
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')  # High DPI for better resolution
        plt.close()
    else:
        plt.show()
