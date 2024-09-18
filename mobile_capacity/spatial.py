import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import geopandas as gpd
import rasterio
import folium
from scipy.spatial import Voronoi
from shapely.geometry import Polygon, box
from shapely import wkt
from shapely.ops import transform
from rasterstats import zonal_stats
from rasterio.plot import show as rio_show


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


def create_voronoi_cells(cellsite_gdf, area_gdf=None, buffer_factor=0.5):
    """
    Creates Voronoi cells for a given set of cell sites and optionally clips them to a specified area.

    Parameters:
    - cellsite_gdf (GeoDataFrame): GeoDataFrame containing the cell sites with geometries.
    - area_gdf (GeoDataFrame, optional): GeoDataFrame containing the boundary of the area to clip the Voronoi cells to.
      If None, the bounding box of the cell sites will be used.
    - buffer_factor (float, optional): Factor to increase the bounding box size. Default is 0.1 (10% increase).

    Returns:
    - GeoDataFrame: A GeoDataFrame containing the original cell sites with an additional column
    for the Voronoi polygons, optionally clipped to the specified area.
    """
    # Verify CRS
    if area_gdf is not None and cellsite_gdf.crs != area_gdf.crs:
        raise ValueError("The CRS of cellsite_gdf and area_gdf do not match.")

    # Extract point data for Voronoi function
    points = np.array([(point.x, point.y) for point in cellsite_gdf.geometry])
    # Extract cellsite ids to assign to Voronoi polygons
    ids = cellsite_gdf['ict_id'].values

    if area_gdf is None:
        # Use the bounding box of the cell sites if no area is provided
        bounding_box = cellsite_gdf.total_bounds
    else:
        bounding_box = area_gdf.geometry.total_bounds

    # Expand the bounding box
    x_min, y_min, x_max, y_max = bounding_box
    x_range = x_max - x_min
    y_range = y_max - y_min
    expanded_bounding_box = (
        x_min - buffer_factor * x_range,
        y_min - buffer_factor * y_range,
        x_max + buffer_factor * x_range,
        y_max + buffer_factor * y_range
    )

    # Generate Voronoi polygons
    voronoi_polygons_with_ids = generate_voronoi_polygons(points, ids, expanded_bounding_box)

    # Create a new GeoDataFrame from the Voronoi polygons
    voronoi_cellsites = gpd.GeoDataFrame(
        [{'geometry': poly, 'ict_id': id} for poly, id in voronoi_polygons_with_ids],
        crs=cellsite_gdf.crs
    )

    if area_gdf is not None:
        # Clip the Voronoi GeoDataFrame with the area if provided
        voronoi_cellsites = gpd.clip(voronoi_cellsites, area_gdf.geometry.item())
    else:
        # If no area is provided, clip to the expanded bounding box
        bbox = box(*expanded_bounding_box)
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


def calculate_zoom_level(minx, miny, maxx, maxy):
    """
    Estimate an appropriate zoom level for a map based on geographical extent.

    Parameters:
    minx, miny, maxx, maxy (float): Bounding box coordinates.

    Returns:
    int: Estimated zoom level (3-12).

    Note:
    Higher zoom levels (e.g., 12) correspond to smaller areas (street view).
    Lower zoom levels (e.g., 3) correspond to larger areas (world view).
    """
    # Function to estimate zoom level based on the extent of the geographical area
    lat_diff = abs(maxy - miny)
    lon_diff = abs(maxx - minx)

    # Calculate approximate zoom level based on the larger difference (lat or lon)
    max_diff = max(lat_diff, lon_diff)

    # Mapping the difference to an appropriate zoom level
    # Approximate scale: smaller area -> higher zoom level
    if max_diff > 60:
        return 3  # World view
    elif max_diff > 20:
        return 5  # Continent view
    elif max_diff > 10:
        return 6  # Large country view
    elif max_diff > 5:
        return 7  # Region view
    elif max_diff > 2:
        return 8  # Sub-region view
    elif max_diff > 1:
        return 9  # City view
    elif max_diff > 0.5:
        return 10  # Town view
    elif max_diff > 0.25:
        return 11  # Neighborhood view
    else:
        return 12  # Street view


def plot_layers(mobilecapacity, poi_sufcapch_result, buffer_areas, output_file=None):
    """
    Create a folium map visualizing mobile network capacity, POIs, and cell towers.

    This function generates an interactive map showing the distribution of Points of Interest (POIs)
    with sufficient or insufficient capacity, cell tower locations, and buffer areas around POIs.

    Parameters:
    -----------
    mobilecapacity : MobileCapacity
        An instance of the MobileCapacity class containing cell site and POI data.
    poi_sufcapch_result : pandas.DataFrame
        A DataFrame containing the sufficiency of capacity for each POI.
    buffer_areas : geopandas.GeoDataFrame
        A GeoDataFrame containing buffer areas around POIs.
    output_file : str, optional
        Path to save the output map as an HTML file. If None, the map object is returned.

    Returns:
    --------
    folium.Map or None
        If output_file is None, returns a folium Map object. Otherwise, saves the map to the
        specified file and returns None.

    Notes:
    ------
    - The map uses the EPSG:4326 coordinate reference system.
    - POIs are color-coded: green for sufficient capacity, red for insufficient capacity.
    - Cell towers are represented by dark blue markers.
    - Buffer areas are shown in blue with 50% opacity.
    - A legend is included in the bottom right corner of the map.
    """

    legend_html = '''
    <div style="
        position: fixed;
        bottom: 50px;
        right: 50px;
        width: 200px;
        height: auto;
        border: 2px solid grey;
        z-index: 1000;
        font-size: 14px;
        background-color: white;
        padding: 10px;
        border-radius: 5px;
    ">
        <b>Capacity</b><br>
        <i class="fa fa-circle" style="color:green"></i> Sufficient Capacity<br>
        <i class="fa fa-circle" style="color:red"></i> Insufficient Capacity<br>
        <i class="fa fa-circle" style="color:darkblue"></i> Cell Towers
    </div>
    '''

    crs = "EPSG:4326"

    # Prepare data for POIs and Cell Towers
    cell_sites = gpd.GeoDataFrame(mobilecapacity.cellsites.data,
                                  geometry=gpd.points_from_xy(mobilecapacity.cellsites.data.lon, mobilecapacity.cellsites.data.lat),
                                  crs=crs)
    pois = gpd.GeoDataFrame(mobilecapacity.poi.data,
                            geometry=gpd.points_from_xy(mobilecapacity.poi.data.lon, mobilecapacity.poi.data.lat),
                            crs=crs)
    pois = pois.merge(poi_sufcapch_result[['poi_id', 'sufcapch']], on='poi_id', how='left')

    # Convert CRS to EPSG:4326 (Lat/Long) for folium
    for df in [cell_sites, pois]:
        if df is not None and not df.empty:
            df.to_crs(epsg=4326, inplace=True)

    # Ensure buffer_areas has a CRS set
    if buffer_areas is not None and not buffer_areas.empty:
        if buffer_areas.crs is None:
            buffer_areas.set_crs(crs, inplace=True)
        buffer_areas.to_crs(epsg=4326, inplace=True)

    # Concatenate geometries, ensuring CRS is set
    all_geometries = pd.concat([cell_sites.geometry, pois.geometry])

    # Dynamically calculate zoom_start
    minx, miny, maxx, maxy = all_geometries.total_bounds
    zoom_level = calculate_zoom_level(minx, miny, maxx, maxy)

    # Create the folium map with dynamic zoom_start
    center_lat = (miny + maxy) / 2
    center_lon = (minx + maxx) / 2
    m = folium.Map(location=[center_lat, center_lon], zoom_start=zoom_level, tiles="cartodb positron")

    # Add buffer areas first
    # Plot polygons from the geometry column of buffer_areas
    for idx, geom in buffer_areas['geometry'].items():
        folium.GeoJson(geom, style_function=lambda x: {'color': 'blue', 'fillOpacity': 0.5}).add_to(m)

    # Add cell towers as circle markers
    for idx, row in cell_sites.iterrows():
        folium.CircleMarker(
            location=[row.geometry.y, row.geometry.x],
            radius=8,  # Marker size
            color='darkblue',  # Border color
            fill=True,  # Fill the circle
            fill_color='darkblue',  # Fill color
            fill_opacity=0.7,  # Opacity of the fill
            popup=f"Cell Tower {row.ict_id}"
        ).add_to(m)

    # Add POIs with sufficient and insufficient capacity as circle markers
    for idx, row in pois.iterrows():
        popup_text = f"POI {row.poi_id}: {'Sufficient' if row['sufcapch'] else 'Insufficient'} Capacity"
        color = 'green' if row['sufcapch'] else 'red'
        folium.CircleMarker(
            location=[row.geometry.y, row.geometry.x],
            radius=5,  # Adjust this value to make the markers smaller
            color=color,  # Border color of the circle
            fill=True,
            fill_color=color,  # Fill color
            fill_opacity=0.7,  # Transparency of the fill color
            popup=popup_text
        ).add_to(m)

    # Add custom legend
    m.get_root().html.add_child(folium.Element(legend_html))

    # Save or show the map
    if output_file:
        m.save(output_file)
    else:
        return m


def display_population_raster(mobilecapacity, output_file=None):
    """
    Display or save a plot of the population density raster.

    Parameters:
    -----------
    mobilecapacity : MobileCapacity
        An instance of the MobileCapacity class containing population data.
    output_file : str, optional
        Path to save the output plot as an image file. If None, the plot is displayed.

    Returns:
    --------
    matplotlib.figure.Figure
        The figure object containing the plot.

    Notes:
    ------
    - The plot shows population density using a color scale.
    - Axes are labeled with latitude and longitude.
    - A color bar is included to interpret the population density values.
    """
    # Open the raster file
    raster = rasterio.open(mobilecapacity.population_data_handler.dataset_path)

    # Create the plot
    fig, ax = plt.subplots(figsize=(10, 10))

    # Display the raster and capture the returned image
    rio_show(raster, ax=ax, cmap='viridis')

    # Set title and labels
    ax.set_title('Population Density')
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')

    # Save or return the figure without displaying
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close(fig)
    else:
        plt.close(fig)  # Close the figure to prevent automatic display
        return fig
