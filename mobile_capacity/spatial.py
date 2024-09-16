import math
import numpy as np
import geopandas as gpd
from shapely.geometry import Polygon
from scipy.spatial import Voronoi
from rasterstats import zonal_stats
import pandas as pd
import matplotlib.pyplot as plt
import rasterio  
from rasterio.plot import show as rio_show 
import folium

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
    cell_sites = gpd.GeoDataFrame(mobilecapacity.cellsites,
                                  geometry=gpd.points_from_xy(mobilecapacity.cellsites.lon, mobilecapacity.cellsites.lat),
                                  crs=crs)
    pois = gpd.GeoDataFrame(mobilecapacity.poi, geometry=gpd.points_from_xy(mobilecapacity.poi.lon, mobilecapacity.poi.lat),
                            crs=crs)
    pois = pois.merge(poi_sufcapch_result[['sufcapch']], left_on='poi_id', right_index=True)

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
    if buffer_areas is not None:
        for radius in range(mobilecapacity.min_radius, mobilecapacity.max_radius + 1, mobilecapacity.radius_step):
            if f'clring_{radius}' in buffer_areas.columns:
                # Ensure CRS is set for each buffer geometry
                buffer_geom = buffer_areas[f'clring_{radius}']
                if buffer_geom.crs is None:
                    buffer_geom = buffer_geom.set_crs(crs)
                all_geometries = pd.concat([all_geometries, buffer_geom])

    minx, miny, maxx, maxy = all_geometries.total_bounds

    # Dynamically calculate zoom_start
    zoom_level = calculate_zoom_level(minx, miny, maxx, maxy)

    # Create the folium map with dynamic zoom_start
    center_lat = (miny + maxy) / 2
    center_lon = (minx + maxx) / 2
    m = folium.Map(location=[center_lat, center_lon], zoom_start=zoom_level, tiles="cartodb positron")

    # Add buffer areas first
    for radius in range(mobilecapacity.min_radius, mobilecapacity.max_radius + 1, mobilecapacity.radius_step):
        if f'clring_{radius}' in buffer_areas.columns:
            for idx, geom in buffer_areas[f'clring_{radius}'].items():
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