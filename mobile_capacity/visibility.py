import numpy as np
import pandas as pd
from shapely.geometry import LineString
from scipy.spatial import cKDTree
import time
import logging

from mobile_capacity.entities.cellsite import CellSiteCollection
from mobile_capacity.entities.pointofinterest import PointOfInterestCollection
from mobile_capacity.handlers.srtmdatahandler import SRTMDataHandler
from mobile_capacity.utils import *
from mobile_capacity.spatial import *


class Visibility:
    """
    A module for computing visibility between points of interest and cell sites.

    Args:
        points_of_interest (PointOfInterestCollection): Collection of points of interest.
        cell_sites (CellSiteCollection): Collection of cell sites.
        srtm_data_handler (SRTMDataHandler): Handler for SRTM data.
        cellsite_search_radius (int): The search radius around the point of interest (in meters),
          within which cell sites will be considered for the visibility analysis.
        poi_antenna_height (float): The antenna height at the point of interest locations (in meters),
            used for creating the line of sight and performing the analysis.
        num_visible_cellsites (int, optional): The number of visible cell sites to be extracted for each poi during the search. Default is 3.
        allowed_radio_types (list, optional): List of allowed radio types. Default is ['unknown', '2G', '3G', '4G', '5G'].
        earth_radius (float, optional): The radius of the Earth in kilometers. Default is 6371.
        refraction_coef (float, optional): The refraction coefficient. Default is 0.
        logger (Logger, optional): Logger instance for logging. If None, a default logger is created.

    Properties:
        results_table (pd.DataFrame): DataFrame containing the results.
        storage_table (pd.DataFrame): DataFrame containing the storage results.

    Methods:
        retrieve_srtm_data_files(): Retrieve SRTM data files for points of interest and cell sites.
        perform_analysis(): Perform visibility analysis.
        get_results_table(): Get the results table.
        get_storage_table(): Get the storage table.
        format_analysis_summary(): Format the analysis summary.
    """

    def __init__(self,
                 points_of_interest: PointOfInterestCollection,
                 cell_sites: CellSiteCollection,
                 srtm_data_handler: SRTMDataHandler = None,
                 cellsite_search_radius=35000,
                 poi_antenna_height=15,
                 num_visible_cellsites: int = 3,
                 allowed_radio_types: list = ['unknown', '2G', '3G', '4G', '5G'],
                 earth_radius=6371,
                 refraction_coef=0,
                 use_srtm=True,
                 logger=None,
                 enable_logging=False,
                 logs_dir=None,
                 ):

        self.points_of_interest = points_of_interest
        self.cell_sites = cell_sites
        self.srtm_data_handler = srtm_data_handler
        self.cellsite_search_radius = cellsite_search_radius
        self.poi_antenna_height = poi_antenna_height
        self.num_visible_cellsites = num_visible_cellsites
        self.allowed_radio_types = allowed_radio_types
        self.earth_radius = earth_radius
        self.refraction_coef = refraction_coef
        self.use_srtm = use_srtm
        self.logs_dir = logs_dir
        self.enable_logging = enable_logging

        self._results_table = None
        self._storage_table = None

        if len(allowed_radio_types) < 5:
            self.cell_sites = cell_sites.filter_sites_by_radio_type(allowed_radio_types=allowed_radio_types)

        if use_srtm and not srtm_data_handler:
            raise TypeError('srtm_data_handler argument is required unless use_srtm = False')

        # Logger
        if self.enable_logging:
            if logger is None:
                self.logger = initialize_logger(self.logs_dir)
            elif not isinstance(logger, logging.Logger):
                raise TypeError(f'logger must be an instance of {logging.Logger}')
            else:
                self.logger = logger
        else:
            self.logger = None

        self.analysis_param = dict(
            cellsite_search_radius=cellsite_search_radius,
            poi_antenna_height=poi_antenna_height,
            num_visible_cellsites=num_visible_cellsites,
            allowed_radio_types=allowed_radio_types,
        )

        self.analysis_stats = dict(
            num_points_of_interest=len(points_of_interest),
            num_cell_sites=len(cell_sites),
        )

    def _log(self, level, message):
        """Conditionally log messages based on enable_logging flag."""
        if self.enable_logging and self.logger:
            if level == 'info':
                self.logger.info(message)
            elif level == 'warn':
                self.logger.warn(message)
            elif level == 'error':
                self.logger.error(message)
            elif level == 'debug':
                self.logger.debug(message)

    @property
    def results_table(self):
        if self._results_table is None:
            self._results_table = self.get_results_table()
        return self._results_table

    @property
    def storage_table(self):
        if self._storage_table is None:
            self._storage_table = self.get_storage_table()
        return self._storage_table

    def retrieve_srtm_data_files(self):
        """
        Retrieve SRTM data files for points of interest and cell sites.
        """
        gdf_poi = to_geodataframe(self.points_of_interest.data, lat_col="lat", lon_col="lon").rename(columns={'poi_id': 'id'}, inplace=False)
        gdf_cellsite = to_geodataframe(self.cell_sites.data, lat_col="lat", lon_col="lon").rename(columns={'ict_id': 'id'}, inplace=False)

        # concatenate school and cellsite data
        all_loc = pd.concat([gdf_poi, gdf_cellsite]).set_index('id')
        all_loc = buffer_gdf_in_meters(all_loc, buffer_distance_meters=35000, cap_style=1, inplace=False)

        # locate srtm tiles
        self._log('info', 'Locating SRTM data files...')

        unique_data_files, unmatched_indices = self.srtm_data_handler.locate_data_files(all_loc)

        if not len(unmatched_indices) == 0:

            self.unmatched_indices = unmatched_indices

            self.logger.error(
                f'{len(unmatched_indices)} locations could not be matched with SRTM data files. \
                    Please see "unmatched_indices" property of the visibility intance!')

        data_files_to_download = self.srtm_data_handler.datafile_check(list(unique_data_files))

        self._log('info', 'Downloading SRTM data files...')

        self.srtm_data_handler.download_data_files(data_files=list(data_files_to_download))

    def perform_analysis(self):
        """
        Perform visibility analysis.
        """

        if len(self.cell_sites) == 0:
            raise ValueError("No cellsites available with desired radio constraint")

        srtm_time = time.time()
        if self.use_srtm:
            # Retrieve SRTM data files for points of interest and cell sites
            self.retrieve_srtm_data_files()
            self.analysis_stats['srtm_data_download_time'] = round(time.time() - srtm_time, 2)
            srtm_collection = self.srtm_data_handler.srtmheightmapcollection
        else:
            self._log('info', 'SRTM data will not be used')
            self.analysis_stats['srtm_data_download_time'] = 0

        analysis_time = time.time()

        # Create a KDTree for cell sites for efficient spatial queries
        cellsite_tree = cKDTree(
            self.cell_sites.get_lat_lon_pairs()
        )

        # Initialize the total visibility checks counter
        self.analysis_stats['total_visibility_checks'] = 0

        # Initialize a dictionary to store analysis results
        analysis_results = dict()

        # Iterate over each point of interest
        total_pois = len(self.points_of_interest.entities)
        for i, poi in enumerate(self.points_of_interest.entities, 1):
            log_progress_bar(self.logger, i, total_pois, prefix='Visibility analysis progress:', suffix='Complete', length=50)
            self._log("info", f"Performing visibility analysis for point of interest {poi.poi_id}")

            # Initialize variables to track the number of visible cell sites and store analysis results for the current point of interest
            num_visible = 0
            analysis_results.update({poi.poi_id: dict(is_visible=False, visible_cellsites=dict())})

            # Query cell sites within the search radius using KDTree
            _, cellsite_index = cellsite_tree.query([poi.lat, poi.lon], len(self.cell_sites))

            # Initialize a counter for the number of visibility checks
            num_visibility_checks = 0

            # Iterate until the required number of visible cell sites is reached
            while num_visible < self.num_visible_cellsites:
                # Get the cell site from the current index
                if num_visibility_checks < len(cellsite_index):
                    cellsite = self.cell_sites.get_nth_entity(cellsite_index[num_visibility_checks])
                else:
                    break
                # Increment the visibility checks counter
                num_visibility_checks += 1

                # Calculate ground distance between the point of interest and cell site
                ground_distance = round(poi.get_distance(cellsite))

                # The rest of this loop can be ignored if we want to use SRTM data (simple analysis)
                # We simply check if the distance between the cell site and the POI is within the radius
                if not self.use_srtm:
                    if (ground_distance <= self.cellsite_search_radius):
                        num_visible += 1
                        analysis_results[poi.poi_id]['visible_cellsites'][cellsite.ict_id] = dict(
                            lat=cellsite.lat,
                            lon=cellsite.lon,
                            radio_type=cellsite.radio_type,
                            ground_distance=ground_distance,
                            antenna_los_distance=ground_distance,
                            azimuth_angle=calculate_azimuth(poi.lat, poi.lon, cellsite.lat, cellsite.lon),
                            los_geometry=LineString([poi.get_point_geometry(), cellsite.get_point_geometry()])
                        )
                    continue

                # Calculate antenna altitudes for the point of interest and cell site
                poi_antenna_altitude = srtm_collection.get_altitude(poi.lat, poi.lon) + self.poi_antenna_height
                cellsite_antenna_altitude = srtm_collection.get_altitude(
                    cellsite.lat, cellsite.lon) + cellsite.antenna_height

                # Check if the cell site is within the search radius and visible based on horizon distances
                if ground_distance > self.cellsite_search_radius or ground_distance > sum_of_horizon_distances(poi_antenna_altitude, cellsite_antenna_altitude):
                    # Break if the cell site is beyond the search radius or not visible
                    break

                # Get elevation and distance profiles between the point of interest and cell site
                e_profile, d_profile = zip(
                    *[(i.elevation, i.distance) for i in srtm_collection.get_elevation_profile(poi.lat, poi.lon, cellsite.lat, cellsite.lon)])

                # Map extreme elevation values to below sea level
                e_profile = list(map(lambda x: x - 65535 if x > 65000 else x, e_profile))

                # Adjust elevation profile to incorporate Earth curvature
                curvature_adjustment = list(map(lambda x: adjust_elevation(
                    poi_antenna_altitude, x, self.earth_radius, self.refraction_coef), d_profile))
                adjusted_e_profile = np.add(e_profile, curvature_adjustment)

                # Calculate line of sight profile
                los_profile = np.linspace(
                    adjusted_e_profile[0] + self.poi_antenna_height, adjusted_e_profile[-1] + cellsite.antenna_height, len(e_profile))
                # Check if line of sight is clear (all points above terrain)
                has_line_of_sight = np.all(los_profile > adjusted_e_profile)

                # Increment the number of visible cell sites if there is line of sight
                num_visible += has_line_of_sight

                if has_line_of_sight:
                    # Store analysis results for the visible cell site
                    analysis_results[poi.poi_id]['visible_cellsites'][cellsite.ict_id] = dict(
                        lat=cellsite.lat,
                        lon=cellsite.lon,
                        radio_type=cellsite.radio_type,
                        ground_distance=ground_distance,
                        antenna_los_distance=round(line_of_sight_distance_with_altitude(
                            poi.lat, poi.lon, poi_antenna_altitude, cellsite.lat, cellsite.lon, cellsite_antenna_altitude)),
                        azimuth_angle=calculate_azimuth(poi.lat, poi.lon, cellsite.lat, cellsite.lon),
                        los_geometry=LineString([poi.get_point_geometry(), cellsite.get_point_geometry()])
                    )

            # Update the analysis results with the visibility status and total visibility checks
            analysis_results[poi.poi_id].update(is_visible=num_visible > 0)
            self.analysis_stats['total_visibility_checks'] += num_visibility_checks

        self.analysis_stats['analysis_time'] = round(time.time() - analysis_time, 2)
        self.analysis_stats['total_time_elapsed'] = round(time.time() - srtm_time, 2)
        self.analysis_stats['avg_visibility_checks_per_poi'] = round(
            self.analysis_stats['total_visibility_checks'] / len(self.points_of_interest), 2)
        self.analysis_results = analysis_results
        self._log("info", self.format_analysis_summary())

    def get_results_table(self):
        """
        Get the results table.

        Returns:
            pd.DataFrame: DataFrame containing the results.
        """
        df_results = self.get_storage_table()
        df_results['num_visible'] = df_results['visible_cellsites'].apply(lambda x: len(x))
        aux_tables = []

        # Iterate over each row in storage_table
        for _, row in df_results.iterrows():

            # No visible cell towers
            if row["num_visible"] == 0:
                aux_table = pd.DataFrame({
                    'poi_id': [row["poi_id"]],
                    'is_visible': row['is_visible'],
                    'num_visible': [row['num_visible']],
                    'order': [1],
                    'ict_id': [None],
                    'radio_type': [None],
                    'ground_distance': [None],
                    'antenna_los_distance': [None],
                    'azimuth_angle': [None],
                    'vis_geometry': [None]
                })
                aux_tables.append(aux_table)

            # At least one visible cell tower
            if row["num_visible"] > 0:
                aux_table = pd.DataFrame(row["visible_cellsites"]).transpose().reset_index().rename(columns={"index": "ict_id"})
                aux_table["poi_id"] = row["poi_id"]
                aux_table['is_visible'] = row['is_visible']
                aux_table['num_visible'] = row['num_visible']
                aux_table['order'] = aux_table['ground_distance'].rank(method='average', ascending=True).astype(int)
                aux_table = aux_table.rename(columns={'los_geometry': 'vis_geometry'})
                aux_table = aux_table.drop(columns=['lat', 'lon'])
                poi_id = aux_table.pop('poi_id')
                aux_table.insert(0, 'poi_id', poi_id)
                aux_tables.append(aux_table)

        # Concatenate all DataFrames in the list into a single DataFrame
        result_table = pd.concat(aux_tables, ignore_index=True)

        return result_table

    def get_storage_table(self):
        """
        Get the storage table.

        Returns:
            pd.DataFrame: DataFrame containing the storage results.
        """
        df_storage = pd.DataFrame(self.analysis_results.values())
        df_storage.insert(0, 'poi_id', self.analysis_results.keys())

        return df_storage

    def format_analysis_summary(self):
        """
        Format the analysis summary.
        """
        summary = ""

        # Format the analysis_stats dictionary into a human-readable summary
        summary += "Visibility Analysis Summary:\n"
        summary += f"Number of points of interest: {self.analysis_stats['num_points_of_interest']}\n"
        summary += f"Number of cell sites: {self.analysis_stats['num_cell_sites']}\n"
        summary += f"Total visibility checks performed: {self.analysis_stats['total_visibility_checks']}\n"
        summary += f"Average visibility checks per point of interest: {self.analysis_stats['avg_visibility_checks_per_poi']}\n"
        summary += f"Time taken for SRTM data download: {self.analysis_stats['srtm_data_download_time']} seconds\n"
        summary += f"Time taken for analysis: {self.analysis_stats['analysis_time']} seconds\n"
        summary += f"Total time elapsed: {self.analysis_stats['total_time_elapsed']} seconds\n"

        return summary
