import numpy as np
import pandas as pd
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

    This class performs line-of-sight analysis between points of interest (POIs) and cell sites
    using SRTM (Shuttle Radar Topography Mission) data for elevation information. It takes into
    account Earth's curvature and atmospheric refraction for accurate visibility calculations.

    Attributes:
        points_of_interest (PointOfInterestCollection): Collection of points of interest.
        cell_sites (CellSiteCollection): Collection of cell sites.
        srtm_data_handler (SRTMDataHandler): Handler for SRTM data.
        poi_antenna_height (float): Antenna height at POI locations (meters).
        allowed_radio_types (list): List of allowed radio types for cell sites.
        earth_radius (float): Radius of the Earth (kilometers).
        refraction_coef (float): Atmospheric refraction coefficient.
        use_srtm (bool): Flag to use SRTM data for elevation information.
        logger (Logger): Logger instance for logging operations.

    Args:
        points_of_interest (PointOfInterestCollection): Collection of points of interest.
        cell_sites (CellSiteCollection): Collection of cell sites.
        srtm_data_handler (SRTMDataHandler, optional): Handler for SRTM data. Required if use_srtm is True.
        poi_antenna_height (float, optional): Antenna height at POI locations (meters). Defaults to 15.
        allowed_radio_types (list, optional): List of allowed radio types. Defaults to ['unknown', '2G', '3G', '4G', '5G'].
        earth_radius (float, optional): Radius of the Earth (kilometers). Defaults to 6371.
        refraction_coef (float, optional): Atmospheric refraction coefficient. Defaults to 0.
        use_srtm (bool, optional): Flag to use SRTM data for elevation information. Defaults to True.
        logger (Logger, optional): Logger instance for logging. If None, a default logger is created.
        enable_logging (bool, optional): Flag to enable logging. Defaults to False.
        logs_dir (str, optional): Directory for log files. Required if enable_logging is True and logger is None.
    """

    def __init__(self,
                 points_of_interest: PointOfInterestCollection,
                 cell_sites: CellSiteCollection,
                 srtm_data_handler: SRTMDataHandler = None,
                 poi_antenna_height=15,
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
        self.poi_antenna_height = poi_antenna_height
        self.allowed_radio_types = allowed_radio_types
        self.earth_radius = earth_radius
        self.refraction_coef = refraction_coef
        self.use_srtm = use_srtm
        self.logs_dir = logs_dir
        self.enable_logging = enable_logging
        # Create index mappings for faster lookups
        self.poi_id_to_index = {poi_id: index for index, poi_id in enumerate(self.points_of_interest.data["poi_id"])}
        self.ict_id_to_index = {ict_id: index for index, ict_id in enumerate(self.cell_sites.data["ict_id"])}

        self.analysis_stats = dict(
            num_points_of_interest=len(points_of_interest),
            num_cell_sites=len(cell_sites),
        )

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

        # Retrieve SRTM data files for points of interest and cell sites
        srtm_time = time.time()
        if self.use_srtm:
            self.retrieve_srtm_data_files()
            self.analysis_stats['srtm_data_download_time'] = round(time.time() - srtm_time, 2)
            self.srtm_collection = self.srtm_data_handler.srtmheightmapcollection
        else:
            self._log('info', 'SRTM data will not be used')
            self.analysis_stats['srtm_data_download_time'] = 0

        self.analysis_param = dict(
            poi_antenna_height=poi_antenna_height,
            allowed_radio_types=allowed_radio_types,
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
            self._log('error', f'{len(unmatched_indices)} locations could not be matched with SRTM data files. Please see "unmatched_indices" property of the visibility intance!')

        data_files_to_download = self.srtm_data_handler.datafile_check(list(unique_data_files))

        self._log('info', 'Downloading SRTM data files...')

        self.srtm_data_handler.download_data_files(data_files=list(data_files_to_download))

    def perform_pair_analysis(self, poi_id, ict_id):
        """
        Perform visibility analysis between a point of interest and a cell site.

        Args:
            poi_id (str): ID of the point of interest.
            ict_id (str): ID of the cell site.

        Returns:
            bool: True if there's a clear line of sight, False otherwise.

        Raises:
            ValueError: If provided poi_id or ict_id is invalid.
        """
        # Retrieve indices for faster entity lookup
        poi_index = self.poi_id_to_index.get(poi_id)
        cellsite_index = self.ict_id_to_index.get(ict_id)

        if poi_index is None or cellsite_index is None:
            raise ValueError(f"Invalid poi_id {poi_id} or ict_id {ict_id}")

        # Get the point of interest and cell site entities
        poi = self.points_of_interest.get_nth_entity(poi_index)
        cellsite = self.cell_sites.get_nth_entity(cellsite_index)

        # Calculate the ground distance between POI and cell site
        ground_distance = round(poi.get_distance(cellsite))

        # Calculate antenna altitudes for the point of interest and cell site
        try:
            poi_antenna_altitude = self.srtm_collection.get_altitude(poi.lat, poi.lon) + self.poi_antenna_height
            cellsite_antenna_altitude = self.srtm_collection.get_altitude(cellsite.lat, cellsite.lon) + cellsite.antenna_height
        except AttributeError as e:
            self._log('error', f"Error accessing SRTM data: {str(e)}")
            return False

        # Check if the cell site is within the search radius and visible based on horizon distances
        if ground_distance > sum_of_horizon_distances(poi_antenna_altitude, cellsite_antenna_altitude):
            return False

        # Get elevation profile between POI and cell site
        try:
            elevation_profile = self.srtm_collection.get_elevation_profile(poi.lat, poi.lon, cellsite.lat, cellsite.lon)
            e_profile, d_profile = zip(*[(i.elevation, i.distance) for i in elevation_profile])
        except Exception as e:
            self._log('error', f"Error retrieving elevation profile: {str(e)}")
            return False

        # Adjust extremely high elevation values (possibly due to data errors)
        e_profile = [e - 65535 if e > 65000 else e for e in e_profile]

        # Adjust elevation profile to incorporate Earth curvature
        curvature_adjustment = [adjust_elevation(poi_antenna_altitude, d, self.earth_radius, self.refraction_coef)
                                for d in d_profile]
        adjusted_e_profile = np.add(e_profile, curvature_adjustment)

        # Calculate line of sight profile
        los_profile = np.linspace(
            adjusted_e_profile[0] + self.poi_antenna_height,
            adjusted_e_profile[-1] + cellsite.antenna_height,
            len(e_profile)
        )

        # Check if line of sight is clear (all points above terrain)
        has_line_of_sight = np.all(los_profile > adjusted_e_profile)

        return ground_distance, has_line_of_sight
