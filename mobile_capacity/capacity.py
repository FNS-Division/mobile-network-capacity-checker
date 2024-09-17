from mobile_capacity.spatial import meters_to_degrees_latitude, create_voronoi_cells, vectorized_population_sum
from mobile_capacity.utils import initialize_logger, log_progress_bar
from mobile_capacity.handlers.populationdatahandler import PopulationDataHandler
from mobile_capacity.handlers.srtmdatahandler import SRTMDataHandler
from mobile_capacity.entities.pointofinterest import PointOfInterestCollection
from mobile_capacity.entities.cellsite import CellSiteCollection
from mobile_capacity.entities.visibilitypair import VisibilityPairCollection
from mobile_capacity.visibility import Visibility
import numpy as np
import pandas as pd
import geopandas as gpd
import os
import io


class Capacity:
    def __init__(self,
                 country_code: str,
                 data_dir: str,
                 logs_dir: str,
                 poi: PointOfInterestCollection,
                 cellsites: CellSiteCollection,
                 bw_L850, bw_L1800, bw_L2600,
                 cco, max_radius, min_radius, radius_step, angles_num, rotation_angle, dlthtarg, nonbhu, mbb_subscr,
                 sectors_per_site: int = 3,
                 cellsite_search_radius: int = 35000,
                 poi_antenna_height: int = 15,
                 rb_num_multiplier: int = 5,
                 visibility: VisibilityPairCollection = None,
                 area: gpd.GeoDataFrame = None,
                 dataset_year: int = 2020,
                 one_km_res: bool = True,
                 un_adjusted: bool = True,
                 nbhours: int = 10,
                 oppopshare: int = 50,
                 enable_logging: bool = False):

        # Parameters
        self.country_code = country_code  # Country ISO3 code
        self.bw_L850 = bw_L850  # MHz on L700 to L900 spectrum bandwidth
        self.bw_L1800 = bw_L1800  # MHz on L1800 to L2100 spectrum bandwidth
        self.bw_L2600 = bw_L2600  # MHz on L2300 to L2600 spectrum bandwidth
        self.cco = cco  # Control channel overhead in %
        self.sectors_per_site = sectors_per_site  # Number of sectors per site
        self.angles_num = angles_num  # Number of angles
        self.rotation_angle = rotation_angle  # Rotation angle in degrees
        self.dlthtarg = dlthtarg  # Download throughput target in Mbps
        self.mbb_subscr = mbb_subscr  # Active mobile-broadband subscriptions per 100 people
        self.oppopshare = oppopshare  # Percentage of population using operator services in %
        self.nonbhu = nonbhu  # Connection usage in non-busy hour in %
        self.nbhours = nbhours  # Number of non-busy hours per day
        self.rb_num_multiplier = rb_num_multiplier  # Resource block number multiplier
        self.max_radius = max_radius  # Maximum buffer radius
        self.min_radius = min_radius  # Maximum buffer radius
        self.radius_step = radius_step  # Maximum buffer radius

        # Visibility analysis parameters
        self.cellsite_search_radius = cellsite_search_radius  # Cell site search radius in meters
        self.poi_antenna_height = poi_antenna_height  # Point of interest antenna height in meters

        # Constants
        self.days = 30.4  # Days in one month
        self.minperhour = 60  # Number of minutes per hour
        self.secpermin = 60  # Number of seconds per minute
        self.bitsingbit = 1000000000  # Bits in one gigabit
        self.bitsinkbit = 1000  # Bits in kilobit
        self.bitsingbyte = 8589934592  # Bits in one gigabyte

        # Population data handler variables
        self.dataset_year = dataset_year
        self.one_km_res = one_km_res
        self.un_adjusted = un_adjusted
        self.data_dir = data_dir
        self.logs_dir = logs_dir

        # Logger
        self.enable_logging = enable_logging
        self.logger = None
        if self.enable_logging:
            self.logger = initialize_logger(self.logs_dir)

        # Assign loaded data to class attributes
        self.poi = poi
        self.cellsites = cellsites
        self.visibility = visibility
        self.area = area
        self.mbbt = pd.read_csv("https://zstagigaprodeuw1.blob.core.windows.net/gigainframapkit-public-container/mobile_capacity_data/MobileBB_Traffic_per_Subscr_per_Month.csv")
        self.mbbsubscr = pd.read_csv(
            "https://zstagigaprodeuw1.blob.core.windows.net/gigainframapkit-public-container/mobile_capacity_data/active-mobile-broadband-subscriptions.csv")
        self.mbbtraffic = pd.read_csv(
            "https://zstagigaprodeuw1.blob.core.windows.net/gigainframapkit-public-container/mobile_capacity_data/mobile-broadband-internet-traffic-within-the-country.csv")

        # Load the secure files
        file_paths = {
            'bwdistance_km': os.path.join(self.data_dir, 'input_data', 'carrier_bandwidth', 'bwdistance_km.csv'),
            'bwdlachievbr_kbps': os.path.join(self.data_dir, 'input_data', 'carrier_bandwidth', 'bwdlachievbr_kbps.csv')
        }
        for key, path in file_paths.items():
            if not os.path.exists(path):
                raise ValueError(f"File {key} not found in {path}")
        self.bwdistance_km = pd.read_csv(file_paths['bwdistance_km'])
        self.bwdlachievbr = pd.read_csv(file_paths['bwdlachievbr_kbps'])

        # Set up the population data handler, and get population data
        self.population_data_handler = PopulationDataHandler(
            data_dir=os.path.join(self.data_dir, 'input_data', self.country_code, 'population'),
            country_code=self.country_code,
            dataset_year=self.dataset_year,
            one_km_res=self.one_km_res,
            un_adjusted=self.un_adjusted,
            logger=self.logger,
            enable_logging=self.enable_logging,
            logs_dir=self.logs_dir)
        self.population_data = self._get_population_data()

        # Set up the SRTM daa handler if required
        self.srtm_data_handler = None
        if visibility is None:
            self._log("info", "Setting up SRTM data handler...")
            self.srtm_data_handler = SRTMDataHandler(srtm_directory=os.path.join(self.data_dir, 'input_data', self.country_code, 'srtm1'),
                                                     enable_logging=self.enable_logging, logger=self.logger, logs_dir=self.logs_dir)
            self.srtm_data_handler.check_directory()  # Check if the SRTM directory exists, creates it if not

        # Set up the Visibility analysis if required
        if visibility is None:
            self._log("info", "Setting up visibility analysis...")
            self.visibility = Visibility(
                points_of_interest=self.poi,
                cell_sites=self.cellsites,
                srtm_data_handler=self.srtm_data_handler,
                poi_antenna_height=self.poi_antenna_height,
                allowed_radio_types=['unknown', '2G', '3G', '4G', '5G'],
                earth_radius=6371,
                use_srtm=True,
                refraction_coef=0,
                logger=self.logger,
                logs_dir=self.logs_dir,
                enable_logging=self.enable_logging
            )

    def _get_population_data(self):
        """
        Property that loads and returns population data for the given country and year.
        """
        pop_gdf = self.population_data_handler.population_data
        pop_gdf = gpd.GeoDataFrame(pop_gdf["population"],
                                   geometry=gpd.points_from_xy(pop_gdf["lon"], pop_gdf["lat"]),
                                   crs="EPSG:4326")
        return pop_gdf

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
    def bw(self):
        """
        Returns the total bandwidth in MHz.
        """
        return self.bw_L850 + self.bw_L1800 + self.bw_L2600

    @property
    def udatavmonth_pu(self):
        """
        Returns average mobile broadband user data traffic volume per month in GBs (Gigabytes) for the latest year in the ITU dataset.
        """
        return self.mbbt.loc[self.mbbt["entityIso_mbbsubscr"]
                             == self.country_code, "mbb_traffic_per_subscr_per_month"].item()

    @property
    def udatavmonth_year(self):
        """
        Returns average monthly mobile broadband user data traffic volume in gigabytes (GB).
        """
        return self.mbbt.loc[self.mbbt["entityIso_mbbsubscr"]
                             == self.country_code, "dataYear"].item()

    @property
    def nrb(self):
        """
        Returns the number of resource blocks.
        """
        return self.bw * self.rb_num_multiplier

    @property
    def avrbpdsch(self):
        """
        Returns the average number of RB available for PDSCH in units.
        """
        return ((100 - self.cco) / self.nrb) * 100

    def get_dl_bitrate(self, poi_distances):
        """
        Calculate the downlink bitrate based on the given POI distances.

        Parameters:
        - poi_distances (list): List of POI distances in meters, can contain a single or multiple distances.

        Returns:
        - np.ndarray: Array of downlink bitrates corresponding to each POI distance.

        Note:
        - `bwdistance_k` and `bwdlachievbr` are expected to be pandas DataFrames with columns
          named as `{bandwidth}MHz`.
        """
        if not isinstance(poi_distances, list):
            poi_distances = [poi_distances]

        # Convert input distances to numpy array
        poi_distances = np.array(poi_distances) / 1000  # converts distances in meters to kilometers

        # Create weights
        weights = np.array([self.bw_L850 / self.bw, self.bw_L1800 / self.bw, self.bw_L2600 / self.bw])

        # Array to populate
        dl_bitrates = np.full((len(poi_distances), 3), np.nan)

        for i, bw in enumerate(["L850", "L1800", "L2600"]):
            # Retrieve distance and bitrate arrays for the given bandwidth
            distances_array = self.bwdistance_km[bw].values.flatten()
            bitrate_array = self.bwdlachievbr[bw].values.flatten()

            # Create a mask to find the first distance in distances_array larger than or equal to each POI-tower distance
            mask = (distances_array[np.newaxis, :] >= poi_distances[:, np.newaxis])
            indices = mask.argmax(axis=1)

            # Identify POI distances that do not have a corresponding larger/equal distance in distances_array
            no_larger_equal = ~mask.any(axis=1)

            # Fetch the corresponding bitrate values and handle out-of-bound values
            dl_bitrate = bitrate_array[indices]
            dl_bitrate[no_larger_equal] = np.nan

            # Store the results
            dl_bitrates[:, i] = dl_bitrate

        # Compute the weighted sum of the downlink bitrates
        weighted_sum = np.dot(dl_bitrates, weights)

        return weighted_sum

    def poiddatareq(self, d):
        """
        Calculate the number of resource blocks required to meet the download throughput target for each distance.

        Parameters:
        - d (list): List of distances from the tower in meters.

        Returns:
        - list: List of resource blocks required to meet the download throughput target for each distance.
                    Returns np.inf for distances exceeding max_radius or None if an error occurs.
        """
        if not isinstance(d, list):
            d = [d]

        results = []
        try:
            # Get the downlink bitrate for the given distances
            dl_bitrate = self.get_dl_bitrate(poi_distances=d)
            for i, distance in enumerate(d):
                # Compute the number of resource blocks required to meet the download throughput target
                if distance > self.max_radius:
                    rbdlthtarg = np.inf
                else:
                    rbdlthtarg = self.dlthtarg * 1024 / (dl_bitrate[i] / self.avrbpdsch)
                # Log the result for each distance
                self._log("debug", f'distance = {distance}, rbdlthtarg = {rbdlthtarg}')
                results.append(rbdlthtarg)
        except ValueError as e:
            self._log("info", f"ValueError in poiddatareq: {e}")
            results = [None] * len(d)
        except Exception as e:
            self._log("info", f"An error occurred in poiddatareq: {e}")
            results = [None] * len(d)
        return results

    def brrbpopcd(self, popcd):
        """
        Bitrate per resource block at population center distance.

        Parameters:
        - popcd (int): Population center distance in meters.

        Returns:
        - brrbpopcd (float): Bitrate per resource block at population center distance in kbps.
        """
        if not isinstance(popcd, list):
            popcd = [popcd]

        results = []
        try:
            # Get the downlink bitrate for the given distances
            dl_bitrate = self.get_dl_bitrate(poi_distances=popcd)
            for i, distance in enumerate(popcd):
                # Compute the bitrate per resource block at the population center distance
                brrbpopcd = dl_bitrate[i] / self.avrbpdsch
                self._log("debug", f'population centre distance = {distance}, brrbpopcd = {brrbpopcd}')
                results.append(brrbpopcd)
        except ValueError as e:
            self._log("info", f"ValueError in brrbpopcd: {e}")
            results = [None] * len(popcd)
        except Exception as e:
            self._log("info", f"An error occurred in brrbpopcd: {e}")
            results = [None] * len(popcd)
        return results

    def avubrnonbh(self, udatavmonth):
        """
        Average user bitrate in non-busy hour.

        Parameters:
        - udatavmonth (int): Average user data traffic volume per month in GB (Gigabyte)
        - nonbhu (int): Connection usage in non-busy hour in %

        Returns:
        - avubrnonbh (float): Average user bitrate in non-busy hour in kbps.

        Note:
        """
        # avubrnonbh = (((((((udatavmonth/days)/nbhours)*nonbhu/100)/minperhour)/secpermin)*bitsingbyte)/bitsinkbit)
        avubrnonbh = (
            ((((((udatavmonth /
                  self.days) /
                 self.nbhours) *
                self.nonbhu
                / 100) /
               self.minperhour) /
              self.secpermin) *
             self.bitsingbyte) /
            self.bitsinkbit)

        self._log("debug", f'avubrnonbh = {avubrnonbh}')

        return avubrnonbh

    def upopbr(self, avubrnonbh, pop):
        """
        User Population Bitrate per sector, kbps

        Parameters:
        - avubrnonbh (float): Average user bitrate in non-busy hour in kbps.
        - upop (int): User population number, people.
        - oppopshare (int): Percentage of population using operator services in %.
        - sectors_per_site (int): No. of Frequency Bands on Site

        Returns:
        - upopbr (float): User Population Bitrate in kbps.

        Note:
        """
        upopbr = avubrnonbh * pop * (self.mbb_subscr / 100) * (self.oppopshare / 100) / self.sectors_per_site
        self._log("debug", f'upopbr = {upopbr}')

        return upopbr

    def upoprbu(self, upopbr, brrbpopcd):
        """
        User population resource blocks utilisation

        Parameters:
        - upopbr (float): User Population Bitrate in kbps.
        - brrbpopcd (float): Bitrate per resource block at population center distance in kbps.

        Returns:
        - upoprbu (float): User population resource blocks utilisation in units.

        Note:
        """
        if not isinstance(upopbr, list):
            upopbr = [upopbr]
        if not isinstance(brrbpopcd, list):
            brrbpopcd = [brrbpopcd]
        if len(upopbr) > 1:
            raise ValueError("upopbr is not of length 1.")

        # Calculate user population resource blocks utilisation in units.
        upoprbu = [upopbr[0] / denom for denom in brrbpopcd]
        self._log("debug", f'upoprbu = {upoprbu}')
        return upoprbu

    def cellavcap(self, avrbpdsch, upoprbu):
        """
        Cell site available capacity check.

        Parameters:
        - avrbpdsch (float): Resource blocks available for PDSCH, resource blocks.
        - upoprbu (float): User population resource blocks utilisation, resource blocks.

        Returns:
        - cellavcap (float): Shows available capacity at the cell site, resource blocks.

        Note:
        """
        if not isinstance(avrbpdsch, np.ndarray):
            avrbpdsch = np.array(avrbpdsch)
        # Check if upoprbu is not already a NumPy array, convert if necessary
        if not isinstance(upoprbu, np.ndarray):
            upoprbu = np.array(upoprbu)

        # Cell site available capacity.
        cellavcap = avrbpdsch - upoprbu
        cellavcap = cellavcap.tolist()
        self._log("debug", f'cellavcap = {cellavcap}')

        return cellavcap

    def sufcapch(self, cellavcap, rbdlthtarg):
        """
        Sufficient capacity check

        Parameters:
        - cellavcap (float): Shows available capacity at the cell site, resource blocks.
        - rbdlthtarg (float): RB number required to meet download throughput target in units.

        Returns:
        - sufcapch (boolean): Shows that capacity requirement is satisfied.

        Note:
        """
        if not isinstance(cellavcap, np.ndarray):
            cellavcap = np.array(cellavcap)
        if not isinstance(rbdlthtarg, np.ndarray):
            rbdlthtarg = np.array(rbdlthtarg)

        sufcapch = cellavcap > rbdlthtarg
        sufcapch = sufcapch.tolist()
        self._log("debug", f'sufcapch = {sufcapch}')
        return sufcapch

    def capacity_checker(self, d, popcd, udatavmonth, pop):
        """
        Performs a capacity check using the provided parameters.

        Parameters:
        - d (float): Distance to the Point of Interest (POI) for data rate calculation, in meters.
        - popcd (float): Population center distance parameter, in meters.
        - udatavmonth (float): Monthly data usage per user, in gigabytes.
        - pop (float): Population parameter.

        Returns:
        - tuple: A tuple containing:
            - float: User population bitrate (upopbr).
            - float: User population resource block utilization (upoprbu).
            - float: Available cell capacity (cellavcap).
            - float: Capacity check result (capcheck).

        This method calculates the following independent functions:
        - `rbdlthtarg`: Target data rate based on the distance `d`.
        - `brrpopcd`: Bitrate per resource block based on the population center distance `popcd`.
        - `avubrnonbh`: Average user bitrate based on the monthly data volume `udatavmonth`.

        It then calculates the following dependent functions:
        - `upopbr`: User population bitrate based on the average user bitrate `avubrnonbh` and the population `pop`.
        - `upoprbu`: User population resource block utilization based on the user population bitrate `upopbr` and the bitrate per resource block `brrpopcd`.
        - `cellavcap`: Available cell capacity based on `avrbpdsch` and `upoprbu`.
        - `capcheck`: Capacity sufficiency check based on `cellavcap` and the target data rate `rbdlthtarg`.

        The capacity check result is stored in `self.capcheck_result`.
        """

        # Independent functions
        rbdlthtarg = self.poiddatareq(d)
        brrpopcd = self.brrbpopcd(popcd)
        avubrnonbh = self.avubrnonbh(udatavmonth)

        # Dependent functions
        upopbr = self.upopbr(avubrnonbh, pop)
        upoprbu = self.upoprbu(upopbr, brrpopcd)
        cellavcap = self.cellavcap(self.avrbpdsch, upoprbu)
        capcheck = self.sufcapch(cellavcap, rbdlthtarg)

        # Store and return the result
        dict_result = {"upopbr": upopbr, "upoprbu": upoprbu, "cellavcap": cellavcap, "capcheck": capcheck}
        return dict_result

    # def visibility_analysis(self):
    #     """
    #     Perform visibility analysis to determine the number of visible cell sites from each point of interest (POI).
    #     """
    #     self._log("info", "Triggering visibiliy analysis.")
    #     visibility = Visibility(
    #         points_of_interest=self.poi,
    #         cell_sites=self.cellsites,
    #         srtm_data_handler=self.srtm_data_handler,
    #         poi_antenna_height=self.poi_antenna_height,
    #         allowed_radio_types=['unknown', '2G', '3G', '4G', '5G'],
    #         earth_radius=6371,
    #         use_srtm=True,
    #         refraction_coef=0,
    #         logger=self.logger,
    #         logs_dir=self.logs_dir,
    #         enable_logging=self.enable_logging
    #     )
    #     visibility.perform_analysis()
    #     visibility_results = visibility.get_results_table()
    #     visibility_results = visibility_results[['poi_id', 'ict_id', 'order', 'ground_distance']]
    #     visibility_results.loc[visibility_results["ground_distance"].isna(), "ground_distance"] = np.inf
    #     return visibility_results

    def calculate_buffer_areas(self):
        """
        Calculates buffer areas around cell sites adjusted with Voronoi polygons
        representing cell sites service areas. Breaks down buffer areas into rings of a specified radius to segment
        demand estimates by distance.

        Returns:
        - buffer_cellsites (geodataframe): Cellsites data containing buffer areas around cell site locations.

        Note:
        """
        def _get_visibility_status(row):
            log_progress_bar(self.logger, row.name + 1, len(pois_within_cellsites), prefix='Visibility Analysis:', length=50)
            if pd.isna(row["ict_id"]):
                return np.nan
            else:
                return self.visibility.perform_pair_analysis(row["poi_id"], row["ict_id"])

        # Copy input data
        cellsites = self.cellsites.data.copy()
        poi = self.poi.data.copy()
        population_gdf = self.population_data.copy()

        # Create GeoDataFrames for cell sites and POIs
        cellsites_gdf = gpd.GeoDataFrame(
            cellsites, geometry=gpd.points_from_xy(cellsites.lon, cellsites.lat), crs="4326"
        ).drop_duplicates(subset='ict_id')  # Drop duplicates in one step

        pois_gdf = gpd.GeoDataFrame(
            poi, geometry=gpd.points_from_xy(poi.lon, poi.lat), crs="4326"
        )

        # Estimate UTM CRS and reproject the GeoDataFrames
        poi_utm = pois_gdf.estimate_utm_crs()
        for gdf in [pois_gdf, population_gdf, cellsites_gdf]:
            gdf.to_crs(poi_utm, inplace=True)

        # Create Voronoi polygons for the cell sites
        cellsites_gdf = create_voronoi_cells(cellsites_gdf, self.area)

        # Generate buffers, rings, and clipped areas
        for radius in range(self.min_radius, self.max_radius + 1, self.radius_step):
            buffer_col = f'buffer_{radius}'
            ring_col = f'ring_{radius}'
            clring_col = f'clring_{radius}'
            clbuffer_col = f'clbuffer_{radius}'

            # Create buffer areas
            cellsites_gdf[buffer_col] = cellsites_gdf.geometry.buffer(radius)

            # Create rings based on buffer differences
            if radius == self.min_radius:
                cellsites_gdf[ring_col] = cellsites_gdf[buffer_col]
            else:
                prev_buffer = f'buffer_{radius - self.radius_step}'
                cellsites_gdf[ring_col] = cellsites_gdf[buffer_col].difference(cellsites_gdf[prev_buffer])

            # Intersect rings and buffers with Voronoi polygons
            cellsites_gdf[clring_col] = cellsites_gdf.apply(lambda row: row[ring_col].intersection(row.voronoi_polygons), axis=1)
            cellsites_gdf[clbuffer_col] = cellsites_gdf.apply(lambda row: row[buffer_col].intersection(row.voronoi_polygons), axis=1)

        # Convert buffers to long format (only for the maximum radius buffer)
        buffers_gdf = cellsites_gdf[['ict_id', f'clbuffer_{self.max_radius}']].rename(columns={f'clbuffer_{self.max_radius}': 'geometry'})
        buffers_gdf = gpd.GeoDataFrame(buffers_gdf, geometry='geometry', crs=poi_utm)

        # Convert rings to long format
        rings_columns = ['ict_id'] + [col for col in cellsites_gdf.columns if col.startswith('clring')]
        rings_gdf = cellsites_gdf[rings_columns].melt(id_vars='ict_id', var_name='buffer_column', value_name='geometry')
        rings_gdf['buffer'] = rings_gdf['buffer_column'].str.extract(r'(\d+)').astype(int)
        rings_gdf = gpd.GeoDataFrame(rings_gdf.drop('buffer_column', axis=1), geometry='geometry', crs=poi_utm)
        rings_gdf['radius'] = rings_gdf['buffer'] - self.radius_step / 2

        # Match POIs to cell towers based on coverage area
        pois_within_cellsites = gpd.sjoin(pois_gdf, buffers_gdf, how='left', predicate='within')            
        pois_within_cellsites["visible"] = pois_within_cellsites.apply(_get_visibility_status, axis=1)

        # PRINT
        print("pois_within_cellsites")
        print(pois_within_cellsites.shape)
        print(pois_within_cellsites[["poi_id", "ict_id", "index_right", "visible"]])

    # def calculate_buffer_areas(self):
    #     """
    #     Calculates buffer areas around cell sites adjusted with Voronoi polygons
    #     representing cell sites service areas. Breaks down buffer areas into rings of a specified radius to segment
    #     demand estimates by distance.

    #     Parameters:
    #     - cellsites (geodataframe): Cellsites data containing cell site latitudes and longitudes.
    #     - min_radius (int): Minimum radius around cell site location for population calculation, meters.
    #     - max_radius (int): Maximum radius around cell site location for population calculation, meters.
    #     - radius_step (int): Radius step size for population calculation, meters.

    #     Returns:
    #     - buffer_cellsites (geodataframe): Cellsites data containing buffer areas around cell site locations.

    #     Note:
    #     """
    #     def _poi_sufcapch(poi, visibility, buffer_cellsites_result):
    #         """
    #         Process and merge POI, visibility, and cell site data.

    #         Args:
    #         poi (DataFrame): POI data with 'poi_id', 'lat', and 'lon'.
    #         visibility (DataFrame): Visibility data with 'poi_id', 'order', 'ground_distance', and 'ict_id'.
    #         buffer_cellsites_result (DataFrame): Cell site data with 'ict_id' and other relevant columns.

    #         Returns:
    #         DataFrame: Merged and processed data indexed by 'poi_id'.
    #         """
    #         # Filter visibility DataFrame to include only rows where order is 1
    #         visibility_filtered = visibility.loc[visibility["order"] == 1, :]

    #         # Merge POI DataFrame with filtered visibility DataFrame on 'poi_id'
    #         poi_merged = poi[["poi_id", "lat", "lon"]].merge(visibility_filtered, on="poi_id", how="left")

    #         # Drop 'lat' and 'lon' columns from buffer_cellsites_result
    #         buffer_cellsites_result_cleaned = buffer_cellsites_result.drop(columns=["lat", "lon"], inplace=False)

    #         # Merge the POI data with the cleaned buffer_cellsites_result on 'ict_id'
    #         poi_data_merged = (
    #             poi_merged[['poi_id', 'lat', 'lon', 'ground_distance', 'ict_id']]
    #             .merge(buffer_cellsites_result_cleaned, on='ict_id', how='left')
    #             .drop(columns="ict_id")
    #             .set_index('poi_id')
    #         )

    #         return poi_data_merged

    #     # Copy input data
    #     cellsites = self.cellsites.data.copy()
    #     poi = self.poi.data.copy()

    #     # For distance conversion from meters to degrees
    #     central_latitude = cellsites['lat'].mean()

    #     # Convert cell sites to a GeoDataFrame - assuming the input data used the EPSG 4326 CRS
    #     # Re-project to the same CRS as the population vector data
    #     cellsites_gdf = gpd.GeoDataFrame(
    #         geometry=gpd.points_from_xy(
    #             x=cellsites.lon,
    #             y=cellsites.lat),
    #         crs="4326",
    #         data=cellsites)
    #     cellsites_gdf = cellsites_gdf.to_crs(self.population_data.crs)

    #     # Drop duplicated ict_ids rows from cellsites to avoid redundancy
    #     # during Voronoi polygons generation
    #     cellsites_gdf = cellsites_gdf.drop(
    #         cellsites_gdf.loc[cellsites_gdf.ict_id.duplicated(), :].index)

    #     # Create buffers for different radii around the cell sites
    #     # This loop adds columns corresponding to the buffer areas and rings
    #     buffer_cellsites = cellsites_gdf.copy().reset_index()
    #     for radius in range(self.min_radius, self.max_radius + 1, self.radius_step):
    #         # convert radius from meters to degrees
    #         radius_in_degrees = meters_to_degrees_latitude(
    #             radius, central_latitude)
    #         # create buffer columns in the geodataframe and insert buffer
    #         # geometries there
    #         buffer_cellsites[f'buffer_{radius}'] = buffer_cellsites['geometry'].apply(
    #             lambda point: point.buffer(radius_in_degrees))
    #         # create population center distances columns in the geodataframe
    #         # and insert distances values there
    #         buffer_cellsites[f'popcd_{radius}'] = int(radius - self.radius_step / 2)
    #         # create ring columns in the geodataframe and insert ring
    #         # geometries there
    #         if radius == self.min_radius:
    #             # first ring is the same as the first buffer area
    #             buffer_cellsites[f'ring_{radius}'] = buffer_cellsites[f'buffer_{radius}']
    #         else:
    #             # ring is the difference between current buffer and the
    #             # previous buffer
    #             buffer_cellsites[f'ring_{radius}'] = buffer_cellsites[f'buffer_{radius}'].difference(
    #                 buffer_cellsites[f'buffer_{radius - self.radius_step}'])

    #     # Create Voronoi cells, this adds a new column with the geometry
    #     # corresponding to the Voronoi polygons
    #     buffer_cellsites = create_voronoi_cells(buffer_cellsites, self.area)

    #     # Check capacity for each cell site
    #     for radius in range(self.min_radius, self.max_radius + 1, self.radius_step):
    #         # Clip ring areas with Voronoi polygons to identify cell sites
    #         # service areas within each ring.
    #         buffer_cellsites[f'clring_{radius}'] = buffer_cellsites.apply(
    #             lambda row: row[f'ring_{radius}'].intersection(row['voronoi_polygons']), axis=1)
    #         # Calculate population count within clipped ring areas.
    #         buffer_cellsites[f'pop_clring_{radius}'] = vectorized_population_sum(buffer_cellsites, self.population_data, radius)
    #         # Calculate avubrnonbh, the average user bitrate in non-busy hour in kbps, from country-level statistics
    #         avubrnonbh = self.avubrnonbh(self.udatavmonth_pu)
    #         # Calculate user population bitrate in kbps within clipped ring areas.
    #         buffer_cellsites[f'upopbr_{radius}'] = buffer_cellsites[f'pop_clring_{radius}'].apply(
    #             lambda row: self.upopbr(avubrnonbh, row))
    #         # Calculate bitrate per resource block at population center distance in kbps.
    #         buffer_cellsites[f'brrbpopcd_{radius}'] = buffer_cellsites[f'popcd_{radius}'].apply(
    #             lambda row: self.brrbpopcd(row)
    #         )
    #         # Calculate user population resource blocks utilisation within clipped ring areas.
    #         buffer_cellsites[f'upoprbu_{radius}'] = buffer_cellsites.apply(
    #             lambda row: self.upoprbu(row[f'upopbr_{radius}'], row[f'brrbpopcd_{radius}']), axis=1
    #         )
    #     # Calculate total number of cell site required resource blocks
    #     upoprbu_columns = [col for col in buffer_cellsites.columns if col.startswith('upoprbu_')]
    #     buffer_cellsites['upoprbu_total'] = buffer_cellsites[upoprbu_columns].apply(lambda row: [sum(x) for x in zip(*row)], axis=1)

    #     # Calculate cell site available capacity
    #     buffer_cellsites['cellavcap'] = buffer_cellsites['upoprbu_total'].apply(
    #         lambda row: self.cellavcap(self.avrbpdsch, row))

    #     # Store the buffer analysis results
    #     self.buffer_cellsites_result = buffer_cellsites.set_index('ict_id').drop(columns='index')

    #     # Trigger visibility analysis if required
    #     if self.visibility:
    #         visibility = self.visibility.data.copy()
    #     else:
    #         visibility = self.visibility_analysis()

    #     # Merge POI, visibility, and cell site data
    #     poi_data_merged = _poi_sufcapch(poi, visibility, self.buffer_cellsites_result)

    #     poi_data_merged['rbdlthtarg'] = poi_data_merged['ground_distance'].apply(
    #         lambda row: self.poiddatareq(row)
    #     )
    #     poi_data_merged['sufcapch'] = poi_data_merged.apply(
    #         lambda row: self.sufcapch(row['cellavcap'], row['rbdlthtarg'])[0], axis=1
    #     )
    #     self.poi_sufcapch_result = poi_data_merged
    #     return self.buffer_cellsites_result, self.poi_sufcapch_result

    def mbbtps(self):
        """
        Reads rows with the latest year data available for both
        mobile-broadband Internet traffic (within the country) and active mobile-broadband subscriptions
        and calculates Mobile broadband internet traffic (within the country) per active mobile broadband subscription
        per month.

        Parameters:
        - mbbsubscr (dataframe): Active mobile broadband subscriptions (ITU, https://datahub.itu.int/data/?i=11632)
        - mbbtraffic (dataframe): Mobile broadband internet traffic (within the country) in Exabytes (ITU, https://datahub.itu.int/data/?i=13068)

        Returns:
        - DataFrame containing mobile broadband internet traffic per subscription per month column in Gigabytes.
        """
        # Combine data tables
        df = pd.merge(self.mbbsubscr, self.mbbtraffic, on=[
                      'entityName', 'dataYear'], suffixes=('_mbbsubscr', '_mbbtraffic'))

        filtered_data = df[(df['dataValue_mbbsubscr'] != 0)
                           & (df['dataValue_mbbtraffic'] != 0)]

        # Group by `entityName` and find the latest `dataYear` for each group
        latest_years = filtered_data.groupby(
            'entityName')['dataYear'].max().reset_index()

        # Filter the original DataFrame
        # Join the DataFrame of latest years back to the filtered DataFrame
        # to get only the rows for the latest year of each entity with non-zero `dataValue`
        df = pd.merge(filtered_data, latest_years,
                      on=['entityName', 'dataYear'])

        # Calculate mobile broadband internet traffic per subscription per month column in Gigabytes.
        df['mbb_traffic_per_subscr_per_month'] = df['dataValue_mbbtraffic'] * \
            1024**3 / df['dataValue_mbbsubscr'] / 12

        # Select relevant columns
        df = df.loc[:, ['entityName', 'entityIso_mbbsubscr', 'dataValue_mbbsubscr',
                        'dataValue_mbbtraffic', 'mbb_traffic_per_subscr_per_month', 'dataYear']]

        # Store the result
        self.mbbtps_result = df
        return df
