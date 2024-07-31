from mobile_capacity.spatial import meters_to_degrees_latitude, create_voronoi_cells, get_population_sum
from mobile_capacity.utils import load_data, initialize_logger
import numpy as np
import pandas as pd
import geopandas as gpd
import os


class Capacity:
    def __init__(self, data_files: dict, country_name: str,
                 bw, cco, fb_per_site, max_radius, min_radius, radius_step, angles_num,
                 rotation_angle, dlthtarg, nonbhu, root_dir, rb_num_multiplier=5,
                 nbhours=10, oppopshare=50, enable_logging=False):

        # Data storage
        self.root_dir = root_dir
        self.input_data_path = os.path.join(root_dir, 'data', 'input_data')
        self.output_data_path = os.path.join(root_dir, 'data', 'output_data')
        self.data_files = data_files

        # Logger
        self.logger = None
        if enable_logging:
            self.logger = initialize_logger(__name__)

        # Input validation
        self._validate_input(bw)

        # Parameters
        self.country_name = country_name  # Country name
        self.bw = bw  # Bandwidth in MHz
        self.cco = cco  # Control channel overhead in %
        self.fb_per_site = fb_per_site  # Number of frequency bands per site
        self.angles_num = angles_num  # Number of angles PLACEHOLDER
        self.rotation_angle = rotation_angle  # Rotation angle in degrees PLACEHOLDER
        self.dlthtarg = dlthtarg  # Download throughput target in Mbps
        self.oppopshare = oppopshare  # Percentage of population using operator services in %
        self.nonbhu = nonbhu  # Connection usage in non-busy hour in %
        self.days = 30.4  # Days in one month
        self.nbhours = nbhours  # number of non-busy hours per day
        self.rb_num_multiplier = rb_num_multiplier  # Resource block number multiplier
        self.max_radius = max_radius  # maximum buffer radius
        self.min_radius = min_radius  # maximum buffer radius
        self.radius_step = radius_step  # maximum buffer radius

        # Constants
        self.minperhour = 60  # number of minutes per hour
        self.secpermin = 60  # number of seconds per minute
        self.bitsingbit = 1000000000  # bits in one gigabit
        self.bitsinkbit = 1000  # bits in kilobit
        self.bitsingbyte = 8589934592  # bits in one gigabyte

        # Load data using the imported function
        loaded_data = load_data(self.root_dir, self.data_files, self.logger)

        # Assign loaded data to class attributes
        self.bwdistance_km = loaded_data['bwdistance_km']
        self.bwdlachievbr = loaded_data['bwdlachievbr']
        self.cellsites = loaded_data['cellsites']
        self.mbbt = loaded_data['mbbt']
        self.poi = loaded_data['poi']
        self.visibility = loaded_data['poi_visibility']
        self.mbbsubscr = loaded_data['mbbsubscr']
        self.mbbtraffic = loaded_data['mbbtraffic']
        self.area = loaded_data['area']
        self.population = loaded_data['population']

    def _validate_input(self, bw):
        """Validates the bandwidth (bw) parameter."""
        valid_bw_values = {5, 10, 15, 20}
        if bw not in valid_bw_values:
            raise ValueError(f"Invalid bandwidth (bw) value: {bw}. Must be one of {valid_bw_values}.")

    def _log_info(self, message):
        """Logs an info message."""
        if self.logger:
            self.logger.info(message)

    @property
    def udatavmonth_pu(self):
        """
        Returns average mobile broadband user data traffic volume per month in GBs (Gigabytes) for the latest year in the ITU dataset.
        """
        return self.mbbt.loc[self.mbbt["entityName"]
                             == self.country_name, "mbb_traffic_per_subscr_per_month"].item()

    @property
    def udatavmonth_year(self):
        """
        Returns average monthly mobile broadband user data traffic volume in gigabytes (GB).
        """
        return self.mbbt.loc[self.mbbt["entityName"]
                             == self.country_name, "dataYear"].item()

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
        - type (str): Type of calculation, either "poidatareq" or "brrpopcd".

        Returns:
        - np.ndarray: Array of downlink bitrates corresponding to each POI distance.

        Note:
        - `bwdistance_k` and `bwdlachievbr` are expected to be pandas DataFrames with columns
          named as `{bandwidth}MHz`.
        """
        if not isinstance(poi_distances, list):
            poi_distances = [poi_distances]

        # Convert input distances to numpy array
        poi_distances = np.array(poi_distances) / 1000

        # Retrieve distance and bitrate arrays for the given bandwidth
        distances_array = self.bwdistance_km[[f'{self.bw}MHz']].values.flatten()
        bitrate_array = self.bwdlachievbr[[f'{self.bw}MHz']].values.flatten()

        # Create a mask to find the first distance in distances_array larger than or equal to each POI-tower distance
        mask = (distances_array[np.newaxis, :] >= poi_distances[:, np.newaxis])
        indices = mask.argmax(axis=1)

        # Identify POI distances that do not have a corresponding larger/equal distance in distances_array
        no_larger_equal = ~mask.any(axis=1)

        # Fetch the corresponding bitrate values and handle out-of-bound values
        dl_bitrate = bitrate_array[indices]
        dl_bitrate[no_larger_equal] = np.nan

        return dl_bitrate

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
                self._log_info(f'distance = {distance}, rbdlthtarg = {rbdlthtarg}')
                results.append(rbdlthtarg)
        except ValueError as e:
            self._log_info(f"ValueError in poiddatareq: {e}")
            results = [None] * len(d)
        except Exception as e:
            self._log_info(f"An error occurred in poiddatareq: {e}")
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
                self._log_info(f'population centre distance = {distance}, brrbpopcd = {brrbpopcd}')
                results.append(brrbpopcd)
        except ValueError as e:
            self._log_info(f"ValueError in brrbpopcd: {e}")
            results = [None] * len(popcd)
        except Exception as e:
            self._log_info(f"An error occurred in brrbpopcd: {e}")
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

        self._log_info(f'avubrnonbh = {avubrnonbh}')

        return avubrnonbh

    def upopbr(self, avubrnonbh, pop):
        """
        User Population Bitrate per frequency band, kbps

        Parameters:
        - avubrnonbh (float): Average user bitrate in non-busy hour in kbps.
        - upop (int): User population number, people.
        - oppopshare (int): Percentage of population using operator services in %.
        - fb_per_site (int): No. of Frequency Bands on Site

        Returns:
        - upopbr (float): User Population Bitrate in kbps.

        Note:
        """
        upopbr = avubrnonbh * pop * self.oppopshare / 100 / self.fb_per_site
        self._log_info(f'upopbr = {upopbr}')

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
        self._log_info(f'upoprbu = {upoprbu}')
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
        self._log_info(f'cellavcap = {cellavcap}')

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
        self._log_info(f'sufcapch = {sufcapch}')
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

    def calculate_buffer_areas(self):
        """
        Calculates buffer areas around cell sites adjusted with Voronoi polygons
        representing cell sites service areas. Breaks down buffer areas into rings of a specified radius to segment
        demand estimates by distance.

        Parameters:
        - cellsites (geodataframe): Cellsites data containing cell site latitudes and longitudes.
        - min_radius (int): Minimum radius around cell site location for population calculation, meters.
        - max_radius (int): Maximum radius around cell site location for population calculation, meters.
        - radius_step (int): Radius step size for population calculation, meters.

        Returns:
        - buffer_cellsites (geodataframe): Cellsites data containing buffer areas around cell site locations.

        Note:
        """
        def _poi_sufcapch(visibility, buffer_cellsites_result):
            visibility = visibility.loc[visibility["order"] == 1, :]
            poi_data_merged = visibility[['poi_id', 'is_visible', 'ground_distance', 'ict_id']].merge(buffer_cellsites_result,
                                                                                                      left_on='ict_id',
                                                                                                      right_on='ict_id',
                                                                                                      how='left').drop(columns="ict_id").set_index('poi_id')
            return poi_data_merged

        # Copy input data
        cellsites = self.cellsites.copy()
        # for distance conversion from meters to degrees
        central_latitude = cellsites['lat'].mean()

        # Convert cell sites to a GeoDataFrame - assuming the input data used the EPSG 4326 CRS
        # Re-project to the same CRS as the population raster
        cellsites_gdf = gpd.GeoDataFrame(
            geometry=gpd.points_from_xy(
                x=cellsites.lon,
                y=cellsites.lat),
            crs="4326",
            data=cellsites)
        cellsites_gdf = cellsites_gdf.to_crs(self.population["crs"])

        # Drop duplicated ict_ids rows from cellsites to avoid redundancy
        # during Voronoi polygons generation
        cellsites_gdf = cellsites_gdf.drop(
            cellsites_gdf.loc[cellsites_gdf.ict_id.duplicated(), :].index)

        # Create buffers for different radii around the cell sites
        # This loop adds columns corresponding to the buffer areas and rings
        buffer_cellsites = cellsites_gdf.copy().reset_index()
        for radius in range(self.min_radius, self.max_radius + 1, self.radius_step):
            # convert radius from meters to degrees
            radius_in_degrees = meters_to_degrees_latitude(
                radius, central_latitude)
            # create buffer columns in the geodataframe and insert buffer
            # geometries there
            buffer_cellsites[f'buffer_{radius}'] = buffer_cellsites['geometry'].apply(
                lambda point: point.buffer(radius_in_degrees))
            # create population center distances columns in the geodataframe
            # and insert distances values there
            buffer_cellsites[f'popcd_{radius}'] = int(radius - self.radius_step / 2)
            # create ring columns in the geodataframe and insert ring
            # geometries there
            if radius == self.min_radius:
                # first ring is the same as the first buffer area
                buffer_cellsites[f'ring_{radius}'] = buffer_cellsites[f'buffer_{radius}']
            else:
                # ring is the difference between current buffer and the
                # previous buffer
                buffer_cellsites[f'ring_{radius}'] = buffer_cellsites[f'buffer_{radius}'].difference(
                    buffer_cellsites[f'buffer_{radius - self.radius_step}'])

        # Create Voronoi cells, this adds a new column with the geometry
        # corresponding to the Voronoi polygons
        buffer_cellsites = create_voronoi_cells(buffer_cellsites, self.area)

        # Check capacity for each cell site
        for radius in range(self.min_radius, self.max_radius + 1, self.radius_step):
            # Clip ring areas with Voronoi polygons to identify cell sites
            # service areas within each ring.
            buffer_cellsites[f'clring_{radius}'] = buffer_cellsites.apply(
                lambda row: row[f'ring_{radius}'].intersection(row['voronoi_polygons']), axis=1)
            # Calculate population count within clipped ring areas.
            buffer_cellsites[f'pop_clring_{radius}'] = buffer_cellsites[f'clring_{radius}'].apply(
                lambda x: get_population_sum(x, f"{self.input_data_path}/{self.data_files['pop']}"))
            # Calculate avubrnonbh, the average user bitrate in non-busy hour in kbps, from country-level statistics
            avubrnonbh = self.avubrnonbh(self.udatavmonth_pu)
            # Calculate user population bitrate in kbps within clipped ring areas.
            buffer_cellsites[f'upopbr_{radius}'] = buffer_cellsites[f'pop_clring_{radius}'].apply(
                lambda row: self.upopbr(avubrnonbh, row))
            # Calculate bitrate per resource block at population center distance in kbps.
            buffer_cellsites[f'brrbpopcd_{radius}'] = buffer_cellsites[f'popcd_{radius}'].apply(
                lambda row: self.brrbpopcd(row)
            )
            # Calculate user population resource blocks utilisation within clipped ring areas.
            buffer_cellsites[f'upoprbu_{radius}'] = buffer_cellsites.apply(
                lambda row: self.upoprbu(row[f'upopbr_{radius}'], row[f'brrbpopcd_{radius}']), axis=1
            )
        # Calculate total number of cell site required resource blocks
        upoprbu_columns = [col for col in buffer_cellsites.columns if col.startswith('upoprbu_')]
        buffer_cellsites['upoprbu_total'] = buffer_cellsites[upoprbu_columns].apply(lambda row: [sum(x) for x in zip(*row)], axis=1)

        # Calculate cell site available capacity
        buffer_cellsites['cellavcap'] = buffer_cellsites['upoprbu_total'].apply(
            lambda row: self.cellavcap(self.avrbpdsch, row))

        # Store the buffer analysis results
        self.buffer_cellsites_result = buffer_cellsites.set_index('ict_id').drop(columns='index')
        poi_data_merged = _poi_sufcapch(self.visibility, self.buffer_cellsites_result)
        poi_data_merged['rbdlthtarg'] = poi_data_merged['ground_distance'].apply(
            lambda row: self.poiddatareq(row)
        )
        poi_data_merged['sufcapch'] = poi_data_merged.apply(
            lambda row: self.sufcapch(row['cellavcap'], row['rbdlthtarg'])[0], axis=1
        )
        self.poi_sufcapch_result = poi_data_merged
        return self.buffer_cellsites_result, self.poi_sufcapch_result

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
