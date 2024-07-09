from mobile_capacity.spatial import meters_to_degrees_latitude, generate_voronoi_polygons
import logging
import numpy as np
import pandas as pd
import geopandas as gpd
import sys
import os
import rasterio
from datetime import datetime
from rasterstats import zonal_stats

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
            # Create a logger
            self.logger = logging.getLogger(__name__)
            # Set minimum logging level to DEBUG
            self.logger.setLevel(logging.DEBUG)

            # Define a console handler to output log messages to console
            ch = logging.StreamHandler()
            ch.setLevel(logging.DEBUG)  # Set console output level to DEBUG

            # Define a file handler to save log messages to a file
            log_filename = datetime.now().strftime('app_%Y-%m-%d_%H-%M-%S.log')
            true_root = os.path.dirname(os.path.abspath('.'))
            # Unique log file based on timestamp
            log_file = os.path.join(true_root, 'logs', log_filename)
            # Ensure the log directory exists
            os.makedirs(os.path.dirname(log_file), exist_ok=True)
            fh = logging.FileHandler(log_file)
            fh.setLevel(logging.DEBUG)  # Set

            # Define a formatter to specify the format of log messages
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            ch.setFormatter(formatter)
            fh.setFormatter(formatter)

            # Add the console handler to the logger
            self.logger.addHandler(ch)
            self.logger.addHandler(fh)

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
        self.max_radius = max_radius # maximum buffer radius
        self.min_radius = min_radius # maximum buffer radius
        self.radius_step = radius_step # maximum buffer radius

        # Constants
        self.minperhour = 60  # number of minutes per hour
        self.secpermin = 60  # number of seconds per minute
        self.bitsingbit = 1000000000  # bits in one gigabit
        self.bitsinkbit = 1000  # bits in kilobit
        self.bitsingbyte = 8589934592  # bits in one gigabyte

        # Intermediary outputs
        # RB number required to meet download throughput target in units
        self.rbdlthtarg_result = None
        # Bitrate per resource block at population center distance in kbps
        self.brrbpopcd_result = None
        self.avubrnonbh_result = None  # Average user bitrate in non-busy hour in kbps
        self.upopbr_result = None  # User Population Bitrate in kbps
        self.upoprbu_result = None  # User population resource blocks utilisation
        self.sufcapch_result = None  # Sufficient capacity check

        # Load file inputs
        self.bwdistance_km = self._load_file(
            f"{self.input_data_path}/{self.data_files['bwdistance_km_file_name']}", 'csv')
        self.bwdlachievbr = self._load_file(
            f"{self.input_data_path}/{self.data_files['bwdlachievbr_file_name']}", 'csv')
        self.cellsites = self._load_file(
            f"{self.input_data_path}/{self.data_files['cellsites_file']}", 'csv')
        self.mbbt = self._load_file(
            f"{self.input_data_path}/{self.data_files['mbbt_file']}", 'csv')
        self.visibility = self._load_file(
            f"{self.input_data_path}/{self.data_files['poi_visibility_file']}", 'csv')
        self.mbbsubscr = self._load_file(
            f"{self.input_data_path}/{self.data_files['mbbsubscr_file']}", 'csv')
        self.mbbtraffic = self._load_file(
            f"{self.input_data_path}/{self.data_files['mbbtraffic_file']}", 'csv')
        self.area = self._load_file(
            f"{self.input_data_path}/{self.data_files['area_file']}", 'gpkg')
        self.population = self._load_file(
            f"{self.input_data_path}/{self.data_files['pop_file']}", 'tif')

    def _log_info(self, message):
        if self.logger:
            self.logger.info(message)    
    
    @property
    def udatavmonth_pu(self):
        """
        Returns average mobile broadband user data traffic volume per month in GBs (Gigabytes) for the latest year in the ITU dataset.
        """
        return self.mbbt.loc[self.mbbt["entityName"] ==
                             self.country_name, "mbb_traffic_per_subscr_per_month"].item()
    
    @property
    def udatavmonth_year(self):
        """
        Returns average monthly mobile broadband user data traffic volume in gigabytes (GB).
        """
        return self.mbbt.loc[self.mbbt["entityName"] ==
                             self.country_name, "dataYear"].item()

    @property
    def nrb(self):
        return self.bw * self.rb_num_multiplier

    @property
    def avrbpdsch(self):
        return ((100 - self.cco) / self.nrb) * 100

    def _load_file(self, file_path, file_type):
        """
        Private method to load the file based on the file type.
        """
        try:
            if file_type == 'csv':
                data = pd.read_csv(file_path)
            elif file_type == 'json':
                data = pd.read_json(file_path)
            elif file_type == 'excel':
                data = pd.read_excel(file_path)
            elif file_type == 'gpkg':
                data = gpd.read_file(file_path)
            elif file_type == 'tif':
                with rasterio.open(file_path) as popdata:
                    # Read the first band of the raster data which represents
                    # population data
                    raster_data = popdata.read(1)
                    desired_crs = popdata.crs
                    # save the affine transformation of the raster data for
                    # geospatial analysis
                    affine = popdata.transform
                    data = {
                        'raster_data': raster_data,
                        'crs': desired_crs,
                        'affine': affine}
            else:
                raise ValueError(f"Unsupported file type: {file_type}")
            self._log_info(f"File loaded successfully from {file_path}")
            return data
        except FileNotFoundError:
            self._log_info(f"File not found: {file_path}")
        except Exception as e:
            self._log_info(f"An error occurred while loading the file: {e}")

    def clear_intermediary_outputs(self):
        """
        Clears intermediary output attributes of the Capacity class.
        """
        self.rbdlthtarg_result = None  # RB number required to meet download throughput target in units
        self.brrbpopcd_result = None  # User population resource blocks utilisation
        self.avubrnonbh_result = None  # Average user bitrate in non-busy hour in kbps
        self.upopbr_result = None  # User Population Bitrate in kbps
        self.upoprbu_result = None  # User population resource blocks utilisation
        self.sufcapch_result = None  # Sufficient capacity check

    def poiddatareq(self, d):
        """
        Resource blocks number required to meet download throughput target.

        Parameters:
        - avrbpdsch (float): Avg. number of RB available for PDSCH in units.
        - d (int): Distance from tower in meters.
        - max_radius (int): 

        Returns:
        - rbdlthtarg (float): RB number required to meet download throughput target in units.

        Note:
        """

        if d > self.max_radius:
            rbdlthtarg = np.inf
            match_result = None
            index_result = None
        else:
            # Read in bwdistance_km
            bwdistance_km = self.bwdistance_km.copy()

            # Read in bwdlachievbr
            bwdlachievbr = self.bwdlachievbr.copy()

            # Attempt to find the match_result
            match_result = bwdistance_km.loc[bwdistance_km[f'{self.bw}MHz'] > d/1000].index.min()

            if pd.isna(match_result) or (match_result < 0):
                # Raises an error if no match is found
                match_result = None
                index_result = None
                rbdlthtarg = None
                print ("No matching index found (poiddatareq).")
            else:
                # Attempt to retrieve index_result if match_result is found
                index_result = bwdlachievbr.loc[match_result, f'{self.bw}MHz']
                rbdlthtarg = self.dlthtarg * 1024 / (index_result / self.avrbpdsch)

        self._log_info(f'match_result = {match_result}')
        self._log_info(f'index_result = {index_result}')
        self._log_info(f'rbdlthtarg = {rbdlthtarg}')

        # Store the result
        self.rbdlthtarg_result = rbdlthtarg
        return rbdlthtarg

    def brrbpopcd(self, popcd):
        """
        Bitrate per resource block at population center distance

        Parameters:
        - bwdistance_km: Distance samples for channels with different bandwidth in km.
        - bwdlachievbr (float): Achievable downlink bitrate for channels with different bandwidth in kbps.
        - popcd (int): Population center distance in meters.
        - avrbpdsch (float): Avg. number of RB available for PDSCH in units.

        Returns:
        - brrbpopcd (float): Bitrate per resource block at population center distance in kbps.

        Note:
        """
        # Read in bwdistance_km
        bwdistance_km = self.bwdistance_km.copy()

        # Read in bwdlachievbr
        bwdlachievbr = self.bwdlachievbr.copy()

        try:
            # Attempt to find the match_result
            match_result = bwdistance_km.loc[bwdistance_km[f'{self.bw}MHz']
                                             > popcd / 1000].index.min()
            if pd.isna(match_result):
                # Raises an error if no match is found
                raise ValueError("No matching index found (brrbpopcd).")
            # Attempt to retrieve index_result if match_result is found
            index_result = bwdlachievbr.loc[match_result, f'{self.bw}MHz']
            brrbpopcd = index_result / self.avrbpdsch
            self._log_info(f'match_result = {match_result}')
        except ValueError as e:
            # Handle the case where no matching index is found
            self._log_info(e)
            # Optionally, set index_result to None or another default value
            index_result = None
            brrbpopcd = None
        except Exception as e:
            # Handle other potential exceptions (e.g., wrong column name)
            self._log_info(f"An error occurred (brrbpopcd): {e}")
            # Optionally, set index_result to None or another default value
            index_result = None
            brrbpopcd = None

            self._log_info(f'index_result = {index_result}')
            self._log_info(f'brrbpopcd = {brrbpopcd}')

        # Store the result
        self.brrbpopcd_result = brrbpopcd
        return brrbpopcd

    def avubrnonbh(self, udatavmonth):
        """
        Average user bitrate in non-busy hour

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
                self.nonbhu /
                100) /
               self.minperhour) /
              self.secpermin) *
             self.bitsingbyte) /
            self.bitsinkbit)

        self._log_info(f'avubrnonbh = {avubrnonbh}')

        # Store the result
        self.avubrnonbh_result = avubrnonbh
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
        upopbr = avubrnonbh*pop*self.oppopshare/100/self.fb_per_site
        self._log_info(f'upopbr = {upopbr}')

        # Store the result
        self.upopbr_result = upopbr
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
        # Calcualte user population resource blocks utilisation in units.
        upoprbu = upopbr / brrbpopcd

        self._log_info(f'upoprbu = {upoprbu}')

        self.upoprbu_result = upoprbu

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

        # Cell site available capacity.
        cellavcap = avrbpdsch - upoprbu

        self._log_info(f'avrbpdsch = {avrbpdsch}')
        self._log_info(f'cellavcap = {cellavcap}')

        # Store the result
        self.cellavcap_result = cellavcap

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

        # Identify if capacity requirement is satisfied.
        if (cellavcap > rbdlthtarg):
            sufcapch = True
        else:
            sufcapch = False

        self._log_info(f'sufcapch = {sufcapch}')

        # Store the result
        self.sufcapch_result = sufcapch
        return sufcapch

    def capacity_checker(self, d, popcd, udatavmonth, pop):
        """
        Calculates capacity check based on given parameters.

        Parameters:
        - d (float): Distance to POI for data rate calculation.
        - popcd (float): Population center distance parameter.
        - udatavmonth (float): User data volume per month parameter.
        - pop (float): Population parameter.

        Returns:
        - float: Capacity check result based on the calculations.

        Clears intermediary outputs, calculates independent functions including:
        - `rbdlthtarg`: Data rate target based on `d`.
        - `brrpopcd`: Bitrate per resource block based on `popcd`.
        - `avubrnonbh`: Average user bitrate based on `udatavmonth`.

        Then calculates dependent functions:
        - `upopbr`: User population bitrate based on `pop` and `avubrnonbh`.
        - `upoprbu`: User population resource block utilization based on `upopbr` and `brrpopcd`.
        - `capcheck`: Sufficiency capacity check based on `avrbpdsch`, `upoprbu`, and `rbdlthtarg`.

        Stores the calculated capacity check result in `self.capcheck_result`.

        Note:
        - Ensure `self.clear_intermediary_outputs()` has been called to reset any previous calculations.
        """
        # Clear intermediary outputs from attributes
        self.clear_intermediary_outputs()

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
        return upopbr, upoprbu, cellavcap, capcheck

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
        
        def get_population_sum(geometry):
            """
            Calculates the population sum within a given geometry using a population density raster.

            Parameters:
            - geometry (shapely.geometry.Polygon): The geometry within which to calculate the population sum.

            Returns:
            - float: The population sum within the given geometry, or None if an error occurs.
            """
            try:
                population_raster_location = f"{self.input_data_path}/{self.data_files['pop_file']}"
                stats = zonal_stats(
                    geometry,
                    population_raster_location,
                    stats="sum")
                return stats[0]['sum']
            except ValueError as ve:
                print(f"ValueError occurred (get_population_sum): {ve}")
                return None
            except Exception as e:
                print(f"An error occurred (get_population_sum): {e}")
                return None

        # Copy input data
        poi_data = self.visibility.copy()
        cellsites = self.cellsites.copy()
        # for distance conversion from meters to degrees
        central_latitude = cellsites['lat'].mean()

        # Convert cell sites to a GeoDataFrame - assuming the input data used the EPSG 4326 CRS
        # Re-project to the same CRS as the population raster
        cellsites_gdf = gpd.GeoDataFrame(
            geometry=gpd.points_from_xy(
                x=cellsites.lon,
                y=cellsites.lat,
                crs="4326"),
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
                get_population_sum)
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
        upoprbu_columns = buffer_cellsites.filter(regex=r'^upoprbu_')
        buffer_cellsites['upoprbu_total'] = upoprbu_columns.sum(axis=1)

        # Calculate cell site available capacity
        buffer_cellsites[f'cellavcap'] = buffer_cellsites[f'upoprbu_total'].apply(
            lambda row: self.cellavcap(self.avrbpdsch, row))
            
        # Store the buffer analysis results
        self.buffer_cellsites_result = buffer_cellsites.set_index('ict_id').drop(columns = 'index')
        self.poi_sufcapch_result = self.poi_sufcapch(poi_data, self.buffer_cellsites_result)
        return self.buffer_cellsites_result, self.poi_sufcapch_result
    
    def poi_sufcapch(self, poi_data, buffer_cellsites_result):
        poi_data_merged = poi_data[['poi_id','is_visible','ground_distance_1','cellsite_1']].merge(buffer_cellsites_result, 
                                     left_on = 'cellsite_1', 
                                     right_on = 'ict_id', 
                                     how = 'left').drop(columns="cellsite_1").set_index('poi_id')
        
        poi_data_merged['rbdlthtarg'] = poi_data_merged['ground_distance_1'].apply(
            lambda row: self.poiddatareq(row)
        )

        poi_data_merged['sufcapch'] = poi_data_merged.apply(
            lambda row: self.sufcapch(row['cellavcap'], row['rbdlthtarg']), axis=1
        )

        return poi_data_merged

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
            1024**3/df['dataValue_mbbsubscr']/12

        # Select relevant columns
        df = df.loc[:, ['entityName', 'entityIso_mbbsubscr', 'dataValue_mbbsubscr',
                        'dataValue_mbbtraffic', 'mbb_traffic_per_subscr_per_month', 'dataYear']]

        # Store the result
        self.mbbtps_result = df
        return df
