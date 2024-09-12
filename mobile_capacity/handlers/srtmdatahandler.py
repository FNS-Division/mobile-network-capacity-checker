import geopandas as gpd
from urllib import request
import requests
import logging
from srtm.height_map_collection import Srtm1HeightMapCollection
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import os
from mobile_capacity.utils import initialize_logger


class SRTMDataHandler:
    """
    A class to handle SRTM (Shuttle Radar Topography Mission) data retrieval and processing.

    Parameters:
    - srtm_directory (str): The directory where the SRTM data is stored or will be stored.
    - country_code (str): 3-Letter ISO code of the folder containing SRTM data, None will search all folders
    - logger (logging.Logger): Logger instance for logging messages.

    Attributes:
    - srtm_directory (str): The directory where the SRTM data is stored or will be stored.
    - srtm_dict (geopandas.GeoDataFrame): GeoDataFrame containing SRTM dictionary information.
    - logger (logging.Logger): Logger instance for logging messages.

    Methods:
    - get_srtm_dict(dict_url): Retrieves SRTM dictionary information as a GeoDataFrame.
    - check_directory(): Checks if the specified SRTM directory exists and creates it if not.
    - datafile_check(data_files): Checks if SRTM data files exist in the SRTM directory.
    - locate_data_files(gdf): Spatially joins the SRTM dictionary with a GeoDataFrame of locations.
    - download_data_files(data_files): Downloads SRTM data files from the specified URLs and saves them to the specified path.
    - parallel_download(files_to_download, max_workers): Downloads SRTM data files in parallel using ThreadPoolExecutor.
    - srtmheightmapcollection(): Returns an instance of Srtm1HeightMapCollection for SRTM data access.

    Example:
    ```python
    srtm_handler = SRTMDataHandler(srtm_directory='/path/to/srtm_data', earthdata_username='your_username', earthdata_password='your_password')
    srtm_files = ['N00E010.hgt', 'N00E011.hgt']
    files_to_download = srtm_handler.datafile_check(srtm_files)
    srtm_handler.parallel_download(files_to_download)
    ```
    """

    def __init__(self,
                 srtm_directory,
                 logger=None,
                 logs_dir=None,
                 country_code=None,
                 enable_logging=False,
                 ):

        self.srtm_directory = srtm_directory
        self.srtm_dict = self.get_srtm_dict()
        self.country_code = country_code
        self.logs_dir = logs_dir
        self.enable_logging = enable_logging

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
    def srtmheightmapcollection(self):
        # Returns an instance of Srtm1HeightMapCollection for SRTM data access.
        return Srtm1HeightMapCollection(auto_build_index=True, hgt_dir=Path(self.srtm_directory))

    def get_srtm_dict(self, dict_url='https://d35k53rhvc9u0d.cloudfront.net/elevation_data/srtm30m_bounding_boxes.json'):
        """
        Retrieve the SRTM dictionary file from the provided url.

        Args:
            dict_url (str): The url of the SRTM dictionary file.

        Returns:
            A GeoDataFrame containing the SRTM dictionary information.

        Raises:
            RuntimeError: If the SRTM dictionary cannot be read from the provided url.
        """
        # Create the request object with user agent header
        req = request.Request(dict_url)
        req.add_header('User-Agent', 'Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:77.0) Gecko/20100101 Firefox/77.0')
        content = request.urlopen(req)
        # Read the content if status code is 200
        if content.status == 200:
            return gpd.read_file(content)
        else:
            self._log("error", f'SRTM dictionary cannot be read from following url: {dict_url}')

    def check_directory(self):
        # Checks if the specified SRTM directory exists and creates it if not.
        if not os.path.exists(self.srtm_directory):
            self._log("info", 'Given srtm directory does not exist. Making a new directory with the given path.')
            os.makedirs(self.srtm_directory)

    def datafile_check(self, data_files: list):
        """
        Checks if SRTM datafiles corresponding to SRTM tiles exist in the SRTM directory.

        Args:
            data_files (list): List of SRTM data files.

        Returns:
            list: List of SRTM data files that need to be downloaded.
        """
        # Initialize an empty set to store file paths of SRTM files to download
        data_files_to_download = set()

        # Check if each file in srtm_tiles exists in the directory
        for file in data_files:
            file_path = os.path.join(self.srtm_directory, file)
            # If the file doesn't exist, add it to the set
            if not os.path.exists(file_path):
                data_files_to_download.add(file)

        self._log("info", f'{len(data_files_to_download)} data file(s) to download.')

        return list(data_files_to_download)

    def locate_data_files(self, gdf: gpd.GeoDataFrame):
        """
        Spatially join the SRTM dictionary with a GeoDataFrame of locations.

        Args:
            gdf (geopandas.GeoDataFrame): GeoDataFrame containing locations.

        Returns:
            tuple: A tuple of two sets - unique_data_files and unmatched_indices.
        """

        # Perform spatial join using 'intersects' predicate
        gdf_data_files = gdf.sjoin(self.srtm_dict, how='left', predicate='intersects')['dataFile']

        # Find the rows with null values for dataFile
        unmatched_indices = set(gdf_data_files[gdf_data_files.isnull()].index)

        matched_indices = set(gdf_data_files.index) - unmatched_indices

        selected_rows = gdf_data_files.loc[list(matched_indices)]

        unique_data_files = selected_rows.unique()

        return unique_data_files, unmatched_indices

    def download_data_files(self, data_files=None, country_code=None):
        """
        Downloads a SRTM tile from the specified URL and saves it to the specified path.

        Args:
            data_files (list): List of SRTM data files. If None, downloading entire country
            country_code (str):  3-Letter ISO code of the folder containing SRTM data, None will search all folders

        Returns:
            None
        """

        if data_files is None and country_code is None:
            raise ValueError("Either 'data_files' and/or 'country_code' must be set.")

        self.check_directory()
        self._log("info", 'Fetching from ITU public storage')

        base_folder_path = "https://d35k53rhvc9u0d.cloudfront.net/elevation_data/"

        country_code_path = country_code or self.country_code

        # Appending country code to the source bucked if specified
        if country_code_path:
            self._log("info", "Country code selected, filtering elevation data")
            base_folder_path += f"{country_code_path.upper()}/"

        response = requests.get(base_folder_path + 'index.json')
        if response.status_code == 200:
            data = response.json()
        else:
            raise Exception(f"Failed to retrieve data from {base_folder_path + 'index.json'}, status code: {response.status_code}")
        file_list = data['files']

        # If a file list was specified, filter blob list to only download specified files
        if data_files != None:  # noqa: E711
            self._log("info", "Filelist specified, downloading only selected files")
            file_list = [file for file in file_list if any([data_file in file for data_file in data_files])]

            # Removing duplicated files, in case they appear twice.
            # Only possible if we look at multiple countries
            if not country_code:
                unique_files_dict = {}
                for file in file_list:
                    filename = file.split('/')[-1]
                    if filename not in unique_files_dict:
                        unique_files_dict[filename] = file
                file_list = list(unique_files_dict.values())
        else:
            self._log("info", "No datafiles selected, downloading files for the entire country")

        # Iterate to download every files
        for file in file_list:
            download_file_path = os.path.join(self.srtm_directory, os.path.basename(file.split('/')[-1]))
            request.urlretrieve(base_folder_path + file, download_file_path)

        self._log("info", "Downloaded completed")

    def parallel_download(self, files_to_download, max_workers=4):
        """
        Downloads SRTM data files in parallel using ThreadPoolExecutor.

        Args:
            files_to_download (list): List of SRTM data files to download.
            max_workers (int): Maximum number of worker threads for parallel download.

        Returns:
            None
        """
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            executor.map(self.download_data_files, files_to_download)
