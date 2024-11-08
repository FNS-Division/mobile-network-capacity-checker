import pandas as pd
from urllib import request
import logging
import os
from mobile_capacity.utils import initialize_logger
from mobile_capacity.spatial import get_tif_xsize, process_tif


class PopulationDataHandler:
    """
    A class to handle population data retrieval and processing.

    Parameters:
    - data_dir (str): The directory where the population data is stored or will be stored.
    - country_code (str): The ISO3 country code for the desired country.
    - dataset_year (int): The year of the dataset.
    - one_km_res (bool): If True, use one-kilometer resolution dataset.
    - un_adjusted (bool): If True, use unadjusted dataset.
    - worldpop_base_url (str): The base URL for WorldPop data.
    - logger (logging.Logger): Logger instance for logging messages.

    Attributes:
    - data_dir (str): The directory where the population data is stored or will be stored.
    - country_code (str): The ISO3 country code for the desired country.
    - dataset_year (int): The year of the dataset.
    - one_km_res (bool): If True, use one-kilometer resolution dataset.
    - un_adjusted (bool): If True, use unadjusted dataset.
    - worldpop_base_url (str): The base URL for WorldPop data.
    - data_processor (DataProcessor): An instance of the DataProcessor class.
    - logger (logging.Logger): Logger instance for logging messages.

    Methods:
    - check_directory(): Checks if the specified data directory exists.
    - get_worldpop_dict(): Retrieves WorldPop dataset information as a DataFrame.
    - get_dataset_url(): Constructs the URL for the WorldPop dataset based on provided parameters.
    - download_tif_file(): Downloads the WorldPop dataset if it does not exist in the specified directory.
    - get_dataset_xsize(): Retrieves the xsize (width) of the WorldPop dataset.
    - generate_population_data(): Processes the WorldPop dataset and returns population data as a DataFrame.

    Example:
    ```python
    data_handler = PopulationDataHandler(data_dir='/path/to/data', country_code='USA', dataset_year=2020)
    population_data = data_handler.population_data
    ```
    """

    def __init__(self,
                 data_dir,
                 country_code: str,
                 dataset_year: int,
                 one_km_res: bool = False,
                 un_adjusted: bool = True,
                 worldpop_base_url='https://zstagigaprodeuw1.blob.core.windows.net/gigainframapkit-public-container/worldpop_data/',
                 logger=None,
                 logs_dir=None,
                 enable_logging=True
                 ):

        self.data_dir = data_dir
        self._country_code = None
        self._dataset_year = None
        self.one_km_res = one_km_res
        self.un_adjusted = un_adjusted
        self.worldpop_base_url = worldpop_base_url
        self.logs_dir = logs_dir
        self.enable_logging = enable_logging
        self._worldpop_dict = None
        self._dataset_url = None
        self._dataset_name = None
        self._dataset_path = None
        self._dataset_xsize = None
        self._population_data = None
        self.country_code = country_code.upper()
        self.dataset_year = dataset_year
        self.check_directory()

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

    def check_directory(self):
        # Checks if the specified data directory exists.
        if not os.path.exists(self.data_dir):
            os.makedirs(self.data_dir)

    @property
    def worldpop_dict(self):
        if self._worldpop_dict is None:
            self._worldpop_dict = self.get_worldpop_dict()
        return self._worldpop_dict

    @worldpop_dict.setter
    def worldpop_dict(self, value):
        self._worldpop_dict = value

    @property
    def country_code(self):
        return self._country_code

    @country_code.setter
    def country_code(self, value):
        if self._is_valid_country_code(value):
            self._country_code = value
        else:
            self._log('error', 'Country code does not exist in the worldpop database!')

    @property
    def dataset_year(self):
        return self._dataset_year

    @dataset_year.setter
    def dataset_year(self, value):
        if self._is_valid_dataset_year(value):
            self._dataset_year = value
        else:
            self._log('error', 'Worldpop dataset for given year does not exist!')

    @property
    def dataset_url(self):
        if self._dataset_url is None:
            self._dataset_url = self.get_dataset_url()
        return self._dataset_url

    @dataset_url.setter
    def dataset_url(self, value):
        self._dataset_url = value

    @property
    def dataset_name(self):
        if self._dataset_name is None:
            self._dataset_name = self.dataset_url.split('/')[-1]
        return self._dataset_name

    @dataset_name.setter
    def dataset_name(self, value):
        self._dataset_name = value

    @property
    def dataset_path(self):
        if self._dataset_path is None:
            self._dataset_path = os.path.join(self.data_dir, self.dataset_name)
        return self._dataset_path

    @dataset_path.setter
    def dataset_path(self, value):
        self._dataset_path = value

    @property
    def dataset_xsize(self):
        if self._dataset_xsize is None:
            self._dataset_xsize = self.get_dataset_xsize()
        return self._dataset_xsize

    @dataset_xsize.setter
    def dataset_xsize(self, value):
        self._dataset_xsize = value

    @property
    def population_data(self):
        if self._population_data is None:
            self._population_data = self.generate_population_data()
        return self._population_data

    @population_data.setter
    def population_data(self, value):
        self._population_data = value

    def _is_valid_country_code(self, value):
        return value.upper() in self.worldpop_dict.ISO3.tolist()

    def _is_valid_dataset_year(self, value):
        return sum(self.worldpop_dict.Covariate.str.contains(str(value))) > 0

    def get_worldpop_dict(self):
        """
        Retrieves WorldPop dataset information as a DataFrame.

        Returns:
        pandas.DataFrame: DataFrame containing WorldPop dataset information.
        """
        return pd.read_csv(self.worldpop_base_url + 'assets/wpgpDatasets.csv')

    def get_dataset_url(self):
        """
        Constructs the URL for the WorldPop dataset based on provided parameters.

        Returns:
        str: URL for the WorldPop dataset.
        """
        dataset_url = self.worldpop_base_url
        dataset_url += self.worldpop_dict[(self.worldpop_dict.ISO3 == self.country_code)
                                          & (self.worldpop_dict.Covariate == 'ppp_' + str(self.dataset_year) + ('_UNadj' if self.un_adjusted else ''))].PathToRaster.values[0]
        if self.one_km_res:
            dataset_url = dataset_url.split('/')
            dataset_url[7] = dataset_url[7] + '_1km' + ('_UNadj' if self.un_adjusted else '')
            dataset_url[10] = dataset_url[10].replace(str(self.dataset_year), str(self.dataset_year) + '_1km_Aggregated')
            dataset_url = '/'.join(dataset_url)

        return dataset_url

    def download_tif_file(self):
        # Downloads the WorldPop dataset if it does not exist in the specified directory.
        if not os.path.exists(self.dataset_path):
            request.urlretrieve(self.dataset_url, self.dataset_path)
            self._log('info', 'Dataset download is complete!')

    def get_dataset_xsize(self):
        """
        Retrieves the xsize (width) of the WorldPop dataset.

        Returns:
        int: The xsize (width) of the WorldPop dataset.
        """
        if not os.path.exists(self.dataset_path):
            self._log('warn', 'Worldpop dataset does not exist in the directory. Downloading the tif file...')
            self.download_tif_file()

        xsize = get_tif_xsize(file_path=self.dataset_path)

        return xsize

    def generate_population_data(self):
        """
        Processes the WorldPop dataset and returns population data as a DataFrame.

        Returns:
        pandas.DataFrame: Population data DataFrame.
        """
        if not os.path.exists(self.dataset_path):
            self._log('warn', 'Worldpop dataset does not exist in the directory. Downloading the tif file...')
            self.download_tif_file()

        self._log('info', 'Processing population tif file...')
        df_population = process_tif(input_file=self.dataset_path, drop_nodata=True)
        df_population.rename(columns={'pixel_value': 'population'}, inplace=True)
        self._log('info', 'Population tif file is processed!')

        return df_population
