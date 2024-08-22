import pandas as pd
import geopandas as gp
import logging
import rasterio
import gzip
import random
import string
import shutil
import time
import os


class DataStorage:
    valid_dataset_subfolders = ['input', 'output']
    valid_dataset_categories = ['pointofinterest', 'cellsite', 'visibility', 'mbbt', 'mbbsubscr', 'mbbtraffic', 'bwdistance_km', 'bwdlachievbr']

    def __init__(self, data_dir):
        """
        Initialize the DataStorage object.

        Parameters:
        - data_dir (str): The path to the data directory.
        """

        if not os.path.isdir(data_dir):
            raise FileNotFoundError('Data directory does not exist.')

        self.data_dir = data_dir
        self._dataset_subfolder = None
        self._dataset_category = None

    @property
    def dataset_subfolder(self):
        """
        Get the current dataset subfolder.

        Returns:
        - str or None: The current dataset subfolder.
        """
        return self._dataset_subfolder

    @dataset_subfolder.setter
    def dataset_subfolder(self, value):
        """
        Set the dataset subfolder.

        Parameters:
        - value (str): The new dataset subfolder.

        Raises:
        - ValueError: If the provided value is not a valid dataset subfolder.
        """
        if self._is_valid_dataset_subfolder(value):
            self._dataset_subfolder = value
        else:
            raise ValueError(f'Invalid dataset subfolder: {value}')

    @property
    def dataset_category(self):
        """
        Get the current dataset category.

        Returns:
        - str or None: The current dataset category.
        """
        return self._dataset_category

    @dataset_category.setter
    def dataset_category(self, value):
        """
        Set the dataset category.

        Parameters:
        - value (str): The new dataset category.

        Raises:
        - ValueError: If the provided value is not a valid dataset category.
        """
        if self._is_valid_dataset_category(value):
            self._dataset_category = value
        else:
            raise ValueError(f'Invalid dataset subfolder: {value}')

    def list_datasets(self, file_extension=None):
        """
        List datasets in the data directory.

        Parameters:
        - file_extension (str): Filter datasets by file extension. Default is None.

        Returns:
        - List[str]: A list of dataset filenames in the data directory.
        """
        datasets = []

        for file in os.listdir(self.data_dir):
            if file_extension is None or file.endswith(file_extension):
                datasets.append(file)

        return datasets

    def read_dataset(self, filename, sheet_name=None):
        """
        Read a file from the data directory.

        Args:
        - file_name (str): The name of the file to read.
        - sheet_name (str, optional): The name of the sheet to read for Excel files.

        Returns:
        - pd.DataFrame, gp.GeoDataFrame, or dict: The data read from the file.
        """
        file_path = os.path.join(self.data_dir, filename)
        suffix = os.path.splitext(filename)[1]

        if not os.path.isfile(file_path):
            raise FileNotFoundError(f"Dataset '{filename}' not found in the data directory")

        if suffix == '.csv':
            return pd.read_csv(file_path)
        elif suffix == '.xlsx' or suffix == '.xls':
            return pd.read_excel(file_path, engine='openpyxl', sheet_name=sheet_name)
        elif suffix == '.shp' or suffix == '.zip':
            return gp.read_file(file_path)
        elif suffix == '.parquet':
            try:
                data = gp.read_parquet(file_path)
            except Exception:
                data = pd.read_parquet(file_path)
            return data
        elif suffix == '.json':
            return pd.read_json(file_path)
        elif suffix == '.geojson':
            return gp.read_file(file_path)
        elif suffix == '.gpkg':
            return gp.read_file(file_path)
        elif suffix in ['.tif', '.tiff']:
            raster_data = {}
            try:
                with rasterio.open(file_path) as src:
                    raster_data['raster_data'] = src.read(1)
                    raster_data['crs'] = src.crs
                    raster_data['affine'] = src.transform
                return raster_data
            except Exception as e:
                raise IOError(f"Error loading raster file from {file_path}: {e}")
        else:
            raise TypeError(f"Unsupported file type: {suffix}")

    def load_data(self, filename):
        """
        Load and convert data from a file into a list of dictionaries.

        Args:
            filename (str): The name of the file to load.

        Returns:
            List[dict]: A list of dictionaries representing the data records.
        """
        data = self.read_dataset(filename)

        # Convert the DataFrame into a list of dictionaries
        data_records = data.to_dict('records')

        return data_records

    def get_dataset_info(self, filename):
        """
        Get information about a dataset file.

        Args:
            filename (str): The name of the dataset file.

        Returns:
            dict: Information about the dataset, including file name, size, and timestamps.
        """
        file_path = os.path.join(self.data_dir, filename)

        if not os.path.isfile(file_path):
            raise FileNotFoundError(f"Dataset '{filename}' not found in the data directory")

        file_info = os.stat(file_path)
        file_size = file_info.st_size
        file_size_kb = round(file_size / 1024, 2)
        file_size_mb = round(file_size / (pow(1024, 2)), 2)
        created_time = time.ctime(file_info.st_ctime)
        modified_time = time.ctime(file_info.st_mtime)

        dataset_info = {
            'file_name': file_path.split('/')[-1],
            'file_size': file_size,
            'file_size_kb': file_size_kb,
            'file_size_mb': file_size_mb,
            'created_time': created_time,
            'modified_time': modified_time,
        }

        return dataset_info

    def rename_dataset(self, current_filename, new_filename):
        """
        Rename a dataset file.

        Args:
            current_filename (str): The current name of the dataset file.
            new_filename (str): The new name for the dataset file.

        Raises:
            FileNotFoundError: If the current dataset file is not found.
            FileExistsError: If a dataset with the new name already exists.

        Returns:
            None
        """
        current_file_path = os.path.join(self.data_dir, current_filename)

        if not os.path.isfile(current_file_path):
            raise FileNotFoundError(f"Dataset '{current_filename}' not found in the data directory")

        new_file_path = os.path.join(self.data_dir, new_filename)

        if os.path.isfile(new_file_path):
            raise FileExistsError(f"A dataset with the name '{new_filename}' already exists in the data directory")

        os.rename(current_file_path, new_file_path)
        logging.info(f"Dataset '{current_filename}' successfully renamed to '{new_filename}'")

    def delete_dataset(self, filename):
        """
        Delete a dataset file.

        Args:
            filename (str): The name of the dataset file to delete.

        Raises:
            FileNotFoundError: If the dataset file is not found.

        Returns:
            None
        """
        file_path = os.path.join(self.data_dir, filename)

        if not os.path.isfile(file_path):
            raise FileNotFoundError(f"Dataset '{filename}' not found in the data directory")

        os.remove(file_path)
        logging.info(f"Dataset '{filename}' successfully deleted from the data directory")

    def copy_dataset(self, current_filename, new_filename):
        """
        Copy a dataset file.

        Args:
            current_filename (str): The name of the current dataset file.
            new_filename (str): The name for the new copy of the dataset file.

        Raises:
            FileNotFoundError: If the current dataset file is not found.
            FileExistsError: If a dataset with the new name already exists.

        Returns:
            None
        """
        current_file_path = os.path.join(self.data_dir, current_filename)

        if not os.path.isfile(current_file_path):
            raise FileNotFoundError(f"Dataset '{current_filename}' not found in the data directory")

        new_file_path = os.path.join(self.data_dir, new_filename)

        if os.path.isfile(new_file_path):
            raise FileExistsError(f"A dataset with the name '{new_filename}' already exists in the data directory")

        shutil.copy2(current_file_path, new_file_path)
        logging.info(f"Dataset '{current_filename}' successfully copied to '{new_filename}'")

    def compress_data(self, filename):
        """
        Compress a dataset file using gzip.

        Args:
            filename (str): The name of the dataset file to compress.

        Raises:
            FileNotFoundError: If the dataset file is not found.
            FileExistsError: If a compressed dataset with the same name already exists.

        Returns:
            None
        """
        file_path = os.path.join(self.data_dir, filename)

        if not os.path.isfile(file_path):
            raise FileNotFoundError(f"Dataset '{filename}' not found in the data directory")

        compressed_filename = f"{filename}.gz"
        compressed_file_path = os.path.join(self.data_dir, compressed_filename)

        if os.path.isfile(compressed_file_path):
            raise FileExistsError(
                f"A compressed dataset with the name '{compressed_filename}' already exists in the data directory")

        with open(file_path, 'rb') as file_in:
            with gzip.open(compressed_file_path, 'wb') as file_out:
                shutil.copyfileobj(file_in, file_out)

        print(f"Dataset '{filename}' successfully compressed to '{compressed_filename}'")

    def decompress_data(self, compressed_filename):
        """
        Decompress a gzip-compressed dataset file.

        Args:
            compressed_filename (str): The name of the compressed dataset file.

        Raises:
            FileNotFoundError: If the compressed dataset file is not found.
            FileExistsError: If a decompressed dataset with the same name already exists.
            ValueError: If the compressed dataset file does not have a '.gz' extension.

        Returns:
            None
        """
        compressed_file_path = os.path.join(self.data_directory, compressed_filename)

        if not os.path.isfile(compressed_file_path):
            raise FileNotFoundError(f"Compressed dataset '{compressed_filename}' not found in the data directory")

        if not compressed_filename.endswith('.gz'):
            raise ValueError("Invalid compressed dataset format, expected '.gz' extension")

        decompressed_filename = compressed_filename[:-3]
        decompressed_file_path = os.path.join(self.data_directory, decompressed_filename)

        if os.path.isfile(decompressed_file_path):
            raise FileExistsError(
                f"A dataset with the name '{decompressed_filename}' already exists in the data directory")

        with gzip.open(compressed_file_path, 'rb') as file_in:
            with open(decompressed_file_path, 'wb') as file_out:
                shutil.copyfileobj(file_in, file_out)

        print(f"Compressed dataset '{compressed_filename}' successfully decompressed to '{decompressed_filename}'")

    def generate_dataset_id(self, prefix=''):
        """
        Generate a unique dataset ID based on the current timestamp and a random string.

        Args:
            prefix (str): An optional prefix to include in the dataset ID.

        Returns:
            str: The generated dataset ID.
        """
        timestamp = int(time.time())  # Get the current UNIX timestamp
        random_str = ''.join(random.choices(string.ascii_lowercase + string.digits, k=4))
        dataset_id = f"{prefix}-{timestamp}-{random_str}"
        return dataset_id

    def save_csv(self, data, filename, index=False):
        """
        Save a DataFrame or GeoDataFrame to a CSV file.

        Args:
            data (pd.DataFrame or gp.GeoDataFrame): The data to be saved.
            filename (str): The desired filename for the CSV file.
            index (bool): Whether to include the index in the CSV file.

        Raises:
            TypeError: If the data is not a DataFrame or GeoDataFrame.

        Returns:
            None
        """
        if not isinstance(data, (pd.DataFrame, gp.GeoDataFrame)):
            raise TypeError('Data must be DataFrame or GeoDataFrame to be saved in csv format!')

        filename = f'{filename}.csv'

        file_path = os.path.join(self.data_dir, filename)
        self.create_directory(file_path)
        data.to_csv(file_path, index=index)

    def save_parquet(self, data, filename, index=False):
        """
        Save a DataFrame or GeoDataFrame to a Parquet file.

        Args:
            data (pd.DataFrame or gp.GeoDataFrame): The data to be saved.
            filename (str): The desired filename for the Parquet file.
            index (bool): Whether to include the index in the Parquet file.

        Raises:
            TypeError: If the data is not a DataFrame or GeoDataFrame.

        Returns:
            None
        """
        if not isinstance(data, (pd.DataFrame, gp.GeoDataFrame)):
            raise TypeError('Data must be DataFrame or GeoDataFrame to be saved in csv format!')

        filename = f'{filename}.parquet'

        file_path = os.path.join(self.data_dir, filename)
        self.create_directory(file_path)
        data.to_parquet(file_path, index=index)

    def save_json(self, data, filename, orient='records'):
        """
        Save a DataFrame or GeoDataFrame to a JSON file.

        Args:
            data (pd.DataFrame or gp.GeoDataFrame): The data to be saved.
            filename (str): The desired filename for the JSON file.
            orient (str): The format in which to save the JSON. Default is 'records'.

        Raises:
            TypeError: If the data is not a DataFrame or GeoDataFrame.

        Returns:
            None
        """
        if not isinstance(data, (pd.DataFrame, gp.GeoDataFrame)):
            raise TypeError('Data must be DataFrame or GeoDataFrame to be saved in JSON format!')

        filename = f'{filename}.json'
        file_path = os.path.join(self.data_dir, filename)
        self.create_directory(file_path)
        data.to_json(file_path, orient=orient)

    def save_geopackage(self, data, filename):
        """
        Save a GeoDataFrame to a GeoPackage file.

        Args:
            data (gp.GeoDataFrame): The GeoDataFrame to be saved.
            filename (str): The desired filename for the GeoPackage file.

        Raises:
            TypeError: If the data is not a GeoDataFrame.

        Returns:
            None
        """
        if not isinstance(data, gp.GeoDataFrame):
            raise TypeError('Data must be a GeoDataFrame to be saved in GeoPackage format!')

        filename = f'{filename}.gpkg'

        file_path = os.path.join(self.data_dir, filename)
        self.create_directory(file_path)
        data.to_file(file_path, driver='GPKG')

    def _is_valid_dataset_subfolder(self, value):
        return value in self.valid_dataset_subfolders

    def _is_valid_dataset_category(self, value):
        return value in self.valid_dataset_categories

    def create_directory(self, file_path):
        directory = os.path.dirname(file_path)
        if not os.path.exists(directory):
            os.makedirs(directory)
