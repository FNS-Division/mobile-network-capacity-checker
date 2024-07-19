import os
import logging
from datetime import datetime
import pandas as pd
import geopandas as gpd
import rasterio


def initialize_logger(name):
    """
    Initializes and returns a logger with both console and file handlers.

    Args:
        name (str): The name to be used for the logger.

    Returns:
        logging.Logger: Configured logger object.
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)

    # Define a console handler
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)

    # Define a file handler
    log_filename = datetime.now().strftime('app_%Y-%m-%d_%H-%M-%S.log')
    true_root = os.path.dirname(os.path.abspath('.'))
    log_file = os.path.join(true_root, 'logs', log_filename)
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    fh = logging.FileHandler(log_file)
    fh.setLevel(logging.DEBUG)

    # Define a formatter
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    ch.setFormatter(formatter)
    fh.setFormatter(formatter)

    # Add the handlers to the logger
    logger.addHandler(ch)
    logger.addHandler(fh)

    return logger


def load_data(root_dir, data_files, logger=None):
    """
    Load all required data files.

    Args:
        root_dir (str): Root directory path.
        data_files (dict): Dictionary containing file names for different data types.
        logger (logging.Logger, optional): Logger object for logging information and errors. Defaults to None.

    Returns:
        dict: Dictionary containing loaded data for each file type.
    """
    input_data_path = os.path.join(root_dir, 'data', 'input_data')
    loaded_data = {}

    # Load CSV files
    csv_files = ['bwdistance_km', 'bwdlachievbr', 'cellsites', 'mbbt', 'poi', 'poi_visibility', 'mbbsubscr', 'mbbtraffic']
    for file_key in csv_files:
        file_path = os.path.join(input_data_path, data_files[file_key])
        try:
            loaded_data[file_key] = pd.read_csv(file_path)
            if logger:
                logger.info(f"Loaded {file_key} file from {file_path}")
        except Exception as e:
            if logger:
                logger.error(f"Error loading {file_key} file from {file_path}: {e}")

    # Load GeoPackage file
    area_file_path = os.path.join(input_data_path, data_files['area'])
    try:
        loaded_data['area'] = gpd.read_file(area_file_path)
        if loaded_data['area'].crs is None:
            loaded_data['area'].crs = '4326'
        if logger:
            logger.info(f"Loaded area file from {area_file_path}")
    except Exception as e:
        if logger:
            logger.error(f"Error loading area file from {area_file_path}: {e}")

    # Load TIFF file
    pop_file_path = os.path.join(input_data_path, data_files['pop'])
    raster_data = {}
    try:
        with rasterio.open(pop_file_path) as src:
            raster_data['raster_data'] = src.read(1)
            raster_data['crs'] = src.crs
            raster_data['affine'] = src.transform
        loaded_data['population'] = raster_data
        if logger:
            logger.info(f"Loaded population raster file from {pop_file_path}")
    except Exception as e:
        if logger:
            logger.error(f"Error loading population raster file from {pop_file_path}: {e}")

    return loaded_data
