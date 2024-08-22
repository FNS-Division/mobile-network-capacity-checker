import os
import logging
from datetime import datetime
import uuid


def initialize_logger(logs_dir):
    """
    Initializes and returns a logger with both console and file handlers.

    Returns:
        logging.Logger: Configured logger object.
    """
    # Use the root logger
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)

    # Define a console handler
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)

    # Define a file handler
    log_filename = datetime.now().strftime('app_%Y-%m-%d_%H-%M-%S.log')
    log_file = os.path.join(logs_dir, log_filename)
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    fh = logging.FileHandler(log_file)
    fh.setLevel(logging.DEBUG)

    # Define a formatter
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    ch.setFormatter(formatter)
    fh.setFormatter(formatter)

    # Add the handlers to the logger
    logger.addHandler(ch)
    logger.addHandler(fh)

    return logger


def generate_uuid(size=1):
    """
    Generate UUIDs.

    Parameters:
    - size (int): Number of UUIDs to generate. Default is 1 for a single UUID.

    Returns:
    - str or List[str]: Single UUID or list of UUIDs.
    """

    if size > 1:
        return [str(uuid.uuid4()) for _ in range(size)]
    else:
        return str(uuid.uuid4())
