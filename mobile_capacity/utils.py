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
    logger.setLevel(logging.INFO)

    # Define a console handler
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)

    # Define a file handler
    log_filename = datetime.now().strftime('app_%Y-%m-%d_%H-%M-%S.log')
    log_file = os.path.join(logs_dir, log_filename)
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    fh = logging.FileHandler(log_file)
    fh.setLevel(logging.INFO)

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


def log_progress_bar(logger, iteration, total, prefix='', suffix='', decimals=1, length=50, fill='█'):
    """
    Log a progress bar using the provided logger, or do nothing if logger is None
    @params:
        logger      - Required  : logging.Logger instance or None
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
    """
    if logger is None:
        return  # Do nothing if logger is None

    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filled_length = int(length * iteration // total)
    bar = fill * filled_length + '-' * (length - filled_length)
    log_message = f'{prefix} |{bar}| {percent}% {suffix}'
    
    # Use carriage return to overwrite the line in console output
    if iteration < total:
        logger.info(log_message + '\r')
    else:
        logger.info(log_message)  # Log the final state without carriage return