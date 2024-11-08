import os
import pytest
import sys
import pandas as pd
from mobile_capacity.entities.pointofinterest import PointOfInterestCollection
from mobile_capacity.entities.cellsite import CellSiteCollection
from mobile_capacity.entities.visibilitypair import VisibilityPairCollection
from mobile_capacity.capacity import Capacity
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(os.path.join(os.path.dirname(__file__), 'unit'))


@pytest.fixture
def poi():
    pointsofinterest_filepath = "https://zstagigaprodeuw1.blob.core.windows.net/gigainframapkit-public-container/mobile_capacity_data/test_data/points-of-interest.csv"
    pointsofinterest = pd.read_csv(pointsofinterest_filepath).to_dict('records')
    return PointOfInterestCollection(poi_records=pointsofinterest)


@pytest.fixture
def cellsites():
    cellsites_filepath = "https://zstagigaprodeuw1.blob.core.windows.net/gigainframapkit-public-container/mobile_capacity_data/test_data/cell-sites.csv"
    cellsites = pd.read_csv(cellsites_filepath).to_dict('records')
    return CellSiteCollection(cellsite_records=cellsites)


@pytest.fixture
def visibilitypairs():
    visibilitypairs_filepath = "https://zstagigaprodeuw1.blob.core.windows.net/gigainframapkit-public-container/mobile_capacity_data/test_data/visibility.csv"
    visibilitypairs = pd.read_csv(visibilitypairs_filepath).to_dict('records')
    return VisibilityPairCollection(pair_records=visibilitypairs)


@pytest.fixture
def init_variable_values():
    init_variable_values = {
        # File and Directory Configuration
        'data_dir': os.path.join(os.path.dirname(__file__), 'data'),
        'logs_dir': os.path.join(os.path.dirname(__file__), 'logs'),
        'country_code': "ESP",
        'enable_logging': False,
        'use_secure_files': False,

        # Network Configuration
        'bw_L850': 5,  # MHz on L700 to L900 spectrum bandwidth
        'bw_L1800': 10,  # MHz on L1800 spectrum bandwidth
        'bw_L2600': 20,  # MHz on L2600 spectrum bandwidth
        'rb_num_multiplier': 5,
        'cco': 18,  # Control channel overheads in %
        'mbb_subscr': 113,  # Active mobile-broadband subscriptions per 100 people
        'sectors_per_site': 3,  # Number of sectors on site
        'angles_num': 360,  # Set the number of angles to be used for azimuth analysis
        'rotation_angle': 60,  # Define the rotation angle to create a sector +/-rotation_angle degrees clockwise and counter-clockwise

        # POI configuration
        'dlthtarg': 20,  # Download throughput target in Mbps.

        # Population information
        'oppopshare': 50,  # % of Population on Operator
        'dataset_year': 2020,  # Year of the population dataset
        'one_km_res': True,  # Use 1km resolution population data
        'un_adjusted': True,  # Use adjusted population data

        # Mobile coverage radius
        'min_radius': 1000,  # meters, minimum radius around cell site location for population calculation
        'max_radius': 2000,  # meters, maximum radius should be divisible by 1000; maximum radius around cell site location for population calculation
        'radius_step': 1000,  # meters, radius step size for population calculation

        # Avg user traffic profile
        'nbhours': 10,  # number of non-busy hours per day
        'nonbhu': 50,  # Connection usage in non-busy hour in %
    }
    return init_variable_values


@pytest.fixture
def mobilecapacity(poi, cellsites, visibilitypairs, init_variable_values):
    mobilecapacity = Capacity(
        poi=poi,
        cellsites=cellsites,
        visibility=visibilitypairs,
        **init_variable_values)
    return mobilecapacity
