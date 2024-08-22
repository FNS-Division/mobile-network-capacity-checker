import os
import pytest
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(os.path.join(os.path.dirname(__file__), 'unit'))


@pytest.fixture
def init_variable_values():
    init_variable_values = {
        'data_files': {
            "cellsites": 'cell-sites.csv',  # Cell sites data
            "mbbt": "MobileBB_Traffic_per_Subscr_per_Month.csv",  # Mobile broadband traffic data
            "poi": "points-of-interest.csv",  # Point of interest data
            "poi_visibility": "visibility.csv",  # Visibility analysis output file
            "bwdistance_km": "_bwdistance_km.csv",  # Distance samples for channels with different bandwidth in km
            "bwdlachievbr": "_bwdlachievbr_kbps.csv",  # Achievable downlink bitrate for channels with different bandwidth in kbps.
            "pop": "population.tif",  # Population density raster file name
            "area": "area.gpkg",  # Area contour file name
            "mbbsubscr": "active-mobile-broadband-subscriptions.csv",  # Mobile broadband subscriptions data
            "mbbtraffic": "mobile-broadband-internet-traffic-within-the-country.csv"  # Mobile broadband traffic data
        },
        'root_dir': os.path.abspath(os.path.join(os.getcwd())),
        'country_name': "Spain",
        'bw': 20,  # Bandwidth, MHz
        'rb_num_multiplier': 5,
        'cco': 18,  # Control channel overheads in %
        'mbb_subscr': 113, # Active mobile-broadband subscriptions per 100 people
        'fb_per_site': 3,  # Number of frequency bands on site
        'angles_num': 360,  # PLACEHOLDER # Set the number of angles to be used for azimuth analysis
        'rotation_angle': 60,  # PLACEHOLDER # Define the rotation angle to create a sector +/-rotation_angle degrees clockwise and counter-clockwise
        'dlthtarg': 20,  # Download throughput target in Mbps.
        'oppopshare': 50,  # % of Population on Operator
        'min_radius': 1000,  # meters, minimum radius around cell site location for population calculation
        'max_radius': 2000,  # meters, maximum radius should be divisible by 1000; maximum radius around cell site location for population calculation
        'radius_step': 1000,  # meters, radius step size for population calculation
        'nbhours': 10,  # number of non-busy hours per day
        'nonbhu': 50,  # Connection usage in non-busy hour in %
        'enable_logging': False
    }
    return init_variable_values
