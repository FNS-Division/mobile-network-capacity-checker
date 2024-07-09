import os
import pytest
import pandas as pd
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(os.path.join(os.path.dirname(__file__), 'unit'))

@pytest.fixture
def init_variable_values():
    init_variable_values = {
        'data_files': {
                    "cellsites_file": 'ESP-1697916284-6wv8-cellsite.csv', # Cell sites data
                    "mbbt_file": "MobileBB_Traffic_per_Subscr_per_Month.csv", # Mobile broadband traffic data
                    "poi_visibility_file": "ESP-1708423221-tgah-visibility.csv", # Visibility analysis output file
                    "bwdistance_km_file_name": "bwdistance_km.csv", # Distance samples for channels with different bandwidth in km
                    "bwdlachievbr_file_name": "bwdlachievbr_kbps.csv", # Achievable downlink bitrate for channels with different bandwidth in kbps.
                    "pop_file": "population.tif", # Population density raster file name
                    "area_file" : "area.gpkg", # Area contour file name
                    "mbbsubscr_file" : "active-mobile-broadband-subscriptions_1711147050645.csv", # Mobile broadband subscriptions data
                    "mbbtraffic_file" : "mobile-broadband-internet-traffic-within-the-country_1711147118571.csv" # Mobile broadband traffic data
                    },
        ##### Directory #####
        'root_dir': os.path.abspath(os.path.join(os.getcwd())),

        ##### Area-Specific Files ##### 
        'country_name': "Spain",

        ##### Calculation Parameters #####
        ### Network Configuration ###
        'bw': 20, # Bandwidth, MHz
        'rb_num_multiplier': 5, 
        'cco': 18, # Control channel overheads in %
        # 'cells_per_site': 3, # Number of cells per site
        'fb_per_site': 3, # Number of frequency bands on site
        'angles_num': 360, # PLACEHOLDER # Set the number of angles to be used for azimuth analysis  
        'rotation_angle': 60, # PLACEHOLDER # Define the rotation angle to create a sector +/-rotation_angle degrees clockwise and counter-clockwise  

        ### POI configuration ###
        # 'd': 10000, # Distance from tower in meters
        'dlthtarg': 20, # Download throughput target in Mbps.

        ### Population information ###
        # 'pop': 10000, # population count, people
        # 'popcd': 5000, # population center distance, meters
        'oppopshare': 50, # % of Population on Operator
        'min_radius': 1000, # meters, minimum radius around cell site location for population calculation
        'max_radius': 2000,  # meters, maximum radius should be divisible by 1000; maximum radius around cell site location for population calculation
        'radius_step': 1000, # meters, radius step size for population calculation

        ### Avg user traffic profile ###
        'nbhours': 10, # number of non-busy hours per day
        'nonbhu': 50, # Connection usage in non-busy hour in % 
        # 'udatavmonth': 5, # User Data Volume per Month in GB
        'enable_logging': False
    }
    return init_variable_values