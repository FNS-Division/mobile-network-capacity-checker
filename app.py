import streamlit as st
import os
from mobile_capacity.capacity import Capacity

# Set up the Streamlit interface
st.title("Mobile capacity checker")

# File paths and configurations
country_name = "Spain"
data_files = {
    "cellsites_file": 'ESP-1697916284-6wv8-cellsite.csv',
    "mbbt_file": "MobileBB_Traffic_per_Subscr_per_Month.csv",
    "poi_visibility_file": "ESP-1708423221-tgah-visibility.csv",
    "bwdistance_km_file_name": "bwdistance_km.csv",
    "bwdlachievbr_file_name": "bwdlachievbr_kbps.csv",
    "pop_file": "population.tif",
    "area_file": "area.gpkg",
    "mbbsubscr_file": "active-mobile-broadband-subscriptions_1711147050645.csv",
    "mbbtraffic_file": "mobile-broadband-internet-traffic-within-the-country_1711147118571.csv"
}

# Network Configuration
bw = 20  # Bandwidth, MHz
cco = 18  # Control Channel Overheads, %
cells_per_site = 3  # Number of cells per site
fb_per_site = 3  # No. of Frequency Bands on Site
angles_num = 360  # Number of angles for azimuth analysis
rotation_angle = 60  # Rotation angle for sector creation

# POI requirements
dlthtarg = 20  # Download throughput target in Mbps

# Population information
oppopshare = 50  # % of Population on Operator

# Avg user traffic profile
nonbhu = 50  # Non-Busy Hour Usage, %

# Instantiate Capacity class
mobilecapacity = Capacity(data_files, country_name,
                          bw, cco, cells_per_site, fb_per_site, angles_num,
                          rotation_angle, dlthtarg, oppopshare, nonbhu,
                          rb_num_multiplier=5, nbhours=10, root_dir=os.getcwd(), enable_logging=False)

# Define function to compute capacity


def compute_capacity(d, popcd, pop):
    return mobilecapacity.capacity_checker(d=d, popcd=popcd, udatavmonth=mobilecapacity.udatavmonth_pu, pop=pop)


# User inputs (outside of main layout function)
d = st.number_input("Distance between POI and tower (km)", min_value=0, step=1)
popcd = st.number_input("Distance between population centre and tower (km)", min_value=0, step=1)
pop = st.number_input("Total Population (pop)", min_value=0, step=1)

# Main layout function


def main():
    st.sidebar.title("Parameters")
    st.sidebar.write("Configure parameters and click 'Compute' to check capacity.")

    if st.sidebar.button("Compute"):
        result = compute_capacity(d, popcd, pop)
        st.write(f"Capacity check: {result}")


# Run the app
if __name__ == "__main__":
    main()
