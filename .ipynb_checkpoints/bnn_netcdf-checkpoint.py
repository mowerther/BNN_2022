#####
# 1. Load libraries
#####

# Generic imports:
import importlib
import os
import numpy as np
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
import pandas as pd
from tqdm import tqdm
tqdm.pandas()
import tensorflow as tf
import tensorflow_probability as tfp # TFP is a necessary import for the output distribution layer of the pre-trained BNNs
# Imports specific to this repository:
import performance_metrics as perf_met
import uncertainty_calibration_metrics as unc_met
from data_pre_processing import owt_flagging
from data_pre_processing import normalise_input
from model_functions import NLL
from model_functions import estimate_chla
from model_functions import calculate_uncertainty
import xarray as xr

#####
# 2. Load model and data
#####
# If you run into loading problems: make sure the correct Python version is used. 
# Python version that corresponds to the .h5 Keras models: 3.8.15

# Use os.path.join() to combine the different components of the path into a single string
# using forward slashes as the separator/
# takes your working directory, assuming the same folder structure as in the repository.
cwd = os.getcwd()
cwd_system_wide = os.path.join(*cwd.split("/"))
print(cwd_system_wide)

# define sensor to apply the BNNs to (case sensitive):
sensor_name = 'OLCI_all' # any of: 'OLCI_all', 'OLCI_polymer, 'MSI_s2a', 'MSI_s2b'.

# 'OLCI_all' bands, see sensor_meta_info.py:
# '413', '443', '490','510', '560', '620', '665', '673', '681', '708', '753','778'

# Load the model

bnn_sensor_model = tf.keras.models.load_model(cwd_system_wide+'/bnns/BNN_'+sensor_name+'.h5', custom_objects = {'NLL': NLL})
print(sensor_name + ' BNN model loaded.')

### Load netcdf and provide


# Load example IN SITU data:
try:
    df_input = pd.read_csv(cwd_system_wide+'/data/example_data.csv')
    print('Example data loaded.')
except:
    print('Error: Failed to load the example data csv file!')


# Load optical water types of Spyrakos et al. (2018):
try:
    owts_iw = pd.read_csv(cwd_system_wide+'/data/spyrakos_owts_inland_waters_standardised.csv')
    print('OWTs inland water loaded.')
except:
    print('Error: Failed to load the OWT inland water dataset!')

####
# 2. OWT flagging - generates OWT flag (0 = inside application scope, 1 = outside application scope)
####

import data_pre_processing
import sensor_meta_info
importlib.reload(data_pre_processing)
importlib.reload(sensor_meta_info)

# 1. in situ dataset

sensor_name = 'OLCI_all'
input_data_owts = data_pre_processing.owt_flagging(owt_rrs = owts_iw, input_dataset=df_input, sensor=sensor_name, data_type='in_situ')

# 2. Satellite scene NETCDF (OLCI/MSI) 

# open dataset using Xarray
df_sat = xr.open_dataset(cwd_system_wide+'/data/subset_0_of_Mosaic_L2C2RCC_NA_S3A_OL_1_EFR_20200806__NT.nc')
# Create a mask where the pixel_classif_flags is equal to -32768
mask = df_sat['pixel_classif_flags'] == -32768
# Filter the data using the mask
filtered_sat = df_sat.where(mask, drop=True)

# Define original sensor + AC used
sensor_name = 'OLCI_c2rcc'

# Calculate Rrs and generate a modified dataset with new Rrs bands
olci_c2rcc_bands = sensor_meta_info.get_sensor_config(sensor_name)
filtered_sat_rrs = sensor_meta_info.calculate_rrs(filtered_sat, olci_c2rcc_bands)

# Use the Rrs sensor bands
sensor_rrs ='OLCI_c2rcc_rrs'

# Process dataset 
input_data_owts = data_pre_processing.owt_flagging(owt_rrs = owts_iw, input_dataset=filtered_sat_rrs, sensor=sensor_rrs, data_type='satellite')

## instead:

# Open the dataset
df = xr.open_dataset('Mosaic_L2C2RCC_NA_S3A_OL_1_EFR_20200806__NT.nc')

# Create a mask where the pixel_classif_flags is equal to -32768 - for this scene specifically this number is generated and corresponds to valid pixels
mask = df['pixel_classif_flags'] == -32768

# Filter the data using the mask
filtered_df = df.where(mask, drop=True)

# Create a list of bands to use in the algorithm
bands = ['rhow_1', 'rhow_2']
# Get the dimensions of one of the existing bands, necessary for writing the new bands later on
dims = df['rhow_1'].dims

# Apply the algorithm to the selected bands
# here first owts, then normalised, then the bnns
# just an example
new_band = np.mean([filtered_df[band] for band in bands], axis=0)


# Load optical water types of Spyrakos et al. (2018):
try:
    owts_iw = pd.read_csv(cwd_system_wide+'/data/spyrakos_owts_inland_waters_standardised.csv')
    print('OWTs inland water loaded.')
except:
    print('Error: Failed to load the OWT inland water dataset!')

####
# 2. OWT flagging - generates OWT flag (0 = inside application scope, 1 = outside application scope)
####

input_data_owts = owt_flagging(owts_iw, df_input, sensor=sensor_name)

####
# 3. Normalise and treat negative values, generates flag "is_negative" (0 = not negative, 1 = negative)
####

input_data_owts_normalised = normalise_input(input_data_owts,sensor=sensor_name, cwd_path = cwd_system_wide)

####
# 5. Calculate chla and associated uncertainty 
####

# Select the correct BNN model - applied to normalised data and all observations - flagging of results later.

# Variable 'variants' is the parameter S in the manuscript (Eq. 3)

input_data_owts_normalised_chla = estimate_chla(input_data_owts_normalised, bnn_model=bnn_sensor_model, sensor=sensor_name, variants=50, parallelise=True, num_cpus=10)

# this file contains the normalised input columns ("suffix "_norm" after the wavelengths) and "owt_class", "owt_flag", "is_negative" and the BNN outputs: "BNN_chla", "BNN_std", "BNN_chla_unc".
# the flags "owt_flag", "is_negative" can be used to filter the output.

# Calculate uncertainty, Eq. 4 in the manuscript

chla_est = input_data_owts_normalised_chla['BNN_chla_'+sensor_name].values
chla_std = input_data_owts_normalised_chla['BNN_std_'+sensor_name].values

# Calculate confidence interval, dif_org_bounds is the confidence interval width (CI_w)
dif_org_bounds, orig_bounds_up, orig_bounds_low = unc_met.calc_ci(chla_est, chla_std)

# Uncertainty in %
perc_unc = calculate_uncertainty(chla_est, dif_org_bounds)
input_data_owts_normalised_chla['BNN_chla_unc_'+sensor_name] = perc_unc


# Get the dimensions of one of the existing bands
dims = df['rhow_1'].dims

# Add the result as a new band in the original dataset with correct dimensions
df['bnn_chla'] = (dims, chla_est)
df['bnn_uncertainty'] = (dims, perc_unc)