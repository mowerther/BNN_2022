###
# 1. Load libraries
###

import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import pandas as pd
from tqdm import tqdm
tqdm.pandas()
import tensorflow as tf
import tensorflow_probability as tfp # TFP is a necessary import for the output distribution layer of the pre-trained BNNs

from data_processing import owt_flagging
from data_processing import normalise_input
from data_processing import load_datasets
from data_processing import valid_pixel_classif
from data_processing import get_rrs

from model_functions import estimate_chla
from model_functions import calculate_uncertainty

# If you run into loading problems: make sure the correct Python version is used. 
# Python version that corresponds to the .h5 Keras models: 3.8.15

######
# 2. Preparation and loading of models and data
######

cwd = os.getcwd()
cwd_system_wide = os.path.join(*cwd.split("/"))
print(cwd_system_wide)

# Any of: 'OLCI_all', 'OLCI_polymer', 'MSI_s2a', 'MSI_s2b'
sensor_name = 'OLCI_all'
#sensor_name= 'MSI_s2b'

# Example subset of L2 AC IDEPIX+C2RCC (OLCI)
file_name = 'subset_0_of_Mosaic_L2C2RCC_NA_S3A_OL_1_EFR_20200806__NT.nc'

# Example subset of L2 AC POLYMER (MSI)
#file_name = 'L2POLY_reproj_NASA_S2B_MSIL1C_20231220T103349_N0510_R108_T31TGN_20231220T112131.SAFE.nc'

bnn_sensor_model, owts_iw, df_sat = load_datasets(sensor_name, file_name)

# Whether to use the valid pixel identification (here for OLCI from IDEPIX)
filtered_sat = valid_pixel_classif(df_sat, apply_mask=True, vp_id=-32768, valid_pixel_band='pixel_classif_flags')
# Or just proceed with the whole scene (here for MSI from POLYMER)
# filtered_sat = valid_pixel_classif(df_sat, apply_mask=False)

# See get_rrs for the sensor_ac accepted strings.
filtered_sat_rrs, sensor_rrs_id = get_rrs(filtered_sat, 'OLCI_c2rcc')
#filtered_sat_rrs, sensor_rrs_id = get_rrs(filtered_sat, 'MSI_s2b_polymer_rw')

############
# 3. OWT flagging - generates OWT flag (0 = inside application scope, 1 = outside application scope)
############

# 3.1 OWT flagging of the satellite image
satellite_owts = owt_flagging(owt_rrs = owts_iw, input_dataset=filtered_sat_rrs, sensor=sensor_rrs_id, data_type='satellite')

# 3.2 Negative flagging and normalisation for the BNN application
satellite_owts_normalised_rrs = normalise_input(satellite_owts, data_type='satellite', sensor=sensor_rrs_id, scaler_name=sensor_name, cwd_path= cwd_system_wide, reset_index=True)

########################
# 4. BNN application
########################

# BNN application - the number of CPUs is user-defined. 1 = no parallelisation. Parallelising through CPUs is strongly recommended.
# Variants (parameter S in the manuscript) is the number of Monte Carlo Dropout (MCD) samples of the network configurations that should be performed for each pixel.
bnn_chla = estimate_chla(satellite_owts_normalised_rrs, data_type='satellite',bnn_model=bnn_sensor_model, sensor=sensor_rrs_id, variants=100, parallelise=True, num_cpus=10)

# BNN chla uncertainty calculation
bnn_chla_unc = calculate_uncertainty(bnn_chla, sensor_rrs_id, 'satellite')

# Inspect the new product bands or save as netcdf to load somewhere else:
bnn_chla_unc

#bnn_chla_unc.to_netcdf(cwd_system_wide+'/data/msi_s2b_polymer_bnn_chla_unc_product.nc')
#bnn_chla_unc.to_netcdf(cwd_system_wide+'/data/olci_c2rcc_bnn_chla_unc_product.nc')
