'''
This script demonstrates the use of BNNs (Bayesian Neural Networks) based on Monte Carlo Dropout to estimate phytoplankton chlorophyll-a chla concentration (chla) and associated chla retrieval uncertainty.
The BNNs were trained with in situ measurements from oligotrophic and mesotrophic lakes with maximum chla 68 mg/m^-3 and reflectance spectra corresponding to OWTs 2, 3, 4, 5, 9 of Spyrakos et al. (2018). 
Using the BNNs outside of this scope is not recommended and will lead to high retrieval inaccuracies and associated uncertainties.

This script utilises the "OLCI_all" model and two datasets (data folder):
(1) the optical water types (OWTs) from Spyrakos et al. (2018)
(2) spectrally convolved Rrs to OLCI and in situ Chla from the manuscript for this script.
Other models can also be tried with this example data, see variable "sensor_name" below.

The script proceeds as follows:

    Import necessary libraries and load the "OLCI_all" BNN model, input data and determine current working directory.
    Perform OWT classification to generate OWT flags.
    Normalise the input data and treat negative values.
    Calculate chla concentration and retrieval uncertainty using the BNN model.
    Calculate performance and uncertainty calibration metrics, saved in df_metrics.
    Produce a scatterplot of the results.

It is important to note that the "OLCI_all" BNN model and the data used in this script are for illustration purposes only. 
The model and data may not be representative of other datasets and application contexts, and using the model outside of its intended scope may lead to inaccurate results. 
Therefore, users should carefully evaluate the suitability of the model and data for their specific use case.
'''
#####
# 1. Load libraries
#####

# Generic imports:
import os
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
from data_processing import owt_flagging
from data_processing import normalise_input
from model_functions import NLL
from model_functions import estimate_chla
from model_functions import calculate_uncertainty

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

# Load the model

bnn_sensor_model = tf.keras.models.load_model(cwd_system_wide+'/bnns/BNN_'+sensor_name+'.h5', custom_objects = {'NLL': NLL})
print(sensor_name + ' BNN model loaded.')

# Load example data:
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
# 3. OWT flagging - generates OWT flag (0 = inside application scope, 1 = outside application scope)
####

input_data_owts = owt_flagging(owts_iw, df_input,sensor=sensor_name, data_type='in_situ')

####
# 4. Normalise and treat negative values, generates flag "is_negative" (0 = not negative, 1 = negative)
####

input_data_owts_normalised = normalise_input(input_data_owts,sensor=sensor_name, scaler_name=sensor_name, data_type='in_situ', cwd_path = cwd_system_wide,reset_index=True)

####
# 5. Calculate chla and associated uncertainty 
####

# Select the correct BNN model - applied to normalised data and all observations - flagging of results later.

# Variable 'variants' is the parameter S in the manuscript (Eq. 3)

input_data_owts_normalised_chla = estimate_chla(input_data_owts_normalised, data_type='in_situ', bnn_model=bnn_sensor_model, sensor=sensor_name, variants=50, parallelise=True, num_cpus=10)

# this file contains the normalised input columns ("suffix "_norm" after the wavelengths) and "owt_class", "owt_flag", "is_negative" and the BNN outputs: "BNN_chla", "BNN_std", "BNN_chla_unc".
# the flags "owt_flag", "is_negative" can be used to filter the output.

# Calculate uncertainty, Eq. 4 in the manuscript
bnn_chla_unc = calculate_uncertainty(input_data_owts_normalised_chla, sensor_name, 'in_situ')

####
# 6. Calculate performance and uncertainty calibrationaccuracy metrics
# _ex = example
####

# Performance 
chla_ref = input_data_owts_normalised_chla['CHLA'].values
chla_est = input_data_owts_normalised_chla['BNN_chla_'+sensor_name].values
chla_std = input_data_owts_normalised_chla['BNN_std_'+sensor_name].values

# Calculate confidence interval, dif_org_bounds is the confidence interval width (CI_w)
dif_org_bounds, orig_bounds_up, orig_bounds_low = unc_met.calc_ci(chla_est, chla_std)

mad_ex = perf_met.MAD(chla_ref, chla_est)
mapd_ex = perf_met.mapd(chla_ref, chla_est)
mdsa_ex = perf_met.mdsa(chla_ref, chla_est)
sspb_ex = perf_met.sspb(chla_ref, chla_est)

# Uncertainty calibration
perc_in_ex, perc_out_ex = unc_met.picp(orig_bounds_low, chla_ref, orig_bounds_up)
sharpness_ex = unc_met.sharpness(chla_std)
macd_ex = unc_met.macd(chla_est, chla_ref, chla_std)

# Metric DF
metrics = [mad_ex, mapd_ex, mdsa_ex, sspb_ex, perc_in_ex, sharpness_ex, macd_ex]
metric_names =['MAD []', 'MAPD [%]', 'MdSA [%]', 'SSPB [%]', 'PICP [%]', 'Sharpness []', 'MACD []']
df_metric=pd.DataFrame(data=metrics,
                index=metric_names,
                columns=['Metric value'])

####
# 7. Plot results
####

fig, ax = plt.subplots(figsize=(4,4))

ax.scatter(chla_ref, chla_est, alpha=0.7, c='black')
ax.grid('True', ls='--', c='black', alpha=0.7)
ax.loglog()
ax.set_xlim(10**-1,10**2)
ax.set_ylim(10**-1,10**2)
ax.plot([10**-1,10**2],[10**-1,10**2], '--', c='black')
for axis in [ax.xaxis, ax.yaxis]:
    axis.set_major_formatter(ScalarFormatter())
ax.set_xlabel('In situ chla [mg m$^{-3}$]',fontsize=10)
ax.set_ylabel('BNN estimated chla [mg m$^{-3}$]',fontsize=10)
ax.set_title('BNN model' + ' ' + sensor_name + ' ' + 'test: region 3 (Switzerland/Italy)')
df_metric
plt.show()
