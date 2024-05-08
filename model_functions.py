import numpy as np
import pandas as pd
from tqdm import tqdm
tqdm.pandas()
import tensorflow as tf
import sensor_meta_info as meta
import multiprocessing
import time

def estimate_chla(
    input_dataset,  # a pandas DataFrame or Xarray dataset
    data_type:str,
    bnn_model,  # a BNN trained to estimate chlorophyll-a concentrations
    sensor: str,  # a string specifying the type of sensor used to collect the data in `df`
    variants: int,  # the number of model NN variants to use in the calculation of BNN chla (parameter S in the paper, see Eq. 3)
    parallelise: bool = True,  # whether to use CPU parallelisation for the calculation - strongly recommended
    num_cpus: int = None,  # the number of CPUs to use for parallelisation (if parallelise is True)
):

    """Calculate BNN chla.

    This function calculateschlorophyll-a concentrations and the corresponding standard deviation using a BNN model.
    The calculation can be done in parallel using the TensorFlow dataset API.
    The user can specify the number of CPUs to use for parallelisation. If no value is specified, the number of CPUs will be used (minus 1 for safety).
    If parallelisation is not used, the calculation will be done using vectorisation, which is slower but still is as efficient as possible given the single CPU.

    Args:
        input_dataset: a pandas DataFrame or Xarray dataset
        bnn_model: a BNN model trained to estimate chlorophyll-a concentration
        sensor: a string specifying the type of sensor used to collect the data
        variants: the number of model NN variants to use in the calculation of BNN chla
        parallelise: whether to use parallelisation for the calculation (default: True)
        num_cpus: the number of CPUs to use for parallelisation (if parallelise is True)

    Returns:
        df or Xarray dataset containing the original data plus two new columns:
            'BNN_chla': the estimated chlorophyll-a concentrations from the BNN model
            'BNN_std': the standard deviation of the predicted chlorophyll-a concentrations from the BNN model
    """
    if data_type == 'in_situ':
        # If parallelise is True, use the number of CPUs specified in the num_cpus parameter. If num_cpus is not defined, use maximum available cpus -1 (hardware safety).
        if parallelise:
            if not num_cpus:
                num_cpus = multiprocessing.cpu_count() - 1
        # if parallelise is False, use one cpu
        else:
            num_cpus = 1

        # Print the number of CPUs that are being used for parallelisation
        print(f"Using {num_cpus} CPUs for parallelisation")

        start_time = time.time()
        
        df=input_dataset
        # get normalised sensor bands
        sensor_bands = meta.get_sensor_config(sensor)
        sensor_bands_norm = [x +'_norm' for x in sensor_bands]
        rrs_df = df[sensor_bands_norm].values

        # Calculate the number of observations to calculate chla for
        num_data_points = rrs_df.shape[0]

        # If parallelise is True, use the TF dataset API to parallelise the prediction
        if parallelise:
            # Create a TF dataset from the input data
            dataset = tf.data.Dataset.from_tensor_slices(rrs_df)

            # If the parallelise argument is True, then a TensorFlow dataset is created from the input data in the rrs_df DataFrame. 
            # The predictions are then made in a list comprehension that iterates over a range of variants and uses the BNN model to make predictions on the dataset, 
            # with a batch size equal to the number of rows in the rrs_df DataFrame. The resulting predictions are stored in the predictions variable as a list of predicted values.

            predictions = [bnn_model.predict(dataset.batch(batch_size=rrs_df.shape[0])) for _ in range(variants)]

            # Here the list of predictions stored in the predictions variable is converted to a 2D numpy array. 
            # The np.stack() function is used to combine the predictions into a single array, with the axis argument set to 0 to stack the predictions along the first dimension of the array. 
            # The resulting array is stored in the mc_cpd variable. The mc_cpd variable now contains the BNN model's predictions for each of the variants iterations in a 2D numpy array. 

            # Convert the list of predictions to a 2D numpy array
            mc_cpd = np.stack(predictions, axis=0)
        else:
            # If parallelise is False, use vectorisation instead
            mc_cpd = np.zeros((variants, len(rrs_df)))

            # Here I initialise a 2D numpy array mc_cpd with the same shape as rrs_df, but with variants rows instead of a single row. 
            # This array will be used to store the BNN model's predictions. 
            # Then, a for loop is used to iterate over a range of variants and make predictions with the BNN model on the rrs_df DataFrame.  
            # The output of this function has an extra dimension, which is removed using the np.squeeze() function. 
            # The resulting prediction is then stored in the corresponding row of the mc_cpd array.

            for i, _ in tqdm(enumerate(range(variants))):
                # Remove extra dimension from the output of bnn_model.predict()
                mc_cpd[i,:] = np.squeeze(bnn_model.predict(rrs_df)) 

        # BNN estimated chla and standard deviation (that is not the uncertainty. The uncertainty is calculated in the below function "calculate_uncertainty", which needs confidence intervals.)
        df['BNN_chla_'+sensor] = np.mean(mc_cpd, axis=0)
        df['BNN_std_'+sensor] = np.std(mc_cpd, axis=0)

        # Stop timer and print elapsed time.
        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"Elapsed computation time: {elapsed_time:.2f} seconds.")
        print(f"Calculated BNN chla for {num_data_points} observations.")

        return df
    
    elif data_type == 'satellite':

        if parallelise:
            if not num_cpus:
                num_cpus = multiprocessing.cpu_count() - 1
        else:
            num_cpus = 1

        print(f"Using {num_cpus} CPUs for parallelisation")
        start_time = time.time()

        ds = input_dataset

        # Identify the scaled bands based on sensor configuration
        sensor_bands = meta.get_sensor_config(sensor)  # Replace with actual function to get sensor bands
        sensor_bands_scaled = ['scaled_' + band for band in sensor_bands if 'scaled_' + band in ds.variables]

        # Extract the relevant data
        rrs_data = ds[sensor_bands_scaled].to_array().values  # shape is (bands, lat, lon)
        rrs_data = rrs_data.transpose(1, 2, 0)  # reordering to (lat, lon, bands)

        # Create mask where 'scaled_Rrs_6' <= 0.000, i.e. flagged, and where owt_flag == 0, i.e. within the OWTs defined for BNN application
        key = 'scaled_Rrs_560' if 'scaled_Rrs_560' in ds else 'scaled_Rrs_6'
        valid_data_mask = (ds[key] > 0.000) & (ds['owt_flag'] == 0)
        valid_data_mask = valid_data_mask.values  # convert to numpy array

        # Apply mask to flatten the data for batch processing, excluding invalid pixels
        valid_indices = np.where(valid_data_mask.ravel())[0]  # flat indices of valid pixels
        rrs_data_flat = rrs_data.reshape(-1, rrs_data.shape[-1])[valid_indices]

        num_data_points = len(valid_indices)  # total number of valid spatial points

        if parallelise:
            dataset = tf.data.Dataset.from_tensor_slices(rrs_data_flat)
            predictions = [bnn_model.predict(dataset.batch(num_data_points)) for _ in range(variants)]
            mc_cpd = np.stack(predictions, axis=0)
        else:
            mc_cpd = np.zeros((variants, num_data_points))
            for i in range(variants):
                mc_cpd[i, :] = np.squeeze(bnn_model.predict(rrs_data_flat))

        # Ensuring the predictions are squeezed to remove singleton dimensions
        mean_predictions = np.mean(mc_cpd, axis=0).squeeze()  # Use squeeze here
        std_predictions = np.std(mc_cpd, axis=0).squeeze()    # Use squeeze here too

        # Reconstruct full results arrays with NaNs for invalid entries
        full_results = np.full((rrs_data.shape[0] * rrs_data.shape[1]), np.nan)
        full_std = np.full((rrs_data.shape[0] * rrs_data.shape[1]), np.nan)

        # Ensure indexing is correctly handled
        full_results[valid_indices] = mean_predictions
        full_std[valid_indices] = std_predictions

        # Reshape the results back to the original lat-lon grid dimensions
        full_results = full_results.reshape(rrs_data.shape[0], rrs_data.shape[1])
        full_std = full_std.reshape(rrs_data.shape[0], rrs_data.shape[1])

        # Updating the xarray dataset with new variables
        ds['BNN_chla_' + sensor] = (('lat', 'lon'), full_results)
        ds['BNN_std_' + sensor] = (('lat', 'lon'), full_std)

        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"Elapsed computation time: {elapsed_time:.2f} seconds.")
        print(f"Calculated BNN chla and std. deviation for {num_data_points} observations.")

        return ds

# negative loss-liklihood (NLL)
def NLL(y, distr): 

    '''
    The negative log likelihood function, which is implemented in the NLL function, is a measure of how well a probability distribution predicts a given set of data.
    It is defined as the negative logarithm of the probability of observing the data, given the probability distribution. 
    '''
    return -distr.log_prob(y) 

def calc_ci(y_est, y_std):
    """
    Calculate the 95% confidence interval for estimates.
    
    Args:
        y_est (array-like): Estimated values (mean).
        y_std (array-like): Standard deviations of the estimates.
    
    Returns:
        tuple: A tuple containing the difference between the bounds, the upper bounds, and the lower bounds.
    """
    z_score = 1.96  # corresponds to 95% CI
    upper_bound = y_est + z_score * y_std
    lower_bound = y_est - z_score * y_std
    diff_bounds = upper_bound - lower_bound
    
    return diff_bounds, upper_bound, lower_bound

def calculate_uncertainty(input_data, sensor_name, data_type):
    """
    Calculate the uncertainty as a percentage based on the 95% confidence interval for chlorophyll-a estimates.

    Args:
        input_data (DataFrame or Dataset): Container holding the chla estimates and std deviations.
        sensor_name (str): Identifier for the sensor used.
        data_type (str): Type of the data ('in_situ' or 'satellite').

    Returns:
        DataFrame or Dataset: The input data with additional fields for CI width, upper and lower bounds, and uncertainty percentage.
    """
    chla_est_key = f'BNN_chla_{sensor_name}'
    chla_std_key = f'BNN_std_{sensor_name}'

    # get estimates and standard deviations
    chla_est = input_data[chla_est_key].values
    chla_std = input_data[chla_std_key].values

    # calculate CI
    ci_width, ci_upper, ci_lower = calc_ci(chla_est, chla_std)

    half_ci_width = ci_width / 2
    uncertainty_percentage = (half_ci_width / chla_est) * 100

    # Integrate results into the original input_data structure
    if data_type == 'in_situ':
        input_data[chla_est_key + '_CI_width'] = ci_width
        input_data[chla_est_key + '_CI_upper'] = ci_upper
        input_data[chla_est_key + '_CI_lower'] = ci_lower
        input_data[chla_est_key + '_uncertainty_percent'] = uncertainty_percentage
    elif data_type == 'satellite':
        input_data[chla_est_key + '_CI_width'] = (('lat', 'lon'), ci_width)
        input_data[chla_est_key + '_CI_upper'] = (('lat', 'lon'), ci_upper)
        input_data[chla_est_key + '_CI_lower'] = (('lat', 'lon'), ci_lower)
        input_data[chla_est_key + '_uncertainty_percent'] = (('lat', 'lon'), uncertainty_percentage)

    return input_data
