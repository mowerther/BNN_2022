import numpy as np
import pandas as pd
from tqdm import tqdm
tqdm.pandas()
import tensorflow as tf
import sensor_meta_info as meta
import multiprocessing
import time

def estimate_chla(
    df: pd.DataFrame,  # a pandas DataFrame containing the sensor data
    bnn_model,  # a Bayesian neural network model trained to predict chlorophyll-a concentrations
    sensor: str,  # a string specifying the type of sensor used to collect the data in `df`
    variants: int,  # the number of model NN variants to use in the calculation of BNN chla (parameter S in the paper, see Eq. 3)
    parallelise: bool = True,  # whether to use CPU parallelisation for the calculation - strongly recommended
    num_cpus: int = None,  # the number of CPUs to use for parallelisation (if parallelise is True)
) -> pd.DataFrame:

    """Calculate BNN chla.

    This function calculateschlorophyll-a concentrations and the corresponding standard deviation using a BNN model.
    The calculation can be done in parallel using the TensorFlow dataset API.
    The user can specify the number of CPUs to use for parallelisation. If no value is specified, the number of CPUs will be used (minus 1 for safety).
    If parallelisation is not used, the calculation will be done using vectorisation, which is slower but still is as efficient as possible given the single CPU.

    Args:
        df: a pandas DataFrame containing the sensor data
        bnn_model: a BNN model trained to estimat chlorophyll-a concentration
        sensor: a string specifying the type of sensor used to collect the data in `df`
        variants: the number of model NN variants to use in the calculation of BNN chla
        parallelise: whether to use parallelisation for the calculation (default: True)
        num_cpus: the number of CPUs to use for parallelisation (if parallelise is True)

    Returns:
        df: a pandas DataFrame containing the original data in `df`, plus two new columns:
            'BNN_chla': the estimated chlorophyll-a concentrations from the BNN model
            'BNN_std': the standard deviation of the predicted chlorophyll-a concentrations from the BNN model
    """
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

def calculate_uncertainty(chla_est, intervals):
    '''
    A utility function that calculates uncertainty in the form of a percentage. 
    It takes in the predicted value chla_pred and confidence intervals intervals and returns the uncertainty in the form of a percentage. 
    This is done by first dividing the intervals by 2 to get the half-width of the intervals, then dividing this half-width by the predicted value and multiplying by 100 to get the uncertainty as a percentage. The result is rounded to 3 decimal places.
    '''
    half_intervals = intervals/2
    percentages = half_intervals/chla_est*100
    return np.round(percentages,3)

# negative loss-liklihood (NLL)
def NLL(y, distr): 

    '''
    The negative log likelihood function, which is implemented in the NLL function, is a measure of how well a probability distribution predicts a given set of data.
    It is defined as the negative logarithm of the probability of observing the data, given the probability distribution. 
    '''
    return -distr.log_prob(y) 