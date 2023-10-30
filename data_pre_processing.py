from typing import Union
import math
from math import pi
import pandas as pd
import numpy as np
import pickle
import warnings
import sensor_meta_info as meta
import xarray as xr

def standardise_rrs(rrs_spectrum, data_type: str):
    """
    Standardisation as in Spyrakos et al. (2018) used in OWT flagging. 
    
    Args:
        rrs_spectrum (pandas.Series or xarray.DataArray): A one-dimensional pandas Series or a multi-dimensional xarray DataArray containing the Rrs spectrum to standardise.
        data_type (str): A string specifying the type of rrs_spectrum. Either 'in_situ' or 'satellite'. 
    
    Returns:
        pandas.Series or xarray.DataArray: The standardised Rrs spectrum.
    
    Raises:
        ValueError: If data_type is not one of 'in_situ' or 'satellite'.
    """

    if data_type not in ['in_situ', 'satellite']:
        raise ValueError("Invalid data_type. Expected one of: 'in_situ', 'satellite'")
    
    if np.isnan(rrs_spectrum).any():
        return rrs_spectrum

    if data_type == 'in_situ':
        # Calculate the area of the Rrs spectrum
        rrs_area = sum(rrs_spectrum)
        
        # Standardise the Rrs spectrum by dividing each element by the Rrs area
        standardised_rrs = rrs_spectrum / rrs_area
        
    elif data_type == 'satellite':
        # Calculate the area of the Rrs spectrum along the 'pixel' dimension, which is a stacked ('lat', 'lon') dimension 
        rrs_area = rrs_spectrum.sum(dim=('pixel'))
        
        # Standardise the Rrs spectrum by dividing each element by the Rrs area
        standardised_rrs = rrs_spectrum / rrs_area
    
    return standardised_rrs


def cal_sa(owt_rrs: np.ndarray, rrs_spectrum, data_type: str) -> float:
    """
    Calculate the spectral angle (SA) between an Rrs spectrum and an OWT.
    
    Args:
        owt_rrs (np.ndarray): A one-dimensional array containing the Rrs spectra of the OWTs.
        rrs_spectrum (pandas.Series or xarray.DataArray): A one-dimensional pandas Series or a two-dimensional xarray DataArray containing the Rrs spectrum to calculate the SA for.
        data_type (str): A string specifying the type of rrs_spectrum. Either 'in_situ' or 'satellite'. 
    
    Returns:
        float or xarray.DataArray: The spectral angle between the given Rrs spectra. If rrs_spectrum is a pandas Series, this is a single float. If rrs_spectrum is an xarray DataArray, this is an xarray DataArray of floats.
    
    Raises:
        ValueError: If data_type is not one of 'in_situ' or 'satellite'.
    """

    if data_type not in ['in_situ', 'satellite']:
        raise ValueError("Invalid data_type. Expected one of: 'in_situ', 'satellite'")

    # Calculate the spectral angle using the given Rrs spectra
    if data_type == 'in_situ':
        alfa = np.arccos(sum(rrs_spectrum * owt_rrs) / (math.sqrt(sum(rrs_spectrum ** 2)) * math.sqrt(sum(owt_rrs ** 2))))
    elif data_type == 'satellite':
        alfa = np.arccos(np.dot(rrs_spectrum, owt_rrs) / (np.linalg.norm(rrs_spectrum) * np.linalg.norm(owt_rrs)))
    
    sa_owt = 1 - alfa / pi
    
    return sa_owt

def calc_max_owt(rrs_spectrum, owt_array: np.ndarray, data_type: str) -> int:
    """
    Calculate the maximum OWT class for a given Rrs spectrum.
    
    Args:
        rrs_spectrum (pandas.Series or xarray.DataArray): A one-dimensional pandas Series or a two-dimensional xarray DataArray containing the Rrs spectrum to calculate the maximum OWT class for.
        owt_array (np.ndarray): A two-dimensional array containing the OWTs to use for the calculation.
        data_type (str): A string specifying the type of rrs_spectrum. Either 'in_situ' or 'satellite'. 
    
    Returns:
        int or xarray.DataArray: The maximum OWT class for the given spectrum. If rrs_spectrum is a pandas Series, this is a single int. If rrs_spectrum is an xarray DataArray, this is an xarray DataArray of ints.
    
    Raises:
        ValueError: If data_type is not one of 'in_situ' or 'satellite'.
    """

    if data_type not in ['in_situ', 'satellite']:
        raise ValueError("Invalid data_type. Expected one of: 'in_situ', 'satellite'")

    # Calculate the spectral angle (SA) for each OWT in the OWT array using the given Rrs spectrum
    if data_type == 'in_situ':
        all_sa = np.apply_along_axis(cal_sa, 1, owt_array, rrs_spectrum=rrs_spectrum, data_type=data_type)
        # Find the maximum SA and corresponding OWT class, + 1 because python starts counting at 0
        owt_class = np.argmax(all_sa) + 1

    elif data_type == 'satellite':
        #all_sa = rrs_spectrum.apply(lambda data_array: np.apply_along_axis(cal_sa, 1, owt_array, rrs_spectrum=data_array, data_type=data_type), dim=('band'))
        # Find the maximum SA and corresponding OWT class, + 1 because python starts counting at 0
        #owt_class = all_sa.argmax(dim=('band')) + 1

        all_sa = np.apply_along_axis(cal_sa, 1, owt_array, rrs_spectrum=rrs_spectrum, data_type=data_type)
        owt_class = np.argmax(all_sa) + 1  # +1 because Python starts counting at 0
        return owt_class
    
    return owt_class

def owt_flagging(owt_rrs: pd.DataFrame, input_dataset, sensor: str, data_type: str) -> Union[pd.DataFrame, xr.Dataset]:
    """
    Calculates an OWT per observation and the OWT quality flag (inside (0) and outside (1) of the application scope, respectively).
    
    Args:
        owt_rrs (pandas.DataFrame): A data frame containing the mean standardised OWTs of Spyrakos et al. (2018).
        input_dataset (pandas.DataFrame or xr.Dataset): A data frame or xarray dataset to produce the OWTs for.
        sensor (str): A sensor configuration, one of ['OLCI_all', 'OLCI_polymer', 'MSI_s2a', 'MSI_s2b'].
        data_type (str): A string specifying the type of input_dataset. Either 'in_situ' or 'satellite'. 
    
    Returns:
        pandas.DataFrame or xr.Dataset: If 'in_situ' data, returns the input pandas DataFrame with 'owt_class' and 'owt_flag' columns. If 'satellite' data, returns the input xarray Dataset with 'owt_class' and 'owt_flag' variables. The return is not standardised, this is just undertaken for the calculation of the OWTs.
    
    Raises:
        ValueError: If the input dataset contains NaN values in the input sensor columns.
        ValueError: If data_type is not one of 'in_situ' or 'satellite'.
        
    """

    if data_type not in ['in_situ', 'satellite']:
        raise ValueError("Invalid data_type. Expected one of: 'in_situ', 'satellite'")
    
    if data_type == 'in_situ':
        print('In situ data owt classification.')
        # Execute original function for pandas DataFrame
 
        # Select the sensor configuration for the given sensor
        sensor_bands = meta.get_sensor_config(sensor)
        
        # If it does, raise a ValueError - there should be no NaN values in any Rrs band in an in situ data file
        if input_dataset[sensor_bands].isnull().values.any():
            raise ValueError('The input dataset contains NaN values in one or more of the sensor columns specified in sensor_meta_info.py. Please treat/remove the corresponding observation.')

        # Select the specified columns from the owt_rrs and input_dataset data frames
        input_dataset_sel = input_dataset[sensor_bands]

        # Standardise the values in each row of input_dataset_sel
        input_dataset_standardised = input_dataset_sel.apply(lambda row: standardise_rrs(row, data_type), axis=1)

        # Calculate the maximum OWT class for each row in the input dataset
        owt_cols = ['wl'] + sensor_bands
        owt_sel = owt_rrs[owt_cols]
        owt_array = owt_sel.iloc[:, 1:].values
        max_owt_classes = np.apply_along_axis(lambda x: calc_max_owt(x, owt_array=owt_array, data_type=data_type), 1, input_dataset_standardised.values)

        # These are the valid OWTs of the manuscript the BNNs were designed for:
        bnn_owts = [2, 3, 4, 5, 9]

        # For each maximum OWT membership, determine the corresponding OWT flag
        owt_flag = 1 - np.isin(max_owt_classes, bnn_owts).astype(int)

        # Add the maximum OWT membership and OWT flag as columns to the input dataset
        input_dataset['owt_class'] = max_owt_classes
        input_dataset['owt_flag'] = owt_flag

        print('OWT flagging complete. New OWT columns "owt_class" and "owt_flag" added to input dataframe.')
    
    elif data_type == 'satellite':
        print('Satellite data owt classification.')
         # Select the sensor configuration for the given sensor
        sensor_bands = meta.get_sensor_config(sensor)
            
        # Create a list of data arrays for each band in the sensor configuration
        list_of_data_arrays = [input_dataset[band] for band in sensor_bands]
        # Combine the list of data arrays into a new data array
        data_array_combined = xr.concat(list_of_data_arrays, dim='band')
        # Stack 'lat' and 'lon' into a single multi-index
        data_array_combined_stacked = data_array_combined.stack(pixel=('lat', 'lon'))
        # Group by 'pixel' and apply the standardise_rrs function to each group
        input_dataset_standardised_stacked = data_array_combined_stacked.groupby('pixel').apply(standardise_rrs, data_type='satellite')
        print('Standardising all valid Rrs spectra.')
        # Unstack the 'pixel' multi-index back to 'lat' and 'lon'
        input_dataset_standardised = input_dataset_standardised_stacked.unstack('pixel')

        # Split the standardised data array back into separate bands and update the original dataset
        ''
        for i, band in enumerate(sensor_bands):
            new_band_name = 'standardised_' + band
            input_dataset[new_band_name] = input_dataset_standardised.isel(band=i)

        ('Standardisation complete. New standardized bands added to input dataset.')

        # Select OWT columns for calculating the maximum OWT membership
        # The OWT dataframe is already standardised and as such both data sources share the same standardisation

        # OWT band selection is independent of AC, but depends on sensor

        # If 'OLCI' is in 'sensor', change 'sensor' to 'OLCI_all', e.g. if it's OLCI_c2rcc
        if 'OLCI' in sensor:
            sensor = 'OLCI_all'
    
        # Fetch the sensor configuration based on the updated 'sensor'
        sensor_bands = meta.get_sensor_config(sensor)
        
        owt_cols = ['wl'] + sensor_bands
        # Select the specified columns from the owt_rrs data frame
        owt_sel = owt_rrs[owt_cols]
        owt_array = owt_sel.iloc[:, 1:].values

        # calculate OWT using the standardised pixels (input_dataset_standardised)
        
        result = xr.apply_ufunc(
            calc_max_owt,  # The function to apply
            input_dataset_standardised,  # The DataArray to operate on - it has to be input_dataset_standardised, because this one has the 'band' value and contains the 12 Rrs bands
            kwargs={'owt_array': owt_array, 'data_type': 'satellite'},  # Additional keyword arguments to the function
            input_core_dims=[['band']],  # The dimensions along which to apply the function
            vectorize=True,  # If true, will vectorize calc_max_owt_ufunc to work on each 'lat', 'lon' pair
            dask='parallelized',  # Enable Dask for parallel computing, if you're using Dask-backed arrays
            output_dtypes=[int]  # The dtype of the output DataArray
        )

        #max_owt_classes = input_dataset_standardised.apply(lambda x: calc_max_owt(x, owt_array=owt_array, data_type=data_type), dim=('lat', 'lon'))
        # These are the valid OWTs of the manuscript the BNNs were designed for:
        bnn_owts = [2, 3, 4, 5, 9]

        # For each maximum OWT membership, determine the corresponding OWT flag
        # Using xarray's where method to handle multi-dimensional array
        owt_flag = xr.where(np.isin(result, bnn_owts), 0, 1)

        # Capture dimension names from input_dataset
        dims = list(input_dataset.dims.keys())

        # Use .data to get the underlying array from 'result' and 'owt_flag'
        input_dataset['owt_class'] = (dims, result.data)
        input_dataset['owt_flag'] = (dims, owt_flag.data)

        print('Satellite owt flagging complete. New OWT variables "owt_class" and "owt_flag" added to input dataset.')

        return input_dataset

def normalise_input(df: pd.DataFrame, sensor: str, cwd_path: str, reset_index:bool=True) -> pd.DataFrame:
    """
    Sets all negative values to 0, and uses a scaler/transformer to apply the 0-1 normalisation to stay within the scale of training dataset.
    This normalisation is independent of the standardisation that is specific to the OWT classification.
    
    Args:
        df: A DataFrame containing the input data.
        sensor: A string representing the satellite sensor.
        reset_index: A boolean indicating whether the index of the DataFrame should be reset to a monotonically increasing sequence (by 1) starting from 0. Defaults to True.
        
    Returns:
        A DataFrame with the same columns as the input DataFrame, but with normalised values and an additional "is_negative" column.

    Raises:
        ValueError: If the index in df is not strictly increasing by 1.
        UserWarning: If any of the data contains negative values, a warning is raised with
        the index of the specific row containing the negative value.
    """
    # Check if reset_index is False
    if not reset_index:
        # Don't reset the index of df
        pass
    else:
        # Reset the index of df
        df = df.reset_index(drop=True)
        print('Index of input "df" reset.')

    # Check index of df
    # Get the index of the DataFrame as array
    index = df.index.to_numpy()

    # Check whether the difference between consecutive indices is always 1
    differences = np.diff(index)
    if not np.all(differences == 1):
        raise ValueError("Error: Index is not strictly increasing. It has to be increasing by 1. You may just use use the reset_index argument of this function (default: True) to achieve this.")

    # continue code after index checks
    # get sensor configuration
    input_bands = meta.get_sensor_config(sensor)
    input_bands_int = np.array([int(x) for x in input_bands])
    
    # separate the wavelengths
    greater_665 = input_bands_int[np.where(input_bands_int>=665)]
    smaller_665 = input_bands_int[np.where(input_bands_int<665)]

    # select the respective columns here
    to_zero = df[[str(x) for x in greater_665]]
    standard = df[[str(x) for x in smaller_665]]

    # set all data that is negative to 0 in the wavelengths > 665 nm 
    # blue and green wavelengths are not set to zero, but a warning is raised, including the index - see also column "is_negative"
    # the BNNs should not be applied to negative values < 665 nm as there is no training data that covers this.
    
    zeroed = to_zero.mask(to_zero < 0, 0)
    non_negative = standard.join(zeroed)

    is_negative = (standard.lt(0) | zeroed.lt(0)).any(axis=1)

    # Raise a warning if any of the values are negative
    if is_negative.any():
        message = "The data contains negative values at index "
        if (standard < 0).any().any():
            negative_index = standard[standard < 0].index
        else:
            negative_index = zeroed[zeroed < 0].index
        # Get the specific row index that contains the negative value
        negative_row = negative_index[is_negative[is_negative].index[0]]
        message += str(negative_row)
        warnings.warn(message, UserWarning)

    # Create a dictionary that maps the sensor value to the appropriate suffix value
    suffix_dict = {
        'OLCI_all': 's3b',
        'OLCI_polymer': 's3b',
        'MSI_s2b': 's2b',
        'MSI_s2a': 's2a',
    }

    # Use the dictionary to get the appropriate suffix for the sensor
    suffix = suffix_dict.get(sensor, '')

    # Create the list of columns used for fitting the normalisation scaler
    scaler_fitting_cols = meta.get_scaler_config(sensor)
    scaler_fitting_cols = [col[:3] + '_res_' + suffix for col in scaler_fitting_cols]
    
    # Create a mapping of columns in the non_negative DataFrame to the fitting columns and rename
    column_mapping = {col: scaler_fitting_cols[i] for i, col in enumerate(non_negative.columns)}
    non_negative = non_negative.rename(columns=column_mapping)

    # 0 - 1 normalisation, load the scaler/transformer to bring new input data into the scale of the training dataset
    try:
        loaded_scaler = pickle.load(open(cwd_path+'/scalers/'+sensor+'_scaler.sav', 'rb'))
    except FileNotFoundError:
        print("Error: The scaler in /BNN_2022/scalers/ could not be found. Please make sure your working directory is the correct repository. Otherwise you can explicitly define it using the cwd_path argument in this function.")
    # scale the data using the transform method
    normalised = loaded_scaler.transform(non_negative)

    # Create the suffix for the normalised wavelengths
    string_norm = [str(int) + '_norm' for int in input_bands]

    # Create the DataFrame with the normalised columns
    df_norm = pd.DataFrame(data=normalised, columns=string_norm)

    # Include the additional columns from the input DataFrame and add a column where negative
    df_preprocessed = df_norm.join(df).assign(is_negative=is_negative.astype(int))

    print('Normalisation and treating of negative values complete. New columns "is_negative" added to the dataframe.')
    return df_preprocessed

