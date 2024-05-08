from typing import Union
import math
from math import pi
import os
import tensorflow as tf
import re
import pandas as pd
import numpy as np
import xarray as xr
from numpy import pi
import pickle
import warnings
import sensor_meta_info as meta
from model_functions import NLL


def load_datasets(sensor_name, file_name):
    """
    Loads a BNN model and data for a specified sensor, applies processing, and returns necessary outputs, automatically determining paths.
    
    :param sensor_name: Name of the sensor (e.g., 'OLCI_all', 'MSI_s2b')
    :param file_name: Name of the file to load the dataset from
    :return: Loaded model, dataset, and any other processed outputs
    """
    # Setup directory paths using the current working directory
    cwd = os.getcwd()
    cwd_system_wide = os.path.join(*cwd.split("/"))
    model_directory = os.path.join(cwd_system_wide, 'bnns')
    data_directory = os.path.join(cwd_system_wide, 'data')
    
    # Load the BNN model
    model_path = f"{model_directory}/BNN_{sensor_name}.h5"
    try:
        bnn_model = tf.keras.models.load_model(model_path, custom_objects={'NLL': NLL})
        print(f"{sensor_name} BNN model loaded.")
    except Exception as e:
        print(f"Error loading model from {model_path}: {e}")
        bnn_model = None

    # Load the optical water types (OWTs)
    try:
        owts_iw = pd.read_csv(f"{data_directory}/spyrakos_owts_inland_waters_standardised.csv")
        print('OWTs inland water loaded.')
    except Exception as e:
        print(f"Error: Failed to load the OWT inland water dataset! {e}")
        owts_iw = None

    # Load the satellite netCDF dataset
    try:
        df_sat = xr.open_dataset(f"{data_directory}/{file_name}")
        print('Satellite dataset loaded.')
    except Exception as e:
        print(f"Error loading satellite data from {file_name}: {e}")
        df_sat = None

    return bnn_model, owts_iw, df_sat

def valid_pixel_classif(df_sat, apply_mask=False, vp_id=None, valid_pixel_band=None):
    """
    Processes the satellite data using optional valid pixel masking. Masking requires a valid pixel_id and a specific pixel classification band that was previously computed.
    
    :param df_sat: Loaded xarray dataset of satellite data.
    :param apply_mask: Boolean to determine if a mask should be applied. Requires vp_id and valid_pixel_band to be provided.
    :param vp_id: The pixel classification identifier used for masking.
    :param valid_pixel_band: The data column to apply the mask to (e.g., 'pixel_classif_flags').
    :return: Processed satellite dataset.
    
    :raises ValueError: If apply_mask is True and either vp_id or valid_pixel_band is not provided.
    """
    if apply_mask:
        if vp_id is None or valid_pixel_band is None:
            raise ValueError("Both 'vp_id' and 'valid_pixel_band' must be provided when 'apply_mask' is True.")
        
        if valid_pixel_band not in df_sat:
            raise ValueError(f"The specified 'valid_pixel_band' {valid_pixel_band} does not exist in the dataset.")
        
        mask = df_sat[valid_pixel_band] == vp_id
        filtered_sat = df_sat.where(mask, drop=True)
        print('Valid pixel masking complete.')
        print("Original shape:", df_sat.dims)
        print("Filtered shape:", filtered_sat.dims)
    else:
        filtered_sat = df_sat
        print('No valid pixel masking applied.')

    return filtered_sat

def get_rrs(filtered_sat, sensor_ac: str):
    """
    Processes filtered satellite data to calculate Rrs based on the sensor configuration,
    and determines the sensor Rrs ID based on the sensor name.

    :param filtered_sat: The filtered satellite data.
    :param sensor_ac: The name of the sensor and AC used that determine the band names. Can be one of the following:
                      'OLCI_c2rcc', 'OLCI_polymer_rw', 'MSI_s2a_polymer_rw', 'MSI_s2b_polymer_rw'.
    :return: A tuple containing the processed Rrs data and the sensor Rrs ID.
    """
    # Access meta info about the sensors
    sensor_config = meta.get_sensor_config(sensor_ac)
    filtered_sat_rrs = meta.calculate_rrs(filtered_sat, sensor_config)

    # Mapping sensor_ac to the respective Rrs ID
    rrs_id_map = {
        'OLCI_c2rcc': 'OLCI_c2rcc_rrs',
        'OLCI_polymer_rw': 'OLCI_polymer_rrs',
        'MSI_s2a_polymer_rw': 'MSI_s2a_polymer_rrs',
        'MSI_s2b_polymer_rw': 'MSI_s2b_polymer_rrs'
    }

    # Get the sensor_rrs_id from the map based on sensor_ac
    sensor_rrs_id = rrs_id_map.get(sensor_ac)

    if sensor_rrs_id is None:
        print('No valid sensor configuration found for:', sensor_ac)
        return None, None

    return filtered_sat_rrs, sensor_rrs_id

def standardise_rrs(rrs_spectrum, data_type: str):
    """
    Standardisation as in Spyrakos et al. (2018) used for OWT flagging. 
    
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
    #elif data_type == 'satellite':
    #    alfa = (rrs_spectrum * owt_rrs).sum(dim=('lat', 'lon')) / (np.sqrt((rrs_spectrum ** 2).sum(dim=('lat', 'lon'))) * np.sqrt((owt_rrs ** 2).sum(dim=('lat', 'lon'))))
    elif data_type == 'satellite':
        alfa = np.arccos(np.dot(rrs_spectrum, owt_rrs) / (np.linalg.norm(rrs_spectrum) * np.linalg.norm(owt_rrs)))
    
    sa_owt = 1 - alfa / pi
    
    return sa_owt

def calc_max_owt(rrs_spectrum, owt_array: np.ndarray, data_type: str) -> int:
    """
    Calculate the maximum OWT class for a given Rrs spectrum.
    """
    if data_type not in ['in_situ', 'satellite']:
        raise ValueError("Invalid data_type. Expected one of: 'in_situ', 'satellite'")

    # Calculate the spectral angle (SA) for each OWT in the OWT array using the given Rrs spectrum
    all_sa = np.apply_along_axis(cal_sa, 1, owt_array, rrs_spectrum=rrs_spectrum, data_type=data_type)
    # Find the maximum SA and corresponding OWT class, +1 because python starts counting at 0
    owt_class = np.argmax(all_sa) + 1

    return owt_class

def owt_flagging(owt_rrs: pd.DataFrame, input_dataset: Union[pd.DataFrame, xr.Dataset], sensor: str, data_type: str) -> Union[pd.DataFrame, xr.Dataset]:
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

        return input_dataset
    
    elif data_type == 'satellite':
        
        print('Starting OWT classification...')
        # Select the sensor configuration for the given sensor
        sensor_bands = meta.get_sensor_config(sensor)
        
        # Create a DataArray for each band and combine them into a new Dataset
        band_data_arrays = {band: input_dataset[band] for band in sensor_bands}
        combined_dataset = xr.Dataset(band_data_arrays)

        # Calculate the area of the Rrs spectrum along each pixel directly, avoiding stacking
        rrs_area = combined_dataset.to_array(dim='band').sum(dim='band')
        
        # Standardise the Rrs spectrum by dividing each element by the Rrs area
        for band in sensor_bands:
            combined_dataset[band] /= rrs_area

        print('Standardisation complete. Bands have been standardised and added to the input dataset.')
        
        # Select OWT columns for calculating the maximum OWT membership
        # The OWT dataframe is already standardised and as such both data sources share the same standardisation

        # OWT band selection is independent of AC, but depends on sensor

        # If 'OLCI' is in 'sensor', change 'sensor' to 'OLCI_all', e.g. if it's OLCI_c2rcc
        if 'OLCI' in sensor:
            sensor = 'OLCI_all'
        elif 'MSI_s2a' in sensor:
            sensor = 'MSI_s2a'
        elif 'MSI_s2b' in sensor:
            sensor = 'MSI_s2b'

        print(sensor)
    
        # Fetch the sensor configuration based on the updated 'sensor'
        sensor_bands = meta.get_sensor_config(sensor)
        owt_cols = ['wl'] + sensor_bands
        # Select the specified columns from the owt_rrs data frame
        owt_sel = owt_rrs[owt_cols]
        owt_array = owt_sel.iloc[:, 1:].values

        # calculate OWT using the standardised pixels (input_dataset_standardised)
        
        standardised_array = combined_dataset.to_array(dim='band')

        result = xr.apply_ufunc(
            calc_max_owt,  # The function to apply
            #input_dataset_standardised,  # The DataArray to operate on - it has to be input_dataset_standardised, because this one has the 'band' value and contains the 12 Rrs bands
            standardised_array,
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


def normalise_input(input_dataset: Union[pd.DataFrame, xr.Dataset], data_type: str, sensor: str, scaler_name: str, cwd_path: str, reset_index:bool=True) -> Union[pd.DataFrame, xr.Dataset]:

    if data_type == 'in_situ':
        # Check if reset_index is False
        if not reset_index:
            # Don't reset the index of df
            pass
        else:
            # Reset the index of df
            df = input_dataset.reset_index(drop=True)
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
        scaler_fitting_cols = meta.get_scaler_config(scaler_name)
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

    elif data_type == 'satellite':

        def configure_bands(sensor):
            # Get the sensor configuration
            input_bands = meta.get_sensor_config(sensor)
            print("Bands available:", input_bands)

            # Try to parse band identifiers to see if they are numeric and within the range 400-800
            try:
                # Convert band identifiers to integers if possible
                band_wavelengths = {band: int(band.split('_')[1]) for band in input_bands if 400 <= int(band.split('_')[1]) <= 800}
                if not band_wavelengths:
                    raise ValueError("No valid numeric bands found, using predefined dictionary.")
            except ValueError:
                # Fallback to pre - defined wavelengths if parsing fails or no valid range bands are found, e.g. for C2RCC that were previously converted from rhow_1 to Rrs_1 etc.
                band_wavelengths = {
                    'Rrs_2': 413,
                    'Rrs_3': 443,
                    'Rrs_4': 490,
                    'Rrs_5': 510,
                    'Rrs_6': 560,
                    'Rrs_7': 620,
                    'Rrs_8': 665,
                    'Rrs_9': 673,
                    'Rrs_10': 681,
                    'Rrs_11': 708,
                    'Rrs_12': 753,
                    'Rrs_16': 778
                }
            
            return input_bands, band_wavelengths

        input_bands,band_wavelengths = configure_bands(sensor)

        # Filter bands based on their wavelengths
        bands_to_zero = [band for band, wavelength in band_wavelengths.items() if wavelength >= 665 and band in input_bands]
        bands_standard = [band for band, wavelength in band_wavelengths.items() if wavelength < 665 and band in input_bands]

        # Output filtered bands to verify
        print("Bands to set negatives to zero:", bands_to_zero)
        print("Standard bands:", bands_standard)

        # Ds = xarray satellite dataset
        ds = input_dataset

        # Set all data that is negative to 0 in the wavelengths > 665 nm
        if bands_to_zero:
            ds[bands_to_zero] = ds[bands_to_zero].where(ds[bands_to_zero] >= 0, 0)
        else:
            print("No bands found for setting negatives to zero.")

        if bands_standard:
            # Create a mask where values are less than zero
            negative_mask = ds[bands_standard] < 0
            
            # Check any() over the dataset dimension to determine if there are any negatives
            if negative_mask.any(dim=['lat', 'lon']).values:
                # Extract the coordinates where there are negative values
                negative_values = ds[bands_standard].where(negative_mask, drop=True)
                
                # Extract coordinates that have at least one negative value
                negative_lats = negative_values.lat.values if 'lat' in negative_values.coords else []
                negative_lons = negative_values.lon.values if 'lon' in negative_values.coords else []
                
                if negative_lats.size > 0 and negative_lons.size > 0:
                    print(f"Warning: Negative values detected in bands < 665 nm at latitudes {negative_lats} and longitudes {negative_lons}.")
                else:
                    print("No specific coordinates found. All pixels valid.")
            else:
                print("No negative values found in bands < 665 nm.")
        else:
            print("No standard bands to check for negatives.")

        # Load the normalization scaler for the BNN application (not the same as the standardization for the OWTs)
        try:
            scaler_path = f"{cwd_path}/scalers/{scaler_name}_scaler.sav"
            with open(scaler_path, 'rb') as f:
                scaler = pickle.load(f)
        except FileNotFoundError:
            print(f"Error: The scaler {scaler_path} could not be found. Please verify the path and the scaler name in sensor_meta_info.py")
            return None
        
        # Bands in the xr ds need to match the naming of the columns the scaler was fit on
        
        def create_dynamic_band_mapping(scaler_name, sensor):
            scaling_bands = meta.get_scaler_config(scaler_name)

            # Revised band mapping to correspond to dataset labels
            band_mapping = {}
            sensor_bands = meta.get_sensor_config(sensor) 
            for i, band in enumerate(sensor_bands):
                band_mapping[band] = scaling_bands[i]  # Direct mapping with proper scaling band names

            return band_mapping

        band_mapping = create_dynamic_band_mapping(scaler_name, sensor)

        # Selecting bands and renaming for scaling
        selected_bands = [band for band in ds.data_vars if band in band_mapping]
        if not selected_bands:
            print("No applicable Rrs_ bands found in the dataset.")
            raise ValueError("No applicable Rrs_ bands found in the dataset.")

        # Prepare data for scale
        stacked_data = ds[selected_bands].to_array(dim='band').stack(pixels=('lat', 'lon')).transpose('pixels', 'band')
        stacked_data_df = stacked_data.to_pandas()
        stacked_data_df.rename(columns=band_mapping, inplace=True)

        # Apply the scaler to a df to make the application explicit.
        scaled_data_df = pd.DataFrame(scaler.transform(stacked_data_df), columns=stacked_data_df.columns, index=stacked_data_df.index)

        # Rename columns back to original band names for adding to dataset
        scaled_data_df.rename(columns={v: k for k, v in band_mapping.items()}, inplace=True)

        # Convert scaled DataFrame back to xr DataArray
        scaled_data_array = xr.DataArray(scaled_data_df, dims=['pixels', 'band'], coords={'pixels': stacked_data.pixels, 'band': selected_bands})

        # Unstack the scaled data back to its original (lat, lon) structure
        unstacked_scaled_data = scaled_data_array.unstack('pixels')

        # Adding scaled data back to the dataset as new bands
        for band in selected_bands:
            new_band_name = f"scaled_{band}"
            ds[new_band_name] = unstacked_scaled_data.sel(band=band)

        # The dataset now contains the original bands along with the new scaled bands
        print("New bands added to the dataset:", [f"scaled_{band}" for band in selected_bands])

        return ds