from typing import Union
import json
import math
from math import pi
import os
os.environ['TF_USE_LEGACY_KERAS'] = '1'
from pathlib import Path

import h5py
import tensorflow as tf
import tensorflow_probability as tfp
import pandas as pd
import numpy as np
import xarray as xr
import warnings
import sensor_meta_info as meta


_SCALER_ARRAY_FIELDS = (
    'scale',
    'min',
    'data_min',
    'data_max',
    'data_range',
)


class _CompatibleDistributionLambda(tfp.layers.DistributionLambda):
    """Reconstruct the Normal output distribution stored by the original models."""

    def __init__(self, **config):
        layer_config = {
            key: config[key]
            for key in ('name', 'trainable', 'dtype')
            if key in config
        }
        super().__init__(
            make_distribution_fn=lambda parameters: tfp.distributions.Normal(
                loc=parameters[..., :1],
                scale=tf.math.exp(parameters[..., 1:]),
            ),
            convert_to_tensor_fn=tfp.distributions.Distribution.sample,
            **layer_config,
        )

    @classmethod
    def from_config(cls, config):
        return cls(**config)


def load_bnn_model(model_path):
    """Load an original BNN for inference without its legacy training state."""
    with h5py.File(model_path, 'r') as model_file:
        model_config = model_file.attrs['model_config']

    if isinstance(model_config, bytes):
        model_config = model_config.decode('utf-8')

    model_config = json.loads(model_config)
    layers = model_config['config']['layers']
    distribution_layer = layers[-1]
    output_layer = layers[-2]

    if distribution_layer['class_name'] != 'DistributionLambda':
        raise ValueError('The final BNN layer must be a DistributionLambda.')
    if distribution_layer['config'].get('convert_to_tensor_fn') != 'sample':
        raise ValueError('The final BNN distribution must return samples.')
    if output_layer['class_name'] != 'Dense':
        raise ValueError('The layer before the distribution must be Dense.')
    if output_layer['config'].get('units') != 2:
        raise ValueError('The final BNN dense layer must have two outputs.')

    distribution_layer['class_name'] = '_CompatibleDistributionLambda'
    distribution_layer['config'] = {
        key: distribution_layer['config'][key]
        for key in ('name', 'trainable', 'dtype')
    }
    bnn_model = tf.keras.models.model_from_json(
        json.dumps(model_config),
        custom_objects={
            '_CompatibleDistributionLambda': _CompatibleDistributionLambda,
        },
    )
    bnn_model.load_weights(model_path)
    return bnn_model


def load_scaler_parameters(scaler_name, cwd_path):
    """Load and validate fitted MinMaxScaler parameters from JSON."""
    parameter_path = Path(cwd_path) / 'scalers' / 'scaler_parameters.json'
    try:
        with parameter_path.open(encoding='utf-8') as parameter_file:
            document = json.load(parameter_file)
    except FileNotFoundError as error:
        raise FileNotFoundError(
            f'Scaler parameters could not be found at {parameter_path}.'
        ) from error

    if document.get('format_version') != 1:
        raise ValueError('Unsupported scaler parameter format version.')

    try:
        stored = document['scalers'][scaler_name]
    except KeyError as error:
        raise ValueError(f'Unknown scaler name: {scaler_name}.') from error

    feature_names = tuple(stored.get('feature_names', ()))
    n_features = stored.get('n_features')
    if not isinstance(n_features, int) or n_features <= 0:
        raise ValueError(f'Scaler {scaler_name} has an invalid feature count.')
    if len(feature_names) != n_features or len(set(feature_names)) != n_features:
        raise ValueError(f'Scaler {scaler_name} has invalid feature names.')
    if stored.get('dtype') != 'float64':
        raise ValueError(f'Scaler {scaler_name} must use float64 parameters.')

    feature_range = np.asarray(stored.get('feature_range'), dtype=np.float64)
    if feature_range.shape != (2,) or not np.isfinite(feature_range).all():
        raise ValueError(f'Scaler {scaler_name} has an invalid feature range.')

    parameters = {
        'feature_names': feature_names,
        'n_features': n_features,
        'feature_range': feature_range,
        'clip': bool(stored.get('clip', False)),
    }
    for field in _SCALER_ARRAY_FIELDS:
        values = np.asarray(stored.get(field), dtype=np.float64)
        if values.shape != (n_features,):
            raise ValueError(
                f'Scaler {scaler_name} field {field} must contain '
                f'{n_features} values.'
            )
        parameters[field] = values

    if not np.isfinite(parameters['scale']).all():
        raise ValueError(f'Scaler {scaler_name} contains a non-finite scale.')
    if not np.isfinite(parameters['min']).all():
        raise ValueError(f'Scaler {scaler_name} contains a non-finite offset.')
    return parameters


def apply_minmax_scaler(input_values, feature_names, scaler_parameters):
    """Apply validated fitted MinMaxScaler parameters using NumPy."""
    expected_names = tuple(scaler_parameters['feature_names'])
    supplied_names = tuple(feature_names)
    missing = [name for name in expected_names if name not in supplied_names]
    additional = [name for name in supplied_names if name not in expected_names]
    if missing:
        raise ValueError(f'Missing required scaler inputs: {missing}.')
    if additional:
        raise ValueError(f'Additional scaler inputs are not permitted: {additional}.')
    if supplied_names != expected_names:
        raise ValueError('Scaler inputs must be supplied in the stored feature order.')

    values = np.asarray(input_values, dtype=np.float64)
    if values.ndim != 2:
        raise ValueError('Scaler input must be a two-dimensional array.')
    if values.shape[1] != scaler_parameters['n_features']:
        raise ValueError(
            'Scaler input has an incorrect number of variables: '
            f"expected {scaler_parameters['n_features']}, got {values.shape[1]}."
        )

    transformed = values * scaler_parameters['scale'] + scaler_parameters['min']
    if scaler_parameters['clip']:
        transformed = np.clip(
            transformed,
            scaler_parameters['feature_range'][0],
            scaler_parameters['feature_range'][1],
        )
    return transformed


def load_datasets(sensor_name, file_name):
    """
    Loads a BNN model and data for a specified sensor, applies processing, and returns necessary outputs, automatically determining paths.
    
    :param sensor_name: Name of the sensor (e.g., 'OLCI_all', 'MSI_s2b')
    :param file_name: Name of the file to load the dataset from
    :return: Loaded model, dataset, and any other processed outputs
    """
    # Setup directory paths using the current working directory
    cwd = os.getcwd()
    cwd_system_wide = cwd
    model_directory = os.path.join(cwd_system_wide, 'bnns')
    data_directory = os.path.join(cwd_system_wide, 'data')
    
    # Load the BNN model
    model_path = f"{model_directory}/BNN_{sensor_name}.h5"
    try:
        bnn_model = load_bnn_model(model_path)
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
        dims = list(input_dataset.sizes.keys())

        # Use .data to get the underlying array from 'result' and 'owt_flag'
        input_dataset['owt_class'] = (dims, result.data)
        input_dataset['owt_flag'] = (dims, owt_flag.data)

        print('Satellite owt flagging complete. New OWT variables "owt_class" and "owt_flag" added to input dataset.')

        return input_dataset


def normalise_input(input_dataset: Union[pd.DataFrame, xr.Dataset], data_type: str, sensor: str, scaler_name: str, cwd_path: str, reset_index:bool=True) -> Union[pd.DataFrame, xr.Dataset]:

    if data_type == 'in_situ':
        df = input_dataset
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
        missing_input_bands = [band for band in input_bands if band not in df.columns]
        if missing_input_bands:
            raise ValueError(
                f'Missing required model input variables: {missing_input_bands}.'
            )
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
        non_negative = df[input_bands].copy()
        non_negative.loc[:, zeroed.columns] = zeroed

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

        scaler_parameters = load_scaler_parameters(scaler_name, cwd_path)
        if len(input_bands) != scaler_parameters['n_features']:
            raise ValueError(
                f'Sensor {sensor} provides {len(input_bands)} model inputs, but '
                f"scaler {scaler_name} requires {scaler_parameters['n_features']}."
            )
        normalised = apply_minmax_scaler(
            non_negative.to_numpy(dtype=np.float64),
            scaler_parameters['feature_names'],
            scaler_parameters,
        )

        # Create the suffix for the normalised wavelengths
        string_norm = [str(int) + '_norm' for int in input_bands]

        # Create the DataFrame with the normalised columns
        df_norm = pd.DataFrame(
            data=normalised,
            columns=string_norm,
            index=df.index,
        )

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

        scaler_parameters = load_scaler_parameters(scaler_name, cwd_path)
        sensor_bands = meta.get_sensor_config(sensor)
        if len(sensor_bands) != scaler_parameters['n_features']:
            raise ValueError(
                f'Sensor {sensor} provides {len(sensor_bands)} model inputs, but '
                f"scaler {scaler_name} requires {scaler_parameters['n_features']}."
            )
        missing_input_bands = [band for band in sensor_bands if band not in ds]
        if missing_input_bands:
            raise ValueError(
                f'Missing required model input variables: {missing_input_bands}.'
            )

        # Construct model inputs in the stored scaler order, independently of
        # the order of variables in the source Dataset.
        selected_bands = list(sensor_bands)
        band_mapping = dict(
            zip(selected_bands, scaler_parameters['feature_names'])
        )

        # Prepare data for scale
        stacked_data = ds[selected_bands].to_array(dim='band').stack(pixels=('lat', 'lon')).transpose('pixels', 'band')
        stacked_data_df = stacked_data.to_pandas()
        stacked_data_df.rename(columns=band_mapping, inplace=True)

        scaled_data_df = pd.DataFrame(
            apply_minmax_scaler(
                stacked_data_df.to_numpy(dtype=np.float64),
                stacked_data_df.columns,
                scaler_parameters,
            ),
            columns=stacked_data_df.columns,
            index=stacked_data_df.index,
        )

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
