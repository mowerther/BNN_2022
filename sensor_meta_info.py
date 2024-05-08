import numpy as np

'''
Sensor band configurations. Currently the BNNs are available for OLCI, MSI S2A/B and at some point will be Landsat-8 OLI.

'''

sensor_bands = {
'OLCI_all': ['413', '443', '490','510', '560', '620', '665', '673', '681', '708', '753','778'],
'OLCI_c2rcc': ['rhow_2', 'rhow_3', 'rhow_4', 'rhow_5','rhow_6','rhow_7','rhow_8', 'rhow_9', 'rhow_10', 'rhow_11', 'rhow_12', 'rhow_16'],
'OLCI_c2rcc_rrs': ['Rrs_2', 'Rrs_3', 'Rrs_4', 'Rrs_5','Rrs_6','Rrs_7','Rrs_8', 'Rrs_9', 'Rrs_10', 'Rrs_11', 'Rrs_12','Rrs_16'],
'OLCI_polymer': ['413', '443', '490','510', '560', '620', '665', '681', '708', '753','778'], # does not include 673 nm
'OLCI_polymer_insitu': ['413', '443', '490','510', '560', '620', '665', '681', '708', '753','778'], # does not include 673 nm
'OLCI_polymer_rw': ['Rw413', 'Rw443', 'Rw490','Rw510', 'Rw560', 'Rw620', 'Rw665', 'Rw681', 'Rw708', 'Rw753','Rw778'], # does not include 673 nm
'MSI_s2a': ['443', '492', '560','665', '704', '739', '780'],
'MSI_s2b': ['443', '490', '560','665', '704', '740', '783'],
'MSI_s2a_polymer_rw': ['Rw443', 'Rw492', 'Rw560','Rw665', 'Rw704', 'Rw740', 'Rw783'],
'MSI_s2b_polymer_rw': ['Rw443', 'Rw490', 'Rw560','Rw665', 'Rw705', 'Rw740', 'Rw783'],
'MSI_s2a_polymer_rrs': ['Rrs_443', 'Rrs_492', 'Rrs_560', 'Rrs_665', 'Rrs_704', 'Rrs_740', 'Rrs_783'],
'MSI_s2b_polymer_rrs': ['Rrs_443', 'Rrs_490', 'Rrs_560', 'Rrs_665', 'Rrs_705', 'Rrs_740', 'Rrs_783'],
'L8_oli': ['443','482','561','655']
}

scaler_bands = {
'OLCI_all': ['413_res_s3b','443_res_s3b','490_res_s3b','510_res_s3b','560_res_s3b','620_res_s3b', '665_res_s3b','673_res_s3b','681_res_s3b','708_res_s3b','753_res_s3b','778_res_s3b'],
'OLCI_polymer':['413_res_s3b','443_res_s3b','490_res_s3b','510_res_s3b','560_res_s3b','620_res_s3b', '665_res_s3b','681_res_s3b','708_res_s3b','753_res_s3b','778_res_s3b'], # does not include 673 nm
'MSI_s2a':['443_res_s2a', '492_res_s2a', '560_res_s2a','665_res_s2a', '704_res_s2a','740_res_s2a', '783_res_s2a'],
'MSI_s2b':['443_res_s2b', '492_res_s2b','560_res_s2b','665_res_s2b', '704_res_s2b', '739_res_s2b','780_res_s2b']
}

def get_scaler_config(scaler_name):
    '''
    Get scaler configuration for the normalisation of the BNNs.
    
    Args:
        sensor_name : a sensor string of ['OLCI_all', 'OLCI_polymer', 'MSI_s2a', 'MSI_s2b'].
    
    Returns current sensor configurations for Sentinel-3 OLCI and Sentinel-2.
    '''
    scaler_config = scaler_bands[scaler_name]
    return scaler_config

def get_sensor_config(sensor_name):
    """
    Get sensor configuration for the functions and the BNNs and modify the dataset if necessary.
    
    Args:
        sensor_name (str): A sensor string, one of ['OLCI_all', 'OLCI_c2rcc', 'OLCI_polymer_rw', 'OLCI_polymer_insitu', 'MSI_s2a', 'MSI_s2b'].
        dataset (xr.Dataset or pd.DataFrame): The dataset to modify if necessary.
    
    Returns:
        xr.Dataset or pd.DataFrame: The possibly modified dataset.
    """
    sensor_config = sensor_bands[sensor_name]
        
    return sensor_config

def calculate_rrs(dataset, sensor_config):
    """
    Calculate remote sensing reflectance if necessary, e.g. for C2RCC or POLYMER.
    
    Args:
        dataset (xr.Dataset or pd.DataFrame): The dataset to modify.
        sensor_config (list): The configuration of the sensor bands.
        
    Returns:
        xr.Dataset or pd.DataFrame: The modified dataset.
    """
    modified_dataset = dataset.copy()
    for band in sensor_config:
        # Extracting numeric part from the band name
        band_suffix = ''.join(filter(str.isdigit, band))
        
        new_band_name = 'Rrs_' + band_suffix
        modified_dataset[new_band_name] = dataset[band] / np.pi
        print('Band ' + band + ' was divided by pi and stored as ' + new_band_name + '.')
    
    return modified_dataset
