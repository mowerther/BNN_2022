'''
Sensor band configurations. Currently the BNNs are available for OLCI, MSI S2A/B and at some point will be Landsat-8 OLI.

'''

sensor_bands = {
'OLCI_all': ['413', '443', '490','510', '560', '620', '665', '673', '681', '708', '753','778'],
'OLCI_polymer': ['413', '443', '490','510', '560', '620', '665', '681', '708', '753','778'], # does not include 673 nm
'MSI_s2a': ['443', '492', '560','665', '704', '740', '783'],
'MSI_s2b': ['443', '492', '560','665', '704', '739', '780'],
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
    scaler_config = sensor_bands[scaler_name]
    return scaler_config

def get_sensor_config(sensor_name):
    '''
    Get sensor configuration for the functions and the BNNs.
    
    Args:
        sensor_name : a sensor string of ['OLCI_all', 'OLCI_polymer', 'MSI_s2a', 'MSI_s2b'].
    
    Returns current sensor configurations for Sentinel-3 OLCI and Sentinel-2.
    '''
    sensor_config = sensor_bands[sensor_name]
    return sensor_config