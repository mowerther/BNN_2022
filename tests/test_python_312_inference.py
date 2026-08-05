import os
os.environ['TF_USE_LEGACY_KERAS'] = '1'

from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest
import tensorflow as tf
import xarray as xr


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT))

from data_processing import (
    apply_minmax_scaler,
    get_rrs,
    load_bnn_model,
    load_scaler_parameters,
    normalise_input,
    owt_flagging,
    valid_pixel_classif,
)
from model_functions import calculate_uncertainty, estimate_chla
import sensor_meta_info as meta


MODEL_SPECS = (
    (
        'OLCI_all',
        12,
        ('413', '443', '490', '510', '560', '620', '665', '673',
         '681', '708', '753', '778'),
        ('413_res_s3b', '443_res_s3b', '490_res_s3b', '510_res_s3b',
         '560_res_s3b', '620_res_s3b', '665_res_s3b', '673_res_s3b',
         '681_res_s3b', '708_res_s3b', '753_res_s3b', '778_res_s3b'),
        (0.043750699488527785, 0.024728171937239887,
         0.020074243229843837, 0.017881189237596472,
         0.012322840107881395, 0.009453623020796501,
         0.010078227209049977, 0.01334561847985418,
         0.014816684588705106, 0.006537785613174208, 0.0, 0.0),
        17.243401244748796,
    ),
    (
        'OLCI_polymer',
        11,
        ('413', '443', '490', '510', '560', '620', '665', '681',
         '708', '753', '778'),
        ('413_res_s3b', '443_res_s3b', '490_res_s3b', '510_res_s3b',
         '560_res_s3b', '620_res_s3b', '665_res_s3b', '681_res_s3b',
         '708_res_s3b', '753_res_s3b', '778_res_s3b'),
        (0.043750699488527785, 0.024728171937239887,
         0.020074243229843837, 0.017881189237596472,
         0.012322840107881395, 0.009453623020796501,
         0.010078227209049977, 0.014816684588705106,
         0.006537785613174208, 0.0, 0.0),
        16.361676974208706,
    ),
    (
        'MSI_s2a',
        7,
        ('443', '492', '560', '665', '704', '739', '780'),
        ('443_res_s2a', '492_res_s2a', '560_res_s2a', '665_res_s2a',
         '704_res_s2a', '740_res_s2a', '783_res_s2a'),
        (0.02471693626464859, 0.01948465075727696,
         0.012548410092062517, 0.009594920586974375,
         0.008867869339005787, 0.018821291431934804, 0.0),
        8.416353276090181,
    ),
    (
        'MSI_s2b',
        7,
        ('443', '490', '560', '665', '704', '740', '783'),
        ('443_res_s2b', '492_res_s2b', '560_res_s2b', '665_res_s2b',
         '704_res_s2b', '739_res_s2b', '780_res_s2b'),
        (0.02491323789212969, 0.019580484865051692,
         0.012572063538167484, 0.009646241265467327,
         0.008816498821443081, 0.02273513198498852, 0.0),
        8.398518670607846,
    ),
)


REFERENCE_DISTRIBUTIONS = {
    'OLCI_all': (8.4100008, 2.3577008, 4.3898440, 8.1546993, 14.0834017),
    'OLCI_polymer': (7.3478837, 2.2174129, 3.7451288, 7.1727324,
                     12.2013146),
    'MSI_s2a': (9.1215410, 2.6794763, 4.6766288, 8.7436104, 15.0222026),
    'MSI_s2b': (8.5676746, 2.4316235, 4.4963952, 8.2844324, 14.7720873),
}


SATELLITE_CASES = (
    (
        'OLCI_all',
        'OLCI_c2rcc',
        'subset_0_of_Mosaic_L2C2RCC_NA_S3A_OL_1_EFR_20200806__NT.nc',
        True,
        -32768,
        'pixel_classif_flags',
        (150, 290),
        37692,
        5808,
        (0.1543426215648651, 0.13049441576004028,
         0.11774247139692307, 0.1209624633193016,
         0.12213404476642609, 0.25900912284851074,
         0.3153928518295288, 0.3141426146030426,
         0.2993938624858856, 0.3624565005302429,
         1.181978702545166, 1.2505221366882324),
    ),
    (
        'MSI_s2b',
        'MSI_s2b_polymer_rw',
        'L2POLY_reproj_NASA_S2B_MSIL1C_20231220T103349_'
        'N0510_R108_T31TGN_20231220T112131.SAFE.nc',
        False,
        None,
        None,
        (15, 15),
        0,
        225,
        (0.3978220522403717, 0.35810983180999756,
         0.38017088174819946, 0.5312719941139221,
         0.3384045660495758, 0.5105759501457214,
         0.5838277339935303),
    ),
)


@pytest.mark.parametrize(
    'sensor, feature_count, sensor_bands, scaler_bands, first_row, total',
    MODEL_SPECS,
)
def test_models_load_with_expected_topology_and_stochastic_predictions(
    sensor, feature_count, sensor_bands, scaler_bands, first_row, total
):
    model = load_bnn_model(REPOSITORY_ROOT / 'bnns' / f'BNN_{sensor}.h5')

    assert tuple(meta.get_sensor_config(sensor)) == sensor_bands
    assert model.input_shape == (None, feature_count)
    assert model.output_shape == (None, 1)
    assert len(model.layers) == 13
    assert [layer.__class__.__name__ for layer in model.layers] == [
        'InputLayer',
        'Dense', 'Dropout',
        'Dense', 'Dropout',
        'Dense', 'Dropout',
        'Dense', 'Dropout',
        'Dense', 'Dropout',
        'Dense', '_CompatibleDistributionLambda',
    ]
    assert model.layers[-1].name == 'normal_exp'
    assert all(np.isfinite(weight).all() for weight in model.get_weights())

    tf.random.set_seed(20220804)
    prediction_sample = model.predict(
        np.full((1000, feature_count), 0.5, dtype=np.float32),
        batch_size=1000,
        verbose=0,
    ).reshape(-1)
    assert prediction_sample.dtype == np.float32
    assert np.isfinite(prediction_sample).all()
    distribution_summary = (
        prediction_sample.mean(),
        prediction_sample.std(),
        np.quantile(prediction_sample, 0.025),
        np.median(prediction_sample),
        np.quantile(prediction_sample, 0.975),
    )
    np.testing.assert_allclose(
        distribution_summary,
        REFERENCE_DISTRIBUTIONS[sensor],
        rtol=1e-5,
        atol=1e-4,
    )

    inputs = np.full((2, feature_count), 0.5, dtype=np.float32)
    first_prediction = model.predict(inputs, verbose=0)
    second_prediction = model.predict(inputs, verbose=0)
    assert not np.array_equal(first_prediction, second_prediction)


@pytest.mark.parametrize(
    'sensor, feature_count, sensor_bands, expected_names, first_row, total',
    MODEL_SPECS,
)
def test_stored_scaler_parameters_and_numpy_transform(
    sensor, feature_count, sensor_bands, expected_names, first_row, total
):
    parameters = load_scaler_parameters(sensor, REPOSITORY_ROOT)
    assert parameters['n_features'] == feature_count
    assert parameters['feature_names'] == expected_names
    assert parameters['scale'].dtype == np.float64
    assert parameters['min'].dtype == np.float64

    cases = np.vstack((
        parameters['data_min'],
        parameters['data_max'],
        parameters['data_min'] - 0.1 * parameters['data_range'],
        parameters['data_max'] + 0.1 * parameters['data_range'],
        np.zeros(feature_count),
    ))
    nan_cases = np.tile(parameters['data_min'], (feature_count, 1))
    nan_cases[np.arange(feature_count), np.arange(feature_count)] = np.nan
    cases = np.vstack((cases, nan_cases))
    transformed = apply_minmax_scaler(cases, expected_names, parameters)

    np.testing.assert_allclose(transformed[0], 0.0, atol=1e-15, rtol=0)
    np.testing.assert_allclose(transformed[1], 1.0, atol=3e-16, rtol=0)
    np.testing.assert_allclose(transformed[2], -0.1, atol=3e-16, rtol=0)
    np.testing.assert_allclose(transformed[3], 1.1, atol=5e-16, rtol=0)
    np.testing.assert_allclose(
        transformed[4], parameters['min'], atol=0, rtol=0
    )
    assert np.array_equal(np.isnan(transformed[5:]), np.isnan(nan_cases))


def test_scaler_transform_rejects_missing_additional_and_reordered_inputs():
    parameters = load_scaler_parameters('MSI_s2b', REPOSITORY_ROOT)
    names = parameters['feature_names']
    values = np.zeros((1, len(names)))

    with pytest.raises(ValueError, match='Missing required scaler inputs'):
        apply_minmax_scaler(values[:, :-1], names[:-1], parameters)
    with pytest.raises(ValueError, match='Additional scaler inputs'):
        apply_minmax_scaler(
            np.column_stack((values, np.zeros(1))),
            names + ('unexpected',),
            parameters,
        )
    with pytest.raises(ValueError, match='stored feature order'):
        apply_minmax_scaler(values, names[::-1], parameters)


@pytest.mark.parametrize(
    'sensor, feature_count, sensor_bands, scaler_bands, first_row, total',
    MODEL_SPECS,
)
def test_all_in_situ_examples_run_end_to_end(
    sensor, feature_count, sensor_bands, scaler_bands, first_row, total
):
    source = pd.read_csv(REPOSITORY_ROOT / 'data' / 'example_data.csv')
    owt_source = pd.read_csv(
        REPOSITORY_ROOT / 'data' /
        'spyrakos_owts_inland_waters_standardised.csv'
    )
    flagged = owt_flagging(
        owt_source, source, sensor=sensor, data_type='in_situ'
    )
    preprocessed = normalise_input(
        flagged,
        data_type='in_situ',
        sensor=sensor,
        scaler_name=sensor,
        cwd_path=str(REPOSITORY_ROOT),
    )

    normalised_columns = [f'{band}_norm' for band in sensor_bands]
    assert preprocessed['owt_class'].iloc[0] == 3
    assert (preprocessed['owt_flag'] == 0).all()
    assert (preprocessed['is_negative'] == 0).all()
    np.testing.assert_allclose(
        preprocessed.loc[0, normalised_columns], first_row, atol=1e-15, rtol=0
    )
    assert preprocessed[normalised_columns].to_numpy().sum() == pytest.approx(
        total, abs=1e-14
    )

    model = load_bnn_model(REPOSITORY_ROOT / 'bnns' / f'BNN_{sensor}.h5')
    tf.random.set_seed(20220804)
    predicted = estimate_chla(
        preprocessed,
        data_type='in_situ',
        bnn_model=model,
        sensor=sensor,
        variants=4,
        parallelise=False,
    )
    completed = calculate_uncertainty(predicted, sensor, 'in_situ')
    for suffix in ('', '_CI_width', '_CI_upper', '_CI_lower',
                   '_uncertainty_percent'):
        assert np.isfinite(completed[f'BNN_chla_{sensor}{suffix}']).all()
    assert np.isfinite(completed[f'BNN_std_{sensor}']).all()
    assert (completed[f'BNN_std_{sensor}'] > 0).all()


def test_in_situ_negative_and_missing_value_handling():
    sensor = 'OLCI_all'
    source = pd.read_csv(REPOSITORY_ROOT / 'data' / 'example_data.csv').head(1)
    source.loc[0, '443'] = -0.001
    with pytest.warns(UserWarning, match='negative values'):
        result = normalise_input(
            source,
            data_type='in_situ',
            sensor=sensor,
            scaler_name=sensor,
            cwd_path=str(REPOSITORY_ROOT),
        )
    assert result.loc[0, 'is_negative'] == 1

    owt_source = pd.read_csv(
        REPOSITORY_ROOT / 'data' /
        'spyrakos_owts_inland_waters_standardised.csv'
    )
    missing_value = source.copy()
    missing_value.loc[0, '413'] = np.nan
    with pytest.raises(ValueError, match='contains NaN values'):
        owt_flagging(
            owt_source,
            missing_value,
            sensor=sensor,
            data_type='in_situ',
        )

    missing_variable = source.drop(columns=['413'])
    with pytest.raises(ValueError, match='Missing required model input variables'):
        normalise_input(
            missing_variable,
            data_type='in_situ',
            sensor=sensor,
            scaler_name=sensor,
            cwd_path=str(REPOSITORY_ROOT),
        )


@pytest.mark.filterwarnings('ignore:The return type of `Dataset.dims`')
@pytest.mark.parametrize(
    'model_sensor, sensor_ac, file_name, apply_mask, vp_id, valid_pixel_band, '
    'expected_shape, expected_owt_flag_sum, expected_valid_count, first_scaled',
    SATELLITE_CASES,
)
def test_satellite_examples_run_end_to_end(
    model_sensor,
    sensor_ac,
    file_name,
    apply_mask,
    vp_id,
    valid_pixel_band,
    expected_shape,
    expected_owt_flag_sum,
    expected_valid_count,
    first_scaled,
):
    owt_source = pd.read_csv(
        REPOSITORY_ROOT / 'data' /
        'spyrakos_owts_inland_waters_standardised.csv'
    )

    with xr.open_dataset(REPOSITORY_ROOT / 'data' / file_name) as satellite:
        filtered = valid_pixel_classif(
            satellite,
            apply_mask=apply_mask,
            vp_id=vp_id,
            valid_pixel_band=valid_pixel_band,
        )
        reflectance, sensor = get_rrs(filtered, sensor_ac)
        flagged = owt_flagging(
            owt_source, reflectance, sensor=sensor, data_type='satellite'
        )
        preprocessed = normalise_input(
            flagged,
            data_type='satellite',
            sensor=sensor,
            scaler_name=model_sensor,
            cwd_path=str(REPOSITORY_ROOT),
        )

        assert tuple(preprocessed.sizes[dim] for dim in ('lat', 'lon')) == \
            expected_shape
        assert int(preprocessed['owt_flag'].sum()) == expected_owt_flag_sum
        scaled_names = [
            f'scaled_{band}' for band in meta.get_sensor_config(sensor)
        ]
        assert all(name in preprocessed for name in scaled_names)
        scaled = preprocessed[scaled_names].to_array().values
        first_lat, first_lon = np.argwhere(np.isfinite(scaled).all(axis=0))[0]
        np.testing.assert_allclose(
            scaled[:, first_lat, first_lon], first_scaled, atol=6e-8, rtol=0
        )

        model = load_bnn_model(
            REPOSITORY_ROOT / 'bnns' / f'BNN_{model_sensor}.h5'
        )
        tf.random.set_seed(20220804)
        predicted = estimate_chla(
            preprocessed,
            data_type='satellite',
            bnn_model=model,
            sensor=sensor,
            variants=2,
            parallelise=True,
            num_cpus=1,
        )
        completed = calculate_uncertainty(predicted, sensor, 'satellite')

        output_names = (
            f'BNN_chla_{sensor}',
            f'BNN_std_{sensor}',
            f'BNN_chla_{sensor}_CI_width',
            f'BNN_chla_{sensor}_CI_upper',
            f'BNN_chla_{sensor}_CI_lower',
            f'BNN_chla_{sensor}_uncertainty_percent',
        )
        for output_name in output_names:
            assert completed[output_name].dims == ('lat', 'lon')
            assert completed[output_name].dtype == np.float64
        chlorophyll = completed[f'BNN_chla_{sensor}'].values
        standard_deviation = completed[f'BNN_std_{sensor}'].values
        valid = np.isfinite(chlorophyll)
        assert chlorophyll.shape == expected_shape
        assert int(valid.sum()) == expected_valid_count
        assert np.isfinite(standard_deviation[valid]).all()
        assert np.any(standard_deviation[valid] > 0)
