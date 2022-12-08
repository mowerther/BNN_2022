from typing import Tuple
import numpy as np
import uncertainty_toolbox as uct

'''
Uncertainty calibration metrics: Percentage Interval Coverage Probability (PICP), Sharpness, Mean Absolute Calibration Difference (MACD). 
Requires estimated chla conc. values and standard deviations for the test set.
Functions are as follows:

    sharpness(): This function calculates the sharpness metric, which is a single scalar value that quantifies the average of the standard deviations of the estimated values of y. 
    The function calculates the sharpness metric by first taking the square root of the mean of the squares of the standard deviations, and then returning the result.

    calc_ci(): This function calculates the 95% confidence interval (CI) for the estimated values of y. 
    The confidence interval is a range of values that is expected to contain the true values of y with a probability of 95%. 
    The function calculates the 95% CI by using the get_prediction_interval() function from the uncertainty_toolbox module, 
    which takes the estimated values of y and their standard deviations as input and returns the lower and upper bounds of the 95% CI. 
    The function then returns the difference between the upper and lower bounds of the CI, as well as the upper and lower bounds themselves.
    
    picp(): This function calculates the percentage interval coverage probability (PICP), which is a percentage that quantifies the amount of true values of y (as given by y_true) that lie
    within the estimated 95% confidence intervals. 
    The function calculates the PICP by first counting the number of true values that are within the 95% CIs and the number that are outside of the CIs, 
    then dividing each count by the total number of true values and multiplying by 100 to express the result as a percentage. 
    The function returns both the percentage of observations that are within the 95% CIs and the percentage that are outside of the CIs.

    macd(): This function calculates the mean absolute calibration difference (MACD), which is a measure of the average absolute difference between the true values of y 
    and the estimated values of y within their 95% confidence intervals. 
    The function calculates the MACD by using the mean_absolute_calibration_error() function from the uncertainty_toolbox module, 
    which takes the estimated values of y, their standard deviations, and the true values of y as input and returns the MACD.

As mentioned, this uses the uncertainty quantification toolbox: https://github.com/uncertainty-toolbox/uncertainty-toolbox
'''

def sharpness(y_std: np.ndarray) -> float:
    '''
    Calculate the sharpness of a set of standard deviation estimates.

    Args:
        y_std: a 1D NumPy array of the estimated standard deviations for a test dataset. The array must be flat and contain only positive values.

    Returns:
        a single scalar value representing the sharpness of the standard deviation estimates.
    '''
    # Check that the input array is flat and contains only positive values
    if y_std.ndim != 1:
        raise ValueError('Input array must be flat (1D)')
    if np.any(y_std <= 0):
        raise ValueError('Input array must contain only positive values')

    # Calculate the sharpness metric
    sharpness_metric = np.sqrt(np.mean(y_std**2))

    return sharpness_metric

def calc_ci(y_est, y_std):
    '''
    Return 95% confidence interval (CI).
    Args: 
        y_est: 1D array of the estimated chla values. Array must be flat.
        y_std: 1D array of the estimated standar deviations. Array must be flat and positive.
    
    Returns the lower and upper confidence bounds and the difference between them (the width of the confidence interval).

    '''
    if y_std.ndim != 1:
        raise ValueError('Input array must be flat (1D)')
    if np.any(y_std <= 0):
        raise ValueError('Input array must contain only positive values')

    orig_bounds = uct.metrics_calibration.get_prediction_interval(y_est, y_std, 0.95, None)

    dif_org_bounds = orig_bounds.upper - orig_bounds.lower
 
    return dif_org_bounds, orig_bounds.upper, orig_bounds.lower

def picp(low_ci: np.ndarray, y_true: np.ndarray, up_ci: np.ndarray) -> Tuple[float, float]:
    """
    Return the Percentage Interval Coverage Probability (PICP):
    a percentage that quantifies the amount of in situ chla reference values that lay within the BNN estimated confidence intervals (CIs).
    
    Args:
        low_ci: 1D array of floats - the lower confidence interval values.
        y_true: 1D array of floats - the in situ reference chla data.
        up_ci: 1D array of floats - the upper confidence interval values.

    Returns:
        A tuple of floats representing the percentages of observations in and outside of the CIs.
    """
    # Count the number of elements in y_true that are greater than or equal to the corresponding element in low_ci and less than or equal to the corresponding element in up_ci.
    count_in = np.sum(np.greater_equal(y_true, low_ci) & np.less_equal(y_true, up_ci))

    # Count the number of elements in y_true that are outside of the confidence interval.
    count_out = len(y_true) - count_in

    # Compute the percentage of elements in y_true that are inside of the confidence interval.
    perc_in = count_in / len(y_true) * 100

    # Compute the percentage of elements in y_true that are outside of the confidence interval.
    perc_out = count_out / len(y_true) * 100

    # Return the percentages of elements in and outside of the confidence interval.
    return perc_in, perc_out

def macd(y_est: np.ndarray, y_true: np.ndarray, y_std: np.ndarray) -> float:
    """
    Return the Mean Absolute Calibration Difference (MACD)
    
    Args:
        y_est: 1D array of floats - the estimated chla data.
        y_true: 1D array of floats - the true in situ reference chla data.
        y_std: 1D array of floats - the estimated standard deviations.

    Returns:
        A float representing the MACD value.
    """
    # Compute the MACD value.
    macd_cal = uct.mean_absolute_calibration_error(y_est, y_std, y_true)

    # Return the MACD value.
    return macd_cal