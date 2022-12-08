import numpy as np

'''

Functions for evaluating the estimation performance of the BNNs.

y_true = in situ chla
y_est = BNN algorithm estimate


'''

def MAD(y_true: np.ndarray, y_est: np.ndarray) -> float:
    ''' Return the mean absolute difference (MAD) '''
    y_true = np.log10(y_true)
    y_est = np.log10(y_est)
    return 10**np.mean(np.abs(y_true - y_est))-1
		
def mapd(y_true: np.ndarray, y_est: np.ndarray) -> float:
	''' Return the median absolute percentage difference (MAPD) '''
	return np.median(np.abs((y_true - y_est) / y_true)) * 100


def mdsa(y_true: np.ndarray, y_est: np.ndarray) -> float:
	''' Return the median symmetric accuracy (MdSA) '''
	return (np.exp(np.median(np.abs(np.log(y_est / y_true)))) - 1) * 100


def sspb(y_true: np.ndarray, y_est: np.ndarray) -> float:
	''' Return the symmetric signed percentage bias (SSPB) '''
	med = np.median( np.log(y_est / y_true) )
	return np.sign(med) * (np.exp(np.abs(med)) - 1) * 100