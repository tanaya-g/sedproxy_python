
import numpy as np
from scipy.stats import multivariate_normal


def proxy_conversion(temp=None, proxy_val=None, calib_type=None,
                     slp_int_mean=None, slp_int_vcov=None,
                     calib=None,point_or_sample="point", n=1,
                     calib_parameters=None):
    """
    Convert temperature to proxy value (or back) using sedproxy calibrations.

    Parameters:
    ----------
    temp : array-like or None
        Input temperature values.
    proxy_val : array-like or None
        Input proxy values (inverse conversion).
    calib_type : str
        One of 'identity', 'MgCa', 'Uk37'.
    slp_int_means : array-like or None
        User-supplied [slope, intercept] means.
    slp_int_vcov : array-like or None
        User-supplied 2x2 variance-covariance matrix.
    calib : str or None
        Calibration dataset name (defaults handled in R code).
    point_or_sample : str
        'point' for deterministic, 'sample' for sampling calibration parameters.
    n : int
        Number of replicates if sampling.
    calib_parameters : DataFrame or dict
        Calibration parameters table with slope, intercept, vcov, etc.
    """
     
    # checks for correct input type
    if (temp is None and proxy_val is None) or (temp is not None and proxy_val is not None):
     raise ValueError("One and only one of temp or proxy_val should be supplied")

    if point_or_sample not in ["point", "sample"]:
        raise ValueError("point_or_sample must be 'point' or 'sample'")

    # checks for correct array size
    if isinstance(temp, np.ndarray) and temp.ndim == 2:
        if point_or_sample == "sample" and temp.shape[1] != n:
            raise ValueError("For matrix input and 'sample' mode, n must equal number of columns in input")

    if point_or_sample == "point" and n > 1:
        raise ValueError("Multiple replicates only allowed if point_or_sample == 'sample'")

    # Get calibration parameters
    if calib_type != "identity":
        if calib_parameters is None:
            raise ValueError("Calibration parameters must be provided")

        # Extract relevant parameters (simulate dataframe filtering)
        cfs_vcov = calib_parameters[
            (calib_parameters['calibration_type'] == calib_type) &
            (calib_parameters['calibration'] == calib)
        ].iloc[0]

        if slp_int_mean is None:
            cfs = np.array([[cfs_vcov['slope'], cfs_vcov['intercept']]])
        else:
            cfs = np.array([slp_int_mean])

        if slp_int_vcov is None:
            vcov = cfs_vcov['vcov']
        else:
            vcov = slp_int_vcov

        # Sample calibration parameters if requested
        if point_or_sample == "sample":
            if slp_int_mean is not None and slp_int_vcov is None:
                print("Warning: Sampling calibration parameters with user-supplied means but default VCOV.")
            cfs = multivariate_normal.rvs(mean=cfs.flatten(), cov=vcov, size=n)
            cfs = cfs.reshape((n, 2))

    # Prepare input arrays
    if isinstance(temp, (list, np.ndarray)):
        temp = np.atleast_2d(temp)
    if isinstance(proxy_val, (list, np.ndarray)):
        proxy_val = np.atleast_2d(proxy_val)

    # Ensure temp and proxy_val are 2D column vectors
    if temp is not None:
        temp = np.atleast_2d(temp).reshape(-1, 1)
    if proxy_val is not None:
        proxy_val = np.atleast_2d(proxy_val).reshape(-1, 1)


    # Conversion logic
    if calib_type == "identity":
        out = proxy_val if temp is None else temp


    elif calib_type == "MgCa":
        cfs[:, 1] = np.exp(cfs[:, 1])  # exponentiate intercept

        if temp is not None:
            temp = temp.reshape(-1, 1)  # shape (3, 1)

        if proxy_val is None:
            slope = cfs[:, 0].reshape(1, -1)      # shape (1, n)
            intercept = cfs[:, 1].reshape(1, -1)   # shape (1, n)
            out = intercept * np.exp(slope * temp)  # shape (3, n)
        else:
            proxy_val = proxy_val.reshape(-1, 1)
            slope = cfs[:, 0].reshape(1, -1)
            intercept = cfs[:, 1].reshape(1, -1)
            out = np.log(proxy_val / intercept) / slope

            
    elif calib_type == "Uk37":
        if proxy_val is None:
            out = (cfs[:, 1] + temp.T * cfs[:, 0]).T
        else:
            out = ((proxy_val.T - cfs[:, 1]) / cfs[:, 0]).T

    else:
        raise ValueError(f"Unknown calibration_type: {calib_type}")

    # Simplify output if it was vector input and n == 1
    if out.shape[1] == 1:
        out = out.flatten()

    return out


