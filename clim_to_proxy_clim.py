import sys
sys.path.append('/Users/tanaya/USC/sedproxy')

import pandas as pd
import numpy as np
import os

# Get the directory where this file (make_pfm_df.py) is located
module_dir = os.path.dirname(__file__)
stages_key_path = os.path.join(module_dir, "data", "stages_key.csv")
stages_key_df = pd.read_csv(stages_key_path)

import warnings
from bioturbation_weights import bioturbation_weights
from proxy_conversion import proxy_conversion

# ------------------------------------------------
# 1. Helper functions (unchanged from your version)
# ------------------------------------------------

def is_rapid_case(bio_depth, sed_acc_rate, layer_width, n_samples, habitat_weights):
    return (
        np.ndim(bio_depth) == 0 and
        np.ndim(sed_acc_rate) == 0 and
        np.ndim(layer_width) == 0 and
        np.ndim(n_samples) == 0 and
        isinstance(habitat_weights, np.ndarray) and habitat_weights.ndim == 1
    )

def sample_indices(prob_weights, total_samples):
    prob_weights = np.asarray(prob_weights)
    prob_weights /= prob_weights.sum()
    return np.random.choice(len(prob_weights), size=total_samples, p=prob_weights, replace=True)

# ------------------------------------------------
# 2. Main model function
# ------------------------------------------------

def clim_to_proxy_clim(
    clim_signal,
    timepoints,
    calibration_type="identity",
    calibration=None,
    slp_int_means=None,
    slp_int_vcov=None,
    noise_type=None,
    plot_sig_res=100,
    habitat_weights=None,
    habitat_wt_args=None,
    bio_depth=10,
    sed_acc_rate=50,
    layer_width=1,
    sigma_meas=0,
    sigma_ind=0,
    meas_bias=0,
    scale_noise=None,
    n_samples=np.inf,
    n_replicates=1,
    top_of_core=None,
    n_bd=3,
    stages_key = None,
    return_full = False,
):
    
    # 2.1 Prep inputs
    timepoints = np.asarray(timepoints)
    n_timepoints = len(timepoints)
    if isinstance(clim_signal, pd.DataFrame):
        clim_signal = clim_signal.values
    if top_of_core is None:
        top_of_core = 0
    
    min_clim_time = 0
    max_clim_time = clim_signal.shape[0] - 1

    # Optional smoothing output (plot_sig_res equivalent)
    if plot_sig_res is not None:
        timepoints_smoothed = np.arange(min_clim_time, max_clim_time, plot_sig_res)
        clim_signal_smoothed = chunk_matrix(timepoints_smoothed, plot_sig_res, clim_signal)
    else:
        timepoints_smoothed = None
        clim_signal_smoothed = None


    if noise_type is None:
        noise_type = "multiplicative" if calibration_type == "MgCa" else "additive"
    if scale_noise is None:
        scale_noise = calibration_type != "identity"
    if calibration is None:
        calibration = {"MgCa": "Ten planktonic species_350-500", "Uk37": "Mueller global", "identity": None}[calibration_type]

    # 2.2 Proxy conversion (assume identity)
    proxy_clim_signal = clim_signal  # identity calibration (no conversion)

    # 2.3 Simulate proxy signal
    use_rapid = is_rapid_case(bio_depth, sed_acc_rate, layer_width, n_samples, habitat_weights)

    proxy_bt, proxy_bt_sb, proxy_bt_sb_sampY, proxy_bt_sb_sampYM = [], [], [], []
    
    for i, tp in enumerate(timepoints):
        bio_ts = int(1000 * bio_depth / (sed_acc_rate if use_rapid else sed_acc_rate[i]))
        layer_yrs = int(np.ceil(1000 * layer_width / (sed_acc_rate if use_rapid else sed_acc_rate[i])))
        first_tp = -bio_ts - layer_yrs // 2
        last_tp = n_bd * bio_ts
        bioturb_window = np.arange(first_tp, last_tp + 1)

        weights = bioturbation_weights(
            z=bioturb_window,
            focal_z=0 if use_rapid else tp,
            layer_width=layer_width if use_rapid else layer_width[i],
            sed_acc_rate=sed_acc_rate if use_rapid else sed_acc_rate[i],
            bio_depth=bio_depth,
            scale="time"
        )

        hab_wts = habitat_weights / habitat_weights.sum() if use_rapid else habitat_weights[i] / habitat_weights[i].sum()
        w_season = np.outer(weights, hab_wts)
        w_season /= w_season.sum()

        proxy_window = proxy_clim_signal[(tp + bioturb_window).clip(min=0, max=proxy_clim_signal.shape[0]-1)]
        # proxy_bt.append((np.outer(weights, np.ones(proxy_clim_signal.shape[1])) * proxy_window).sum())
        # compute total bioturbation-weighted average across months
        proxy_bt.append(((np.outer(weights, np.ones(proxy_clim_signal.shape[1])) * proxy_window).sum()) / proxy_clim_signal.shape[1])


        proxy_bt_sb.append((w_season * proxy_window).sum())

        # Replace this section in your code (around line 120-130):

        # Replace the entire finite sampling section in your code (around lines 120-140):

        ns = n_samples if use_rapid else n_samples[i]
        if np.isinf(ns):
            proxy_bt_sb_sampYM.append(np.full((n_replicates,), np.nan))
            proxy_bt_sb_sampY.append(np.full((n_replicates,), np.nan))
        else:
            ns = int(ns)

            # FIXED SECTION - Replace your existing sampling code with this:
            prob_weights = w_season.flatten()
            prob_weights = prob_weights / prob_weights.sum()  # Ensure normalized
            
            # Sample indices from the correct range using np.random.choice
            samples = np.random.choice(len(prob_weights), size=ns * n_replicates, 
                                     p=prob_weights, replace=True)
            
            # Get the flattened proxy values and sample from them
            proxy_flat = proxy_window.flatten()
            sampled_vals = proxy_flat[samples]
            
            # Reshape to (n_replicates, ns) and take mean across samples (axis=1)
            vals = sampled_vals.reshape(n_replicates, ns).mean(axis=1)
            proxy_bt_sb_sampYM.append(vals)
            
            # For sampY (no seasonal aliasing), get annual means first
            annual_means = (proxy_window * hab_wts[np.newaxis, :]).sum(axis=1)
            row_indices = samples % proxy_window.shape[0]  # Get row indices
            annual_sampled = annual_means[row_indices]
            annual_vals = annual_sampled.reshape(n_replicates, ns).mean(axis=1)
            proxy_bt_sb_sampY.append(annual_vals)

    proxy_bt = np.array(proxy_bt)
    proxy_bt_sb = np.array(proxy_bt_sb)
    proxy_bt_sb_sampYM = np.array(proxy_bt_sb_sampYM)
    proxy_bt_sb_sampY = np.array(proxy_bt_sb_sampY)

    # 2.4 Noise rescaling (scale_noise block)
    sigma_ind_scaled = sigma_ind / np.sqrt(n_samples) if np.isfinite(n_samples) else 0
    sigma_meas_ind = np.sqrt(sigma_meas**2 + sigma_ind_scaled**2)

    if scale_noise:
        mean_temperature = proxy_bt  # identity case (no back conversion needed)
        sigma_meas_ind = sigma_meas_ind  # identity still in correct units

    if noise_type == "additive":
        noise = np.random.normal(0, sigma_meas_ind, size=proxy_bt_sb_sampYM.shape)
        bias = np.random.normal(0, meas_bias, size=(n_replicates,))
        proxy_bt_sb_sampYM_b = proxy_bt_sb_sampYM + bias
        proxy_bt_sb_sampYM_b_n = proxy_bt_sb_sampYM_b + noise
    else:
        noise = np.exp(np.random.normal(0, sigma_meas_ind, size=proxy_bt_sb_sampYM.shape))
        bias = np.exp(np.random.normal(0, meas_bias, size=(n_replicates,)))
        proxy_bt_sb_sampYM_b = proxy_bt_sb_sampYM * bias
        proxy_bt_sb_sampYM_b_n = proxy_bt_sb_sampYM_b * noise

    # Infinite sample bias+noise
    if noise_type == "additive":
        inf_noise = np.random.normal(0, sigma_meas_ind, size=(n_timepoints, n_replicates))
        inf_bias = np.random.normal(0, meas_bias, size=(n_replicates,))
        proxy_bt_sb_inf_b = proxy_bt_sb[:, np.newaxis] + inf_bias
        proxy_bt_sb_inf_b_n = proxy_bt_sb_inf_b + inf_noise
    else:
        inf_noise = np.exp(np.random.normal(0, sigma_meas_ind, size=(n_timepoints, n_replicates)))
        inf_bias = np.exp(np.random.normal(0, meas_bias, size=(n_replicates,)))
        proxy_bt_sb_inf_b = proxy_bt_sb[:, np.newaxis] * inf_bias
        proxy_bt_sb_inf_b_n = proxy_bt_sb_inf_b * inf_noise

    # 2.5 Select output path depending on finite vs infinite sampling
    simulated_proxy = proxy_bt_sb_sampYM_b_n if np.isfinite(n_samples) else proxy_bt_sb_inf_b_n
    recon_temp = simulated_proxy
    recon_proxy = simulated_proxy  # no calibration uncertainty under identity

    # 2.6 Build PFM dictionary (fully shape-safe)
    PFM = {
        "proxy_bt": np.squeeze(proxy_bt),
        "proxy_bt_sb": np.squeeze(proxy_bt_sb),
        "proxy_bt_sb_inf_b": np.squeeze(proxy_bt_sb_inf_b),
        "proxy_bt_sb_inf_b_n": np.squeeze(proxy_bt_sb_inf_b_n),
        "proxy_bt_sb_sampY": np.squeeze(proxy_bt_sb_sampY),
        "proxy_bt_sb_sampYM": np.squeeze(proxy_bt_sb_sampYM),
        "proxy_bt_sb_sampYM_b": np.squeeze(proxy_bt_sb_sampYM_b),
        "proxy_bt_sb_sampYM_b_n": np.squeeze(proxy_bt_sb_sampYM_b_n),
        "simulated_proxy": np.squeeze(simulated_proxy),
        "simulated_proxy_cal_err": np.squeeze(recon_proxy),
        "reconstructed_climate": np.squeeze(recon_temp),
        "timepoints": np.squeeze(timepoints),
        "n_samples": n_samples,
        "clim_signal_ann": np.squeeze(clim_signal[timepoints].mean(axis=1)),
        "clim_timepoints_ssr": None,  # Smoothed signal left out for now
        "timepoints_smoothed": None,
        "clim_signal_smoothed": None
    }

    pfm_df = make_pfm_df(PFM, stages_key=stages_key)
    
    # Build the tidy long-form dataframe:
    pfm_df = make_pfm_df(PFM, stages_key)

    # Build the simulated.proxy tibble equivalent:
    sim_df = pd.DataFrame({
    "timepoints": timepoints,
    "simulated_proxy": simulated_proxy[:, 0] if simulated_proxy.ndim == 2 else simulated_proxy,
    "reconstructed_climate": recon_temp[:, 0] if recon_temp.ndim == 2 else recon_temp
    })

    # Build smoothed.signal equivalent:
    smoothed_signal_df = pd.DataFrame({
        "timepoints": timepoints_smoothed,
        "value": clim_signal_smoothed,
        "stage": "clim_signal_smoothed"
    }) if clim_signal_smoothed is not None else None

    # Return Everything 
    return {
        "everything": pfm_df,
        "simulated_proxy": sim_df,
        "smoothed_signal": smoothed_signal_df,
        "calibration_pars": {
            "calibration_type": calibration_type,
            "calibration": calibration,
            "slp_int_means": slp_int_means
        }
    }



# ------------------------------------------------
# 3. Fully safe dataframe constructor
# ------------------------------------------------

def make_pfm_df2(pfm, stages_key):
    """
    Build dataframe safely from PFM dictionary with full broadcasting.
    """
    stage_cols = [
        "proxy_bt_sb_sampY",
        "proxy_bt_sb_sampYM",
        "proxy_bt_sb_inf_b",
        "proxy_bt_sb_sampYM_b",
        "proxy_bt_sb_inf_b_n",
        "proxy_bt_sb_sampYM_b_n",
        "simulated_proxy",
        "simulated_proxy_cal_err",
        "reconstructed_climate",
        "proxy_bt",
        "proxy_bt_sb",
        "clim_signal_ann",
        "clim_timepoints_ssr"
    ]

    # Build wide dataframe
    df = pd.DataFrame({
        col: np.squeeze(pfm[col]) for col in stage_cols
    })

    df['timepoints'] = np.squeeze(pfm['timepoints'])
    df['n_samples'] = np.repeat(pfm['n_samples'], len(df['timepoints']))
    df['replicate'] = 1

    # Melt wide to long
    df_long = df.melt(
        id_vars=['timepoints', 'n_samples', 'replicate'],
        value_vars=stage_cols,
        var_name='stage',
        value_name='value'
    )

    # Merge stages_key to bring in plotting info
    full_df = df_long.merge(stages_key, on='stage', how='left')

    return full_df

def make_pfm_df(pfm, stages_key):
    """
    Build dataframe safely from PFM dictionary with full broadcasting.
    Handles both 1D and 2D arrays properly for multiple replicates.
    """
    stage_cols = [
        "proxy_bt_sb_sampY",
        "proxy_bt_sb_sampYM", 
        "proxy_bt_sb_inf_b",
        "proxy_bt_sb_sampYM_b",
        "proxy_bt_sb_inf_b_n",
        "proxy_bt_sb_sampYM_b_n",
        "simulated_proxy",
        "simulated_proxy_cal_err",
        "reconstructed_climate",
        "proxy_bt",
        "proxy_bt_sb",
        "clim_signal_ann",
        "clim_timepoints_ssr"
    ]

    # Get dimensions
    timepoints = np.asarray(pfm['timepoints'])
    n_timepoints = len(timepoints)
    
    # Determine number of replicates by checking a 2D array
    sample_array = pfm.get('proxy_bt_sb_sampYM', pfm.get('simulated_proxy'))
    if sample_array is not None and sample_array.ndim == 2:
        n_replicates = sample_array.shape[1]
    else:
        n_replicates = 1

    # Build list of records for each replicate
    all_records = []
    
    for rep in range(n_replicates):
        for i, tp in enumerate(timepoints):
            record = {
                'timepoints': tp,
                'n_samples': pfm['n_samples'],
                'replicate': rep + 1  # 1-indexed like R
            }
            
            # Add stage values
            for col in stage_cols:
                if col in pfm and pfm[col] is not None:
                    arr = np.asarray(pfm[col])
                    if arr.ndim == 0:  # scalar
                        record[col] = float(arr)
                    elif arr.ndim == 1:  # 1D array
                        if len(arr) > i:
                            record[col] = arr[i]
                        else:
                            record[col] = np.nan
                    elif arr.ndim == 2:  # 2D array (timepoints x replicates)
                        if i < arr.shape[0] and rep < arr.shape[1]:
                            record[col] = arr[i, rep]
                        else:
                            record[col] = np.nan
                    else:
                        record[col] = np.nan
                else:
                    record[col] = np.nan
            
            all_records.append(record)
    
    # Create DataFrame from records
    df = pd.DataFrame(all_records)
    
    # Melt to long format
    df_long = df.melt(
        id_vars=['timepoints', 'n_samples', 'replicate'],
        value_vars=stage_cols,
        var_name='stage',
        value_name='value'
    )
    
    # Remove rows with NaN values
    df_long = df_long.dropna(subset=['value'])
    
    # Merge with stages_key if provided
    if stages_key is not None:
        df_long = df_long.merge(stages_key, on='stage', how='left')
    
    return df_long

def chunk_matrix(timepoints, width, climate_matrix, start_year=None):
    """
    Compute block-averaged climate values centered on each timepoint.

    Parameters
    ----------
    timepoints : list or array
        Timepoints at which to average the climate signal.
    width : int
        Width (in years) of the averaging window.
    climate_matrix : np.ndarray
        Climate signal, typically shape (years, ...).
    start_year : int or None
        If known, the start year of the climate_matrix (e.g., for annual ts). 
        If None, assumes first row = time index 0.

    Returns
    -------
    np.ndarray
        Array of averaged climate values at each timepoint.
    """
    timepoints = np.asarray(timepoints)
    rel_window = np.arange(width) - int(np.ceil(width / 2))

    n_rows = climate_matrix.shape[0]
    results = []

    for tp in timepoints:
        if start_year is not None:
            center_index = tp - start_year + 1
        else:
            center_index = tp  # Assume row index matches timepoint

        inds = rel_window + center_index
        inds = inds[(inds >= 0) & (inds < n_rows)]

        if len(inds) == 0:
            results.append(np.nan)
            continue

        block = climate_matrix[inds, :]
        results.append(np.nanmean(block))  # Drop NaNs just in case

    return np.array(results)