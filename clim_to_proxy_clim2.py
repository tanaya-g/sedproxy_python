import sys
sys.path.append('/Users/tanaya/Documents/GitHub/sedproxy_python')

import pandas as pd
import numpy as np
import os

# Get the directory where this file (make_pfm_df.py) is located
module_dir = os.path.dirname(__file__)
stages_key_path = os.path.join(module_dir, "data", "stages_key.csv")
stages_key_df = pd.read_csv(stages_key_path)

import warnings
from bioturbation_weights_old import bioturbation_weights
from proxy_conversion import proxy_conversion

# ------------------------------------------------
# 1. Helper functions
# ------------------------------------------------

def is_rapid_case(bio_depth, sed_acc_rate, layer_width, n_samples, habitat_weights):
    """
    Check if we can use the rapid computation path.
    All parameters must be scalars and habitat_weights must be 1D.
    """
    return (
        np.ndim(bio_depth) == 0 and
        np.ndim(sed_acc_rate) == 0 and
        np.ndim(layer_width) == 0 and
        np.ndim(n_samples) == 0 and
        isinstance(habitat_weights, np.ndarray) and 
        habitat_weights.ndim == 1
    )

def sample_indices(prob_weights, total_samples):
    prob_weights = np.asarray(prob_weights)
    prob_weights /= prob_weights.sum()
    return np.random.choice(len(prob_weights), size=total_samples, p=prob_weights, replace=True)

# ------------------------------------------------
# 2. Main model function with R-style preprocessing
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
    stages_key=None,
    return_full=False,
):
    
    # ========================================
    # SECTION 1: Input validation and preprocessing (R-style)
    # ========================================
    
    # Convert timepoints and get basic info
    timepoints = np.asarray(timepoints, dtype=int)  # Force integer conversion
    n_timepoints = len(timepoints)
    
    if isinstance(clim_signal, pd.DataFrame):
        clim_signal = clim_signal.values
    
    if top_of_core is None:
        top_of_core = 0
    
    # Check input dimensions
    if not isinstance(clim_signal, np.ndarray) or clim_signal.ndim != 2:
        raise ValueError("clim_signal must be a 2D array or DataFrame")
    
    min_clim_time = 0
    max_clim_time = clim_signal.shape[0] - 1
    
    # Handle habitat weights - ensure it's a numpy array and properly structured
    if habitat_weights is None:
        habitat_weights = np.ones(clim_signal.shape[1]) / clim_signal.shape[1]
    else:
        habitat_weights = np.asarray(habitat_weights)
        
        # If 2D, flatten to 1D (assuming it should be seasonal weights)
        if habitat_weights.ndim == 2:
            if habitat_weights.shape[0] == 1:
                habitat_weights = habitat_weights.flatten()
            elif habitat_weights.shape[1] == 1:
                habitat_weights = habitat_weights.flatten()
            else:
                warnings.warn("habitat_weights is 2D but not obviously reducible to 1D. Using first row.")
                habitat_weights = habitat_weights[0, :]
        
        # Validate length matches climate signal columns
        if len(habitat_weights) != clim_signal.shape[1]:
            raise ValueError(f"habitat_weights length ({len(habitat_weights)}) must match "
                           f"number of climate signal columns ({clim_signal.shape[1]})")
    
    # ========================================
    # SECTION 2: Parameter preprocessing (like R version)
    # ========================================
    
    # Convert scalar parameters to arrays if needed - this is the key R-style preprocessing
    if np.ndim(sed_acc_rate) == 0:
        sed_acc_rate_array = np.full(n_timepoints, sed_acc_rate)
    else:
        sed_acc_rate_array = np.asarray(sed_acc_rate)
        if len(sed_acc_rate_array) != n_timepoints:
            raise ValueError("sed_acc_rate array length must match timepoints length")
    
    if np.ndim(layer_width) == 0:
        layer_width_array = np.full(n_timepoints, layer_width)
    else:
        layer_width_array = np.asarray(layer_width)
        if len(layer_width_array) != n_timepoints:
            raise ValueError("layer_width array length must match timepoints length")
    
    if np.ndim(n_samples) == 0:
        n_samples_array = np.full(n_timepoints, n_samples)
    else:
        n_samples_array = np.asarray(n_samples)
        if len(n_samples_array) != n_timepoints:
            raise ValueError("n_samples array length must match timepoints length")
    
    if np.ndim(sigma_meas) == 0:
        sigma_meas_array = np.full(n_timepoints, sigma_meas)
    else:
        sigma_meas_array = np.asarray(sigma_meas)
        if len(sigma_meas_array) != n_timepoints:
            raise ValueError("sigma_meas array length must match timepoints length")
    
    if np.ndim(sigma_ind) == 0:
        sigma_ind_array = np.full(n_timepoints, sigma_ind)
    else:
        sigma_ind_array = np.asarray(sigma_ind)
        if len(sigma_ind_array) != n_timepoints:
            raise ValueError("sigma_ind array length must match timepoints length")
    
    # ========================================
    # SECTION 3: Bounds checking and warnings (R-style)
    # ========================================
    
    # Calculate bioturbation windows for bounds checking
    bio_depth_timesteps = np.round(1000 * bio_depth / sed_acc_rate_array).astype(int)
    layer_width_years = np.ceil(1000 * layer_width_array / sed_acc_rate_array).astype(int)
    
    # Check if bioturbation windows extend beyond climate signal
    max_windows = timepoints + n_bd * bio_depth_timesteps
    min_windows = timepoints - bio_depth_timesteps - layer_width_years // 2
    
    # Identify problematic timepoints
    too_old = max_windows >= max_clim_time
    too_young = timepoints < top_of_core
    mixed_layer = min_windows < min_clim_time
    
    # Filter valid timepoints and give warnings like R version
    valid_inds = (~too_old) & (~too_young)
    
    if np.any(too_old):
        warnings.warn(f"One or more requested timepoints is too old. Timepoints "
                     f"{timepoints[too_old]} extend beyond end of input climate signal. "
                     f"Returning proxy for valid timepoints only.")
    
    if np.any(too_young):
        warnings.warn(f"One or more requested timepoints is too recent. Timepoints "
                     f"{timepoints[too_young]} are more recent than the top of the core.")
    
    if np.any(mixed_layer & valid_inds):
        warnings.warn(f"Timepoints {timepoints[mixed_layer & valid_inds]} are in the mixed layer")
    
    valid_mask = (timepoints >= 0) & (timepoints < clim_signal.shape[0])
    if not np.all(valid_mask):
        n_invalid = np.sum(~valid_mask)
        print(f"Warning: {n_invalid} timepoints outside climate data range (0 to {clim_signal.shape[0]-1})")
        print(f"Invalid timepoints: {timepoints[~valid_mask]}")
        
        # Filter all arrays to keep only valid timepoints
        timepoints = timepoints[valid_mask]
        sed_acc_rate_array = sed_acc_rate_array[valid_mask]
        layer_width_array = layer_width_array[valid_mask]
        n_samples_array = n_samples_array[valid_mask]
        sigma_meas_array = sigma_meas_array[valid_mask]
        sigma_ind_array = sigma_ind_array[valid_mask]
        
        # Update timepoint count
        n_timepoints = len(timepoints)
        
        if n_timepoints == 0:
            raise ValueError("No valid timepoints remain after bounds checking")

    # Initialize mixed_layer for the current timepoints
    mixed_layer = np.zeros(n_timepoints, dtype=bool)
    
    # ========================================
    # SECTION 4: Setup remaining parameters
    # ========================================
    
    # Optional smoothing output
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

    # Proxy conversion (assume identity for now)
    proxy_clim_signal = clim_signal

    # ========================================
    # SECTION 5: Determine computation path
    # ========================================
    
    # Now determine if we can use rapid case with preprocessed arrays
    use_rapid = is_rapid_case(bio_depth, sed_acc_rate, layer_width, n_samples, habitat_weights)
    
    # ========================================
    # SECTION 6: Main computation loop
    # ========================================
    
    proxy_bt, proxy_bt_sb, proxy_bt_sb_sampY, proxy_bt_sb_sampYM = [], [], [], []
    
    for i, tp in enumerate(timepoints):
        # Get current parameters for this timepoint
        current_sed_acc = sed_acc_rate if use_rapid else sed_acc_rate_array[i]
        current_layer_width = layer_width if use_rapid else layer_width_array[i]
        current_n_samples = n_samples if use_rapid else n_samples_array[i]
        
        # Handle mixed layer adjustment (like R version)
        tp_adjusted = tp
        if mixed_layer[i]:
            # In mixed layer, center bioturbation window around bottom of mixed layer
            tp_adjusted = min_clim_time + int(1000 * bio_depth / current_sed_acc) + int(current_layer_width / 2)
        
        # Calculate bioturbation window
        bio_ts = int(1000 * bio_depth / current_sed_acc)
        layer_yrs = int(np.ceil(1000 * current_layer_width / current_sed_acc))
        first_tp = -bio_ts - layer_yrs // 2
        last_tp = n_bd * bio_ts
        bioturb_window = np.arange(first_tp, last_tp + 1)

        # Get bioturbation weights
        weights = bioturbation_weights(
            z=bioturb_window,
            focal_z=0, #if use_rapid else tp_adjusted,
            layer_width=current_layer_width,
            sed_acc_rate=current_sed_acc,
            bio_depth=bio_depth,
            scale="time"
        )

        # Habitat weights (seasonal) - same for all timepoints since flattened to 1D
        hab_wts = habitat_weights / habitat_weights.sum()
        w_season = np.outer(weights, hab_wts)
        w_season /= w_season.sum()

        # Get proxy window with bounds checking
        window_indices = (tp_adjusted + bioturb_window).clip(min=0, max=proxy_clim_signal.shape[0]-1)
        proxy_window = proxy_clim_signal[window_indices]
    
        # DEBUG: Move the debug code HERE (after proxy_window is defined)
        print(f"\nTimepoint {tp} (i={i}):")
        print(f"  tp_adjusted: {tp_adjusted}")
        print(f"  bioturb_window range: {bioturb_window.min()} to {bioturb_window.max()}")
        print(f"  weights sum: {weights.sum()}")
        print(f"  weights non-zero: {np.sum(weights > 0)}")
        print(f"  proxy_window shape: {proxy_window.shape}")
        print(f"  proxy_window mean: {proxy_window.mean()}")

        if weights.sum() == 0:
            print("  ERROR: All weights are zero!")
        # Calculate bioturbation-only average

        # Right before the bt_weighted_avg line, add:
        print(f"Timepoint {tp} (i={i}):")
        print(f"  current_sed_acc: {current_sed_acc}")
        print(f"  bio_ts: {bio_ts}")
        print(f"  layer_yrs: {layer_yrs}")
        print(f"  bioturb_window range: {bioturb_window.min()} to {bioturb_window.max()}")
        print(f"  tp_adjusted: {tp_adjusted}")
        print(f"  weights sum: {weights.sum()}")
        print(f"  weights shape: {weights.shape}")
        print(f"  proxy_window shape: {proxy_window.shape}")
        print(f"  window_indices range: {window_indices.min()} to {window_indices.max()}")

        if weights.sum() == 0:
            print("  ERROR: All weights are zero!")
            print(f"  weights: {weights}")

        #proxy_bt.append(((np.outer(weights, np.ones(proxy_clim_signal.shape[1])) * proxy_window).sum()) / proxy_clim_signal.shape[1])
        bt_weighted_avg = np.average(proxy_window.mean(axis=1), weights=weights)
        proxy_bt.append(bt_weighted_avg)
        # Calculate bioturbation + seasonal bias
        proxy_bt_sb.append((w_season * proxy_window).sum())

        # Handle finite sampling
        if np.isinf(current_n_samples):
            proxy_bt_sb_sampYM.append(np.full((n_replicates,), np.nan))
            proxy_bt_sb_sampY.append(np.full((n_replicates,), np.nan))
        else:
            ns = int(current_n_samples)

            # Handle sampling with NaN-safe probability weights
            prob_weights = w_season.flatten()
            
            # Check for NaN values and handle them
            valid_mask = ~np.isnan(prob_weights)
            if not np.any(valid_mask):
                # If all weights are NaN, use uniform weights
                prob_weights = np.ones_like(prob_weights) / len(prob_weights)
                warnings.warn(f"All probability weights are NaN for timepoint {tp}. Using uniform weights.")
            else:
                # Set NaN weights to 0 and renormalize
                prob_weights[~valid_mask] = 0
                if prob_weights.sum() > 0:
                    prob_weights = prob_weights / prob_weights.sum()
                else:
                    prob_weights = np.ones_like(prob_weights) / len(prob_weights)
                    warnings.warn(f"All valid probability weights sum to 0 for timepoint {tp}. Using uniform weights.")
            
            # Sample indices
            samples = np.random.choice(len(prob_weights), size=ns * n_replicates, 
                                     p=prob_weights, replace=True)
            
            # Get sampled values
            proxy_flat = proxy_window.flatten()
            sampled_vals = proxy_flat[samples]
            
            # Reshape and take means
            vals = sampled_vals.reshape(n_replicates, ns).mean(axis=1)
            proxy_bt_sb_sampYM.append(vals)
            
            # For sampY (no seasonal aliasing)
            annual_means = (proxy_window * hab_wts[np.newaxis, :]).sum(axis=1)
            row_indices = samples % proxy_window.shape[0]
            annual_sampled = annual_means[row_indices]
            annual_vals = annual_sampled.reshape(n_replicates, ns).mean(axis=1)
            proxy_bt_sb_sampY.append(annual_vals)

    # Convert lists to arrays
    proxy_bt = np.array(proxy_bt)
    proxy_bt_sb = np.array(proxy_bt_sb)
    proxy_bt_sb_sampYM = np.array(proxy_bt_sb_sampYM)
    proxy_bt_sb_sampY = np.array(proxy_bt_sb_sampY)

    # ========================================
    # SECTION 7: Noise and bias (unchanged from your version)
    # ========================================
    
    # Use the preprocessed arrays for noise calculation
    sigma_ind_scaled = sigma_ind_array / np.sqrt(n_samples_array)
    sigma_ind_scaled[~np.isfinite(n_samples_array)] = 0
    sigma_meas_ind = np.sqrt(sigma_meas_array**2 + sigma_ind_scaled**2)

    if scale_noise:
        mean_temperature = proxy_bt
        # For identity case, sigma_meas_ind already in correct units

    if noise_type == "additive":
        noise = np.random.normal(0, sigma_meas_ind[:, np.newaxis], size=proxy_bt_sb_sampYM.shape)
        bias = np.random.normal(0, meas_bias, size=(n_replicates,))
        proxy_bt_sb_sampYM_b = proxy_bt_sb_sampYM + bias
        proxy_bt_sb_sampYM_b_n = proxy_bt_sb_sampYM_b + noise
    else:
        noise = np.exp(np.random.normal(0, sigma_meas_ind[:, np.newaxis], size=proxy_bt_sb_sampYM.shape))
        bias = np.exp(np.random.normal(0, meas_bias, size=(n_replicates,)))
        proxy_bt_sb_sampYM_b = proxy_bt_sb_sampYM * bias
        proxy_bt_sb_sampYM_b_n = proxy_bt_sb_sampYM_b * noise

    # Infinite sample bias+noise
    if noise_type == "additive":
        inf_noise = np.random.normal(0, sigma_meas_ind[:, np.newaxis], size=(n_timepoints, n_replicates))
        inf_bias = np.random.normal(0, meas_bias, size=(n_replicates,))
        proxy_bt_sb_inf_b = proxy_bt_sb[:, np.newaxis] + inf_bias
        proxy_bt_sb_inf_b_n = proxy_bt_sb_inf_b + inf_noise
    else:
        inf_noise = np.exp(np.random.normal(0, sigma_meas_ind[:, np.newaxis], size=(n_timepoints, n_replicates)))
        inf_bias = np.exp(np.random.normal(0, meas_bias, size=(n_replicates,)))
        proxy_bt_sb_inf_b = proxy_bt_sb[:, np.newaxis] * inf_bias
        proxy_bt_sb_inf_b_n = proxy_bt_sb_inf_b * inf_noise

    # Select output path
    simulated_proxy = proxy_bt_sb_sampYM_b_n if np.any(np.isfinite(n_samples_array)) else proxy_bt_sb_inf_b_n
    recon_temp = simulated_proxy
    recon_proxy = simulated_proxy

    # ========================================
    # SECTION 8: Build output (unchanged from your version)
    # ========================================
    
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
        "n_samples": n_samples,  # Keep original for output
        "clim_signal_ann": np.squeeze(clim_signal[timepoints].mean(axis=1)),
        "clim_timepoints_ssr": None,
        "timepoints_smoothed": timepoints_smoothed,
        "clim_signal_smoothed": clim_signal_smoothed
    }

    pfm_df = make_pfm_df(PFM, stages_key)

    sim_df = pd.DataFrame({
        "timepoints": timepoints,
        "simulated_proxy": simulated_proxy[:, 0] if simulated_proxy.ndim == 2 else simulated_proxy,
        "reconstructed_climate": recon_temp[:, 0] if recon_temp.ndim == 2 else recon_temp
    })

    smoothed_signal_df = pd.DataFrame({
        "timepoints": timepoints_smoothed,
        "value": clim_signal_smoothed,
        "stage": "clim_signal_smoothed"
    }) if clim_signal_smoothed is not None else None

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
# 3. Helper functions (unchanged from your version)
# ------------------------------------------------

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
    """
    timepoints = np.asarray(timepoints)
    rel_window = np.arange(width) - int(np.ceil(width / 2))

    n_rows = climate_matrix.shape[0]
    results = []

    for tp in timepoints:
        if start_year is not None:
            center_index = tp - start_year + 1
        else:
            center_index = tp

        inds = rel_window + center_index
        inds = inds[(inds >= 0) & (inds < n_rows)]

        if len(inds) == 0:
            results.append(np.nan)
            continue

        block = climate_matrix[inds, :]
        results.append(np.nanmean(block))

    return np.array(results)