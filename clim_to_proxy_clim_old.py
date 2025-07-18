
import sys
sys.path.append('/Users/tanaya/USC/sedproxy')

import pandas as pd
import os

# Get the directory where this file (make_pfm_df.py) is located
module_dir = os.path.dirname(__file__)
stages_key_path = os.path.join(module_dir, "data", "stages_key.csv")
stages_key_df = pd.read_csv(stages_key_path)

import warnings
import numpy as np
import pandas as pd
from .bioturbation_weights_old import bioturbation_weights
from .proxy_conversion import proxy_conversion

def is_rapid_case(bio_depth, sed_acc_rate, layer_width, n_samples, habitat_weights):
    return (
        np.ndim(bio_depth) == 0 and
        np.ndim(sed_acc_rate) == 0 and
        np.ndim(layer_width) == 0 and
        np.ndim(n_samples) == 0 and
        isinstance(habitat_weights, np.ndarray) and habitat_weights.ndim == 1
    )


def sample_indices(prob_weights, total_samples):
    """
    Sample indices according to probability weights.
    Equivalent to R's sample(..., prob=..., replace=TRUE).
    """
    prob_weights = np.asarray(prob_weights)
    prob_weights /= prob_weights.sum()
    return np.random.choice(len(prob_weights), size=total_samples, p=prob_weights, replace=True)

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
    stages_key=stages_key_df
):
    # -------------------------------
    # Step 1: Input preparation
    # -------------------------------
    timepoints = np.asarray(timepoints)
    n_timepoints = len(timepoints)
    if isinstance(clim_signal, pd.DataFrame):
        clim_signal = clim_signal.values

    if top_of_core is None:
        top_of_core = 0

    min_clim_time = 0
    max_clim_time = clim_signal.shape[0] - 1

    if noise_type is None:
        noise_type = "multiplicative" if calibration_type == "MgCa" else "additive"
    if scale_noise is None:
        scale_noise = calibration_type != "identity"
    if calibration is None:
        calibration = {
            "MgCa": "Ten planktonic species_350-500",
            "Uk37": "Mueller global",
            "identity": None
        }[calibration_type]

    # -------------------------------
    # Step 2: Calibration conversion
    # -------------------------------
    if calibration_type != "identity":
        proxy_clim_signal = proxy_conversion(
            temperature=clim_signal,
            calibration_type=calibration_type,
            calibration=calibration,
            slp_int_means=slp_int_means,
            slp_int_vcov=slp_int_vcov,
            point_or_sample="point",
            n=1
        )
    else:
        proxy_clim_signal = clim_signal

    # -------------------------------
    # Step 3: Optional smoothed signal
    # -------------------------------
    if plot_sig_res is not None:
        timepoints_smoothed = np.arange(min_clim_time - 1, max_clim_time - 1, plot_sig_res)
        clim_signal_smoothed = chunk_matrix(timepoints_smoothed, plot_sig_res, clim_signal)
    else:
        timepoints_smoothed = None
        clim_signal_smoothed = None

    # -------------------------------
    # Step 4: Simulate proxy signal (Rapid or Slow, inlined)
    # -------------------------------
    use_rapid = is_rapid_case(bio_depth, sed_acc_rate, layer_width, n_samples, habitat_weights)

    proxy_bt = []
    proxy_bt_sb = []
    proxy_bt_sb_sampY = []
    proxy_bt_sb_sampYM = []

    for i, tp in enumerate(timepoints):
        # Set bioturbation window
        if use_rapid:
            bio_ts = int(1000 * bio_depth / sed_acc_rate)
            layer_yrs = int(np.ceil(1000 * layer_width / sed_acc_rate))
        else:
            bio_ts = int(1000 * bio_depth / sed_acc_rate[i])
            layer_yrs = int(np.ceil(1000 * layer_width / sed_acc_rate[i]))

        first_tp = -bio_ts - layer_yrs // 2
        last_tp = n_bd * bio_ts
        bioturb_window = np.arange(first_tp, last_tp + 1)

        # Get bioturbation weights
        weights = bioturbation_weights(
            z=bioturb_window,
            focal_z=0 if use_rapid else tp,
            layer_width=layer_width if use_rapid else layer_width[i],
            sed_acc_rate=sed_acc_rate if use_rapid else sed_acc_rate[i],
            bio_depth=bio_depth,
            scale="time"
        )

        # Define seasonal weights
        if use_rapid:
            hab_wts = habitat_weights / habitat_weights.sum()
            w_season = np.outer(weights, hab_wts)
        else:
            hab_wts = habitat_weights[i] / habitat_weights[i].sum()
            w_season = weights[:, None] * hab_wts

        w_season /= w_season.sum()
        proxy_window = proxy_clim_signal[(tp + bioturb_window).clip(min=0, max=proxy_clim_signal.shape[0] - 1)]

        # Compute proxy signal
        proxy_bt.append((np.outer(weights, np.ones(proxy_clim_signal.shape[1])) * proxy_window).sum())
        proxy_bt_sb.append((w_season * proxy_window).sum())

        # Aliasing
        ns = n_samples if use_rapid else n_samples[i]
        if np.isinf(ns):
            proxy_bt_sb_sampYM.append(np.full(n_replicates, np.nan))
            proxy_bt_sb_sampY.append(np.full(n_replicates, np.nan))

        else:
            ns = int(ns)
            prob_weights = w_season.flatten()
            samples = sample_indices(prob_weights, ns * n_replicates)
            vals = proxy_window.reshape(-1, proxy_window.shape[-1])[samples]
            vals = vals.reshape(n_replicates, ns, -1).mean(axis=1).mean(axis=1)
            proxy_bt_sb_sampYM.append(vals)
            proxy_bt_sb_sampY.append(vals)

    # Convert to arrays
    proxy_bt = np.array(proxy_bt)
    proxy_bt_sb = np.array(proxy_bt_sb)
    proxy_bt_sb_sampYM = np.array(proxy_bt_sb_sampYM)
    proxy_bt_sb_sampY = np.array(proxy_bt_sb_sampY)

    # -------------------------------
    # Step 5: Add bias and noise
    # -------------------------------

    sigma_ind_scaled = sigma_ind / np.sqrt(n_samples) if np.isfinite(n_samples) else 0
    sigma_meas_ind = np.sqrt(sigma_meas**2 + sigma_ind_scaled**2)

    # Rescale measurement error if requested
    if scale_noise:
        # Use the bioturbated signal (proxy_bt) as mean proxy value for rescaling
        if calibration_type == "identity":
            pct = scale_noise  # either True or False, but likely True
        else:
            pct = calibration_type

        # Convert proxy_bt to estimated mean temperature using ProxyConversion
        mean_temperature = proxy_conversion(
            proxy_val=proxy_bt,
            calibration_type=calibration_type,
            calibration=calibration,
            slp_int_means=slp_int_means,
            slp_int_vcov=slp_int_vcov,
            point_or_sample="point",
            n=1
        )

        # Apply conversion for rescaling noise
        sigma_high = proxy_conversion(
            temperature=mean_temperature + sigma_meas_ind,
            calibration_type=calibration_type,
            calibration=calibration,
            slp_int_means=slp_int_means,
            slp_int_vcov=slp_int_vcov,
            point_or_sample="point",
            n=1
        )

        sigma_low = proxy_conversion(
            temperature=mean_temperature,
            calibration_type=calibration_type,
            calibration=calibration,
            slp_int_means=slp_int_means,
            slp_int_vcov=slp_int_vcov,
            point_or_sample="point",
            n=1
        )

        # Rescaled sigma_meas_ind now in proxy-units
        sigma_meas_ind = (sigma_high - sigma_low).squeeze()

        # If multiplicative, convert sigma to relative uncertainty
        if noise_type == "multiplicative":
            sigma_meas_ind = sigma_meas_ind / proxy_bt


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

    # Infinite sample (proxy_bt_sb) bias and noise
    if noise_type == "additive":
        inf_noise = np.random.normal(0, sigma_meas_ind, size=(n_timepoints, n_replicates))
        inf_bias = np.random.normal(0, meas_bias, size=(n_replicates,))
        proxy_bt_sb_inf_b = proxy_bt_sb[:, np.newaxis] + inf_bias  # broadcast across time
        proxy_bt_sb_inf_b_n = proxy_bt_sb_inf_b + inf_noise
        proxy_bt_sb_inf_b = proxy_bt_sb_inf_b.squeeze()
    else:
        inf_noise = np.exp(np.random.normal(0, sigma_meas_ind, size=(n_timepoints, n_replicates)))
        inf_bias = np.exp(np.random.normal(0, meas_bias, size=(n_replicates,)))
        proxy_bt_sb_inf_b = proxy_bt_sb[:, np.newaxis] * inf_bias
        proxy_bt_sb_inf_b_n = proxy_bt_sb_inf_b * inf_noise
        proxy_bt_sb_inf_b_n = proxy_bt_sb_inf_b_n.squeeze()


    # -------------------------------
    # Step 5C: Select simulated_proxy depending on finite or infinite sampling
    # -------------------------------
    if np.isfinite(n_samples):
        simulated_proxy = proxy_bt_sb_sampYM_b_n
    else:
        simulated_proxy = proxy_bt_sb_inf_b_n



    # -------------------------------
    # Step 6: Apply calibration error (fully updated version)
    # -------------------------------
    if calibration_type != "identity":
        if n_replicates > 1:
            # First convert back to temperature units using mean parameters
            recon_temp = proxy_conversion(
                proxy_val=simulated_proxy,
                calibration_type=calibration_type,
                calibration=calibration,
                slp_int_means=slp_int_means,
                slp_int_vcov=slp_int_vcov,
                point_or_sample="point",
                n=1
            )

            # Then resample new calibration parameters for each replicate
            recon_proxy = proxy_conversion(
                temperature=recon_temp,
                calibration_type=calibration_type,
                calibration=calibration,
                slp_int_means=slp_int_means,
                slp_int_vcov=slp_int_vcov,
                point_or_sample="sample",
                n=n_replicates
            )
        else:
            recon_temp = proxy_conversion(
                proxy_val=simulated_proxy,
                calibration_type=calibration_type,
                calibration=calibration,
                slp_int_means=slp_int_means,
                slp_int_vcov=slp_int_vcov,
                point_or_sample="point",
                n=1
            )
            recon_proxy = simulated_proxy  # No replicate error if n_replicates=1
    else:
        recon_temp = simulated_proxy
        recon_proxy = simulated_proxy
    
    print(recon_temp.shape)
    print(timepoints.shape)


    # -------------------------------
    # Step 7: Assemble outputs
    # -------------------------------
    simulated_proxy = proxy_bt_sb_sampYM_b_n if np.isfinite(n_samples) else proxy_bt_sb

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
        "clim_timepoints_ssr": np.squeeze(chunk_matrix(timepoints, plot_sig_res, clim_signal))
        if plot_sig_res is not None else None,
        "timepoints_smoothed": np.squeeze(timepoints_smoothed),
        "clim_signal_smoothed": np.squeeze(clim_signal_smoothed)
    }



    sim_df = pd.DataFrame({
        "timepoints": timepoints,
        "simulated_proxy": simulated_proxy[:, 0] if simulated_proxy.ndim == 2 else simulated_proxy,
        "reconstructed_climate": recon_temp.squeeze()
    })


    everything_df = make_pfm_df(PFM)

    return {
        "simulated_proxy": sim_df,
        "smoothed_signal": pd.DataFrame({
            "timepoints": timepoints_smoothed,
            "value": clim_signal_smoothed,
            "stage": "clim.signal.smoothed"
        }) if clim_signal_smoothed is not None else None,
        "everything": everything_df,
        "calibration_pars": {
            "calibration_type": calibration_type,
            "calibration": calibration,
            "slp_int_means": slp_int_means
        }
    }


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


def make_pfm_df(pfm, stages_key):
    """
    Convert PFM output into a long-form tidy DataFrame with stage metadata.

    Parameters
    ----------
    pfm : dict
        Dictionary containing all proxy stages from ClimToProxyClim.
    stages_key : DataFrame
        Metadata table with columns ['stage', 'scale', 'label']

    Returns
    -------
    DataFrame
        Long-form DataFrame of proxy simulation stages with metadata joined.
    """

    # initial temporary check for dimension size (remove when packaging)
    if pfm['proxy_bt_sb_inf_b'].ndim == 1:
        n_replicates = 1
    else:
        n_replicates = pfm['proxy_bt_sb_inf_b'].shape[1]

    n_timepoints = len(pfm['timepoints'])

    # --- Replicated proxy stages ---
    stage_cols = [
        "proxy_bt_sb_sampY",
        "proxy_bt_sb_sampYM",
        "proxy_bt_sb_inf_b",
        "proxy_bt_sb_sampYM_b",
        "proxy_bt_sb_inf_b_n",
        "proxy_bt_sb_sampYM_b_n",
        "simulated_proxy",
        "simulated_proxy_cal_err",
        "reconstructed_climate"
    ]

    df = pd.DataFrame({
        col: np.full(len(pfm['timepoints']), np.squeeze(pfm[col])) 
            if np.isscalar(np.squeeze(pfm[col])) or np.squeeze(pfm[col]).shape == () 
            else np.squeeze(pfm[col])
        for col in stage_cols
    })


    df['timepoints'] = np.tile(pfm['timepoints'], n_replicates)
    df['n_samples'] = np.tile(pfm['n_samples'], n_replicates)
    df['replicate'] = np.repeat(np.arange(1, n_replicates + 1), n_timepoints)

    df = df.melt(
        id_vars=['timepoints', 'n_samples', 'replicate'],
        value_vars=stage_cols,
        var_name='stage',
        value_name='value'
    ).dropna(subset=['value'])

    # --- Single-replicate stages ---
    df2 = pd.DataFrame({
        'timepoints': pfm['timepoints'],
        'n_samples': np.repeat(pfm['n_samples'], len(pfm['timepoints'])),
        'replicate': 1,
        'proxy_bt': pfm['proxy_bt'],
        'proxy_bt_sb': pfm['proxy_bt_sb'],
        'clim_signal_ann': pfm['clim_signal_ann'],
        'clim_timepoints_ssr': pfm['clim_timepoints_ssr'] if pfm['clim_timepoints_ssr'] is not None else np.full(len(pfm['timepoints']), np.nan)
    })




    df2 = df2.melt(
        id_vars=['timepoints', 'n_samples', 'replicate'],
        var_name='stage',
        value_name='value'
    ).dropna(subset=['value'])

    # --- Smoothed signal ---
    df_smoothed = pd.DataFrame({
        'replicate': 1,
        'timepoints': pfm['timepoints_smoothed'],
        'stage': 'clim_signal_smoothed',
        'value': pfm['clim_signal_smoothed']
    })

    # --- Combine all ---
    full_df = pd.concat([df, df2, df_smoothed], ignore_index=True)

    # --- Join with stages key (adds plotting scale, labels, etc.) ---
    full_df = full_df.merge(stages_key, on='stage', how='left')

    return full_df
