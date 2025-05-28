import numpy as np

def bioturbation_weights(z, focal_z, layer_width=1, sed_acc_rate=1, bio_depth=10, scale='time'):
    """
    Compute bioturbation weights for a focal depth or time.

    Parameters
    ----------
    z : array-like
        Vector of depths or times to evaluate weights at.
    focal_z : float
        Central depth or time for the sample.
    layer_width : float, optional
        Thickness of the sampled sediment layer.
    sed_acc_rate : float
        Sediment accumulation rate (same units as z/focal_z).
    bio_depth : float
        Depth of the bioturbation layer.
    scale : str
        Either "time" or "depth". If "depth", converts z to time using sed_acc_rate.

    Returns
    -------
    np.ndarray
        Array of bioturbation weights, normalized to sum to 1.
    """
    sed_acc_rate = sed_acc_rate / 1000  # Convert from mm/ka → mm/a (or cm/ka → cm/a)

    if scale not in ["time", "depth"]:
        raise ValueError("scale must be 'time' or 'depth'")

    z = np.array(z, dtype=np.float64)

    if scale == "depth":
        z = z / sed_acc_rate
        focal_z = focal_z / sed_acc_rate

    lwy = int(np.ceil(layer_width / sed_acc_rate))
    mdy = int(np.ceil(bio_depth / sed_acc_rate))

    if lwy == 0 and mdy == 0:
        lwy = 1

    C = lwy / 2
    lam = 1 / mdy if mdy != 0 else np.inf

    z_shifted = z - focal_z + mdy

    # Compute fz based on R logic

    #shallow depth
    if mdy <= 1 and lwy > 0:
        fz = np.where(
            (z_shifted >= -C) & (z_shifted <= C),
            1 / (2 * C),
            0
        )
    #no sample thickness
    elif lwy == 0:
        fz = lam * np.exp(-lam * z_shifted)
        fz[z_shifted < 0] = 0
    # general case
    else:
        fz = np.zeros_like(z_shifted)

        in_center = (z_shifted >= -C) & (z_shifted <= C)
        fz[in_center] = (
            lam * (1 - np.exp(-lam * (C + z_shifted[in_center]))) / (2 * C)
        )

        right_tail = z_shifted > C
        fz[right_tail] = (
            lam * (np.exp(lam * (C - z_shifted[right_tail])) - np.exp(-lam * (C + z_shifted[right_tail]))) / (2 * C)
        )

    # Normalize to sum to 1 if nonzero
    if fz.sum() > 0:
        fz /= np.sum(fz)

    return fz