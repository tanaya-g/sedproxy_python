import numpy as np
from scipy import stats

def bioturbation_weights(z, focal_z, layer_width=1, sed_acc_rate=1, bio_depth=10, scale='time'):
    """
    Compute bioturbation weights for a focal depth or time.
    
    This is a direct Python translation of the R BioturbationWeights function.
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

    # Shift z coordinates (this matches R exactly)
    z_shifted = z - focal_z + mdy

    # Compute fz using R logic exactly
    if mdy <= 1 and lwy > 0:
        # Use uniform distribution like R's dunif
        fz = np.where(
            (z_shifted >= -C) & (z_shifted <= C),
            1 / (2 * C),  # This is equivalent to stats.uniform.pdf(z_shifted, -C, 2*C)
            0
        )
    elif lwy == 0:
        # Use exponential distribution like R's dexp
        # Note: R's dexp(x, rate) = rate * exp(-rate * x) for x >= 0
        fz = np.where(z_shifted >= 0, lam * np.exp(-lam * z_shifted), 0)
    else:
        # General case - match R formula exactly
        fz = np.zeros_like(z_shifted)
        
        # Case 1: z < -C
        mask1 = z_shifted < -C
        fz[mask1] = 0
        
        # Case 2: -C <= z <= C  
        mask2 = (z_shifted >= -C) & (z_shifted <= C)
        if np.any(mask2):
            fz[mask2] = (lam * (1/lam - np.exp(-lam*C - lam*z_shifted[mask2])/lam)) / (2*C)
        
        # Case 3: z > C
        mask3 = z_shifted > C
        if np.any(mask3):
            fz[mask3] = (lam * (np.exp(lam*C - lam*z_shifted[mask3])/lam - np.exp(-lam*C - lam*z_shifted[mask3])/lam)) / (2*C)

    # Normalize exactly like R
    if np.sum(fz) == 0:
        return fz
    else:
        fz = fz / np.sum(fz)
        return fz