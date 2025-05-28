def foram_growth_ft(temperature_K, foram="ruber", norm=False, min_growth_thresh=0):
    """
    Compute growth rate of planktonic foraminifera as a function of temperature (in Kelvin).

    Parameters:
    - temperature_K: temperature values in Kelvin (scalar, list, or np.ndarray)
    - foram: name of foram species
    - norm: if True, normalize to species' max growth
    - min_growth_thresh: values below this fraction of max are set to zero

    Returns:
    - np.ndarray of daily growth rate
    """
    if foram not in l09_cnsts_dic:
        raise ValueError(f"Foram '{foram}' not found in constants dictionary")

    muT1, TA, TL, TH, TAL, TAH = l09_cnsts_dic[foram]
    max_growth = l09_maxgrowth_dic[foram]

    T = np.asarray(temperature_K)

    # Growth rate function from Lombard et al. (2009)
    growth = (
        muT1 * np.exp(TA / 293 - TA / T) /
        (1 + np.exp(TAL / T - TAL / TL) + np.exp(TAH / TH - TAH / T))
    )

    # Threshold below which growth is set to zero
    growth[growth < min_growth_thresh * max_growth] = 0

    # Zero growth below freezing
    growth[T < 271.15] = 0  # -2°C in Kelvin

    if norm:
        growth = growth / max_growth

    return growth