import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import os

# Get the directory where this file (make_pfm_df.py) is located
module_dir = os.path.dirname(__file__)
stages_key_path = os.path.join(module_dir, "data", "stages_key.csv")
stages_key_df = pd.read_csv(stages_key_path)


def plot_pfms(
    pfm_df,
    stage_order="var",
    plot_stages="default",
    max_replicates=5,
    color_palette="default",
    alpha_palette="default",
    level_labels="default",
    stages_key=None
):
    """
    Plot forward-modeled sedimentary proxy stages from sedproxy output.

    Parameters:
    - pfm_df: DataFrame from make_pfm_df()
    - stage_order: "var" or "seq" (variance or sequence-based stage ordering)
    - plot_stages: "default", "all", or a list of stages
    - max_replicates: number of replicates to display
    - color_palette: dict or "default" (uses stages_key)
    - alpha_palette: dict or "default" (uses stages_key)
    - level_labels: dict or "default" (uses stages_key)
    - stages_key: required if using defaults (dict with keys: stage, scale, color, alpha, label)
    """

    # Handle full dictionary input vs raw dataframe input
    if isinstance(pfm_df, dict):
        cal_type = pfm_df.get("calibration_pars", {}).get("calibration_type", "identity")
        df = pfm_df["everything"]
    else:
        cal_type = "identity"  # fallback if just raw dataframe passed
        df = pfm_df

    # Make sure 'scale' column exists safely before filling
    if "scale" not in df.columns:
        df["scale"] = "unspecified"
    else:
        df["scale"] = df["scale"].fillna("unspecified")


    if "replicate" not in df.columns:
        df["replicate"] = 1

    # Add missing metadata columns
    for col in ["Location", "ID.no", "Proxy"]:
        if col not in df.columns:
            df[col] = ""

    # Stage filtering
    if stages_key is None and plot_stages != "all":
        raise ValueError("stages_key is required if using default plotting palettes")

    if isinstance(stages_key, pd.DataFrame):
        skey = stages_key.set_index("stage")
    else:
        skey = pd.DataFrame(stages_key).set_index("stage")

    if plot_stages == "default":
        cal_type = df.attrs.get("calibration_pars", {}).get("calibration_type", "identity")
        if cal_type == "identity":
            plotting_levels = [
                "clim.signal.monthly", "clim.signal.smoothed", "proxy.bt", "proxy.bt.sb",
                "proxy.bt.sb.sampYM", "simulated.proxy", "observed.proxy"
            ]
        else:
            plotting_levels = [
                "clim.signal.monthly", "clim.signal.smoothed", "proxy.bt", "proxy.bt.sb",
                "proxy.bt.sb.sampYM", "simulated.proxy", "simulated.proxy.cal.err",
                "reconstructed.climate", "observed.proxy"
            ]
    elif plot_stages == "all":
        plotting_levels = skey.index.tolist()
    else:
        plotting_levels = plot_stages

    # Filter for desired stages and replicates
    df = df[df["stage"].isin(plotting_levels) & (df["replicate"] <= max_replicates)]

    # Merge scale if available
    if "scale" not in df.columns and "scale" in skey.columns:
        df = df.merge(skey[["scale"]], left_on="stage", right_index=True, how="left")

    # Determine stage order
    if stage_order == "seq":
        df["stage"] = pd.Categorical(df["stage"], categories=plotting_levels, ordered=True)
    elif stage_order == "var":
        stage_vars = df.groupby("stage")["value"].var().sort_values(ascending=False)
        df["stage"] = pd.Categorical(df["stage"], categories=stage_vars.index.tolist(), ordered=True)

    # Set color and alpha palettes
    if color_palette == "default":
        color_palette = skey["plotting_colour"].to_dict()
    if alpha_palette == "default":
        alpha_palette = skey["plotting_alpha"].to_dict()
    if level_labels == "default":
        level_labels = skey["label"].to_dict()

    # Plot
    plt.figure(figsize=(12, 6))
    sns.set(style="whitegrid")

    g = sns.relplot(
        data=df,
        x="timepoints", y="value",
        kind="line",
        hue="stage", col="scale" if "scale" in df.columns else None,
        palette=color_palette,
        alpha=0.8,
        facet_kws={"sharey": False} if "scale" in df.columns else {}
    )

    g.set_axis_labels("Timepoints", "Proxy Value")
    g.add_legend()

    plt.tight_layout()
    return g