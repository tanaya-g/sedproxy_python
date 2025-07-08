import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def plot_pfms(
    pfm_obj,  # full output dictionary from clim_to_proxy_clim()
    stages_key,
    stage_order="var",
    plot_stages="default",
    max_replicates=5,
    color_palette="default",
    alpha_palette="default",
    level_labels="default"
):

    # Handle full object or direct dataframe input
    if isinstance(pfm_obj, dict):
        cal_type = pfm_obj.get("calibration_pars", {}).get("calibration_type", "identity")
        df = pfm_obj["everything"].copy()
    else:
        cal_type = "identity"
        df = pfm_obj.copy()

    # Convert ALL categorical columns to strings to avoid issues
    for col in df.columns:
        if df[col].dtype.name == 'category':
            df[col] = df[col].astype(str)

    # Make sure replicate column exists
    if "replicate" not in df.columns:
        df["replicate"] = 1

    # Handle any missing metadata columns
    for meta_col in ["Location", "ID.no", "Proxy"]:
        if meta_col not in df.columns:
            df[meta_col] = ""

    # Normalize column names for stages_key
    stages_key = stages_key.copy()  # Don't modify original
    stages_key.columns = [c.replace(".", "_") for c in stages_key.columns]
    skey = stages_key.set_index("stage")

    # Handle plotting levels based on calibration type
    if plot_stages == "default":
        # Use the main stages that show the proxy forward modeling process
        plotting_levels = [
            "clim_signal_ann",           # (1) Input climate
            "proxy_bt",                  # (2) +Bioturbation  
            "proxy_bt_sb",               # (3) +Habitat bias
            "proxy_bt_sb_sampYM",        # (4) +Aliasing YM
            "simulated_proxy"            # (5) +Independent error (final result)
        ]
    elif plot_stages == "all":
        plotting_levels = [
            "clim_signal_ann", "proxy_bt", "proxy_bt_sb", 
            "proxy_bt_sb_sampY", "proxy_bt_sb_sampYM",
            "proxy_bt_sb_inf_b", "proxy_bt_sb_sampYM_b",
            "proxy_bt_sb_inf_b_n", "proxy_bt_sb_sampYM_b_n",
            "simulated_proxy", "simulated_proxy_cal_err", "reconstructed_climate"
        ]
    else:
        plotting_levels = plot_stages

    # Filter stages
    df = df[df["stage"].isin(plotting_levels) & (df["replicate"] <= max_replicates)]

    # Merge scale info if needed
    if "scale" not in df.columns and "scale" in skey.columns:
        df = df.merge(skey[["scale"]], left_on="stage", right_index=True, how="left")

    # Always make sure scale column exists
    if "scale" not in df.columns:
        df["scale"] = "unspecified"
    else:
        df["scale"] = df["scale"].fillna("unspecified")

    # *** FIX: Put climate and proxy data on same scale for unified plotting ***
    # Set all relevant stages to "Proxy units" so they plot together
    df.loc[df["stage"].isin(["clim_signal_ann"]), "scale"] = "Proxy units"

    # Build palettes
    if color_palette == "default":
        color_palette = skey["plotting_colour"].to_dict()
    if alpha_palette == "default":
        alpha_palette = skey["plotting_alpha"].to_dict()
    if level_labels == "default":
        level_labels = skey["label"].to_dict()

    # *** CLEAN FIX: Apply the level labels ***
    #print("Available stages in data:", df["stage"].unique())
    #print("Available labels in stages_key:", level_labels)
    
    # Create stage_label column using simple mapping
    df["stage_label"] = [level_labels.get(stage, stage) for stage in df["stage"]]
    
    # Debug: Check if mapping worked
    print("Mapping result:")
    for stage in df["stage"].unique():
        label = level_labels.get(stage, "NOT FOUND")
        #print(f"  {stage} -> {label}")
    
    #print("Unique stage_labels:", df["stage_label"].unique())

    # Apply stage ordering to the labels
    if stage_order == "seq":
        # Keep original order
        label_order = [level_labels.get(stage, stage) for stage in plotting_levels if stage in df["stage"].values]
        df["stage_label"] = pd.Categorical(df["stage_label"], categories=label_order, ordered=True)
    elif stage_order == "var":
        # Order by variance
        stage_vars = df.groupby("stage_label")["value"].var().sort_values(ascending=False)
        df["stage_label"] = pd.Categorical(df["stage_label"], categories=stage_vars.index.tolist(), ordered=True)

    # Subset palettes to avoid seaborn errors if some stages are missing
    plot_stages_present = df["stage"].unique().tolist()
    color_palette = {k: v for k, v in color_palette.items() if k in plot_stages_present}
    alpha_palette = {k: v for k, v in alpha_palette.items() if k in plot_stages_present}

    # *** FIX: Create label-to-color mapping ***
    label_color_palette = {}
    for stage, label in level_labels.items():
        if stage in color_palette:
            label_color_palette[label] = color_palette[stage]

    # Safely suppress style if only one replicate exists
    style = "replicate" if df["replicate"].nunique() > 1 else None

    # *** FIX: Use stage_label for hue instead of stage ***
    g = sns.relplot(
        data=df,
        x="timepoints", y="value",
        kind="line",
        hue="stage_label",  # Changed from "stage" to "stage_label"
        style=style,
        col="scale",
        palette=label_color_palette,  # Use the label-based palette
        alpha=0.8,
        facet_kws={"sharey": False}
    )

    # *** FIX: Set proper axis labels and title ***
    g.set_axis_labels("Timepoints", "Proxy value")
    
    # Set the scale titles properly
    for ax, scale in zip(g.axes.flat, df["scale"].unique()):
        ax.set_title(f"scale = {scale}")

    g._legend.set_bbox_to_anchor((1.02, 0.5))
    g._legend.set_loc('center left')

    plt.tight_layout()
    return g