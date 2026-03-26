
import os
import math
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.dates
from matplotlib.backends.backend_pdf import PdfPages
import seaborn as sns

from config import ARTIFACTS_DIR, PLOTS_DIR, EOL_SOH, EKF_CSV

# ── Constants ──────────────────────────────────────────────────────────────────
FLEET_LIFE_YEARS   = 10
LFP_PRIOR_K_DAY    = math.log(100.0 / EOL_SOH) / 3652.0   # ~6.11e-5 /day
LFP_PRIOR_WEIGHT   = 180.0                                  # days of prior weight

STYLE = "seaborn-v0_8-whitegrid"
FIG_SIZE = (14, 8)
DPI = 150

FIT_COLOURS = {
    "reliable"              : "#2ca02c",   # green
    "indicative"            : "#d4ac0d",   # gold
    "unreliable"            : "#ff7f0e",   # orange
    "insufficient_data"     : "#4682b4",   # steelblue
    "no_degradation_signal" : "#aaaaaa",   # grey
}

ANOM_COLS   = ["n_if_anomalies", "n_cusum_soh", "n_cusum_epk",
               "n_cusum_heat", "n_cusum_spread"]
ANOM_LABELS = ["Isolation Forest", "CUSUM SoH", "CUSUM Energy/km",
               "CUSUM Heat", "CUSUM Cell Spread"]
ANOM_COLORS = ["#e41a1c", "#377eb8", "#ff7f00", "#4daf4a", "#984ea3"]

SLOPE_COLS   = ["soh_health_norm", "vsag_slope_norm", "ir_slope_norm",
                "energy_slope_norm", "heat_slope_norm", "spread_slope_norm"]
SLOPE_LABELS = ["SoH Health\n(EKF)", "Voltage Sag", "Internal\nResistance",
                "Energy/km", "Heat Rate", "Cell Spread"]


def short_id(reg):
    """Last 7 characters of registration number."""
    return str(reg)[-7:]


# ── Data loader ────────────────────────────────────────────────────────────────
def load_data():
    rul    = pd.read_csv(os.path.join(ARTIFACTS_DIR, "rul_report.csv"))
    trends = pd.read_csv(os.path.join(ARTIFACTS_DIR, "soh_trends.csv"))
    anom   = pd.read_csv(os.path.join(ARTIFACTS_DIR, "anomaly_scores.csv"))

    # Parse dates in anomaly_scores
    anom["date"] = pd.to_datetime(anom["start_time"], unit="ms", utc=True).dt.tz_localize(None)

    neural_path = os.path.join(ARTIFACTS_DIR, "neural_predictions.csv")
    neural = None
    if os.path.exists(neural_path):
        neural = pd.read_csv(neural_path)
        # Join date from anomaly_scores via (registration_number, cycle_number)
        date_map = anom[["registration_number", "cycle_number", "date"]].drop_duplicates()
        neural = neural.merge(date_map, on=["registration_number", "cycle_number"], how="left")

    ekf = None
    if os.path.exists(EKF_CSV):
        ekf = pd.read_csv(EKF_CSV)
        ekf["date"] = pd.to_datetime(ekf["start_time"], unit="ms", utc=True).dt.tz_localize(None)
        print(f"  ekf_soh           : {ekf.shape[0]} sessions")

    return rul, trends, anom, neural, ekf


# ── Fig 1: Fleet SoH Trajectories ─────────────────────────────────────────────
def fig1_fleet_soh(trends, rul):
    plt.style.use(STYLE)
    fig, ax = plt.subplots(figsize=FIG_SIZE)

    # Map current_soh per vehicle for colour coding
    soh_map = rul.set_index("registration_number")["current_soh"].to_dict()

    discharge = trends[trends["session_type"] == "discharge"].copy()
    vehicles  = discharge["registration_number"].unique()

    # Normalise for viridis colour map
    soh_vals  = np.array([soh_map.get(v, 98.0) for v in vehicles], dtype=float)
    norm      = plt.Normalize(vmin=soh_vals.min(), vmax=soh_vals.max())
    cmap      = plt.cm.viridis

    for veh in vehicles:
        vdf = discharge[discharge["registration_number"] == veh].sort_values("date_days")
        colour = cmap(norm(soh_map.get(veh, 98.0)))
        ax.plot(vdf["date_days"], vdf["soh_smooth"],
                color=colour, linewidth=0.8, alpha=0.7)

    ax.axhline(EOL_SOH, color="red", linestyle="--", linewidth=1.5,
               label=f"End-of-Life ({EOL_SOH:.0f}%)")

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, fraction=0.025, pad=0.01)
    cbar.set_label("Current SoH (%)", fontsize=11)

    # Fleet annotation
    date_min = trends["date"].min()
    date_max = trends["date"].max()
    n_veh    = len(vehicles)
    ax.annotate(
        f"Fleet: {n_veh} vehicles\n{date_min} to {date_max}",
        xy=(0.02, 0.05), xycoords="axes fraction",
        fontsize=9, color="#333333",
        bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.7)
    )

    ax.set_xlabel("Days since first record", fontsize=12)
    ax.set_ylabel("Smoothed SoH (%)", fontsize=12)
    ax.set_title("Fleet SoH Trajectories (All 66 Vehicles)", fontsize=14, fontweight="bold")
    ax.legend(loc="upper right", fontsize=10)
    ax.set_ylim(bottom=75)
    fig.tight_layout()

    path = os.path.join(PLOTS_DIR, "fig1_fleet_soh_trajectories.png")
    fig.savefig(path, dpi=DPI)
    plt.close(fig)
    print(f"  Saved: {path}")
    return fig


# ── Fig 2: RUL Rankings ────────────────────────────────────────────────────────
def fig2_rul_rankings(rul):
    plt.style.use(STYLE)

    df = rul.copy()
    df["short_id"] = df["registration_number"].apply(short_id)
    df["rul_plot"]  = df["rul_years_exp_day"].clip(upper=15.0)
    df = df.sort_values("rul_plot", ascending=True).reset_index(drop=True)

    colours = [FIT_COLOURS.get(q, "#888888") for q in df["fit_quality"]]

    fig, ax = plt.subplots(figsize=(FIG_SIZE[0], max(10, len(df) * 0.22)))

    bars = ax.barh(df["short_id"], df["rul_plot"], color=colours,
                   edgecolor="white", linewidth=0.4, height=0.75)

    ax.axvline(FLEET_LIFE_YEARS, color="#333333", linestyle="--", linewidth=1.5,
               label=f"Expected fleet life ({FLEET_LIFE_YEARS} yr)")

    ax.set_xlabel("Remaining Useful Life (years)", fontsize=12)
    ax.set_ylabel("Vehicle ID", fontsize=12)
    ax.set_title(
        "Remaining Useful Life -- Calendar-Day Model (Prior-Blended)",
        fontsize=14, fontweight="bold"
    )
    ax.set_xlim(0, 15)

    legend_patches = [mpatches.Patch(color=c, label=l)
                      for l, c in FIT_COLOURS.items()]
    ax.legend(handles=legend_patches, loc="lower right", fontsize=9,
              title="Fit Quality", title_fontsize=10)

    fig.tight_layout()
    path = os.path.join(PLOTS_DIR, "fig2_rul_rankings.png")
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")
    return fig


# ── Fig 3: Exponential Decay Fits ─────────────────────────────────────────────
def fig3_exp_fits(trends, rul):
    plt.style.use(STYLE)

    top6 = rul.nlargest(6, "composite_degradation_score").reset_index(drop=True)
    discharge = trends[trends["session_type"] == "discharge"]

    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    axes_flat  = axes.flatten()

    for idx, row in top6.iterrows():
        ax  = axes_flat[idx]
        reg = row["registration_number"]
        A   = row["exp_A_day"]
        k   = row["exp_k_day_blended"]
        rul_yr    = row["rul_years_exp_day"]
        fit_qual  = row["fit_quality"]
        span_days = row["data_span_days"]

        vdf = discharge[discharge["registration_number"] == reg].sort_values("date_days")

        # scatter: actual data
        ax.scatter(vdf["date_days"], vdf["soh_smooth"],
                   s=12, alpha=0.55, color="#4682b4", zorder=3, label="Observed SoH")

        # exponential curve extended to EOL
        if A > 0 and k > 0:
            t_eol = math.log(A / EOL_SOH) / k  # days to EOL
        else:
            t_eol = span_days + 365 * 5

        t_max = max(t_eol * 1.05, span_days + 200)
        t_fit = np.linspace(0, t_max, 300)
        soh_fit = A * np.exp(-k * t_fit)

        ax.plot(t_fit, soh_fit, color="#e74c3c", linewidth=1.8,
                label=f"SoH = {A:.1f}*exp(-{k:.2e}*t)", zorder=4)

        # vertical: data span today
        ax.axvline(span_days, color="#555555", linestyle=":", linewidth=1.2,
                   label="Data span today")

        # horizontal: EOL
        ax.axhline(EOL_SOH, color="red", linestyle="--", linewidth=1.0)
        ax.text(t_max * 0.98, EOL_SOH + 0.3, f"EOL {EOL_SOH:.0f}%",
                ha="right", va="bottom", fontsize=7, color="red")

        # annotation
        ax.annotate(
            f"RUL = {rul_yr:.1f} yr\n{fit_qual}",
            xy=(0.97, 0.97), xycoords="axes fraction",
            ha="right", va="top", fontsize=8,
            bbox=dict(boxstyle="round,pad=0.25", fc="white", alpha=0.8)
        )

        ax.set_title(short_id(reg), fontsize=11, fontweight="bold")
        ax.set_ylim(78, 102)
        ax.set_xlabel("Days", fontsize=8)
        ax.set_ylabel("SoH (%)", fontsize=8)
        ax.tick_params(labelsize=7)

    fig.suptitle(
        "Exponential Decay Fits -- Top-6 Degradation-Signal Vehicles",
        fontsize=14, fontweight="bold", y=1.01
    )
    fig.tight_layout()
    path = os.path.join(PLOTS_DIR, "fig3_exponential_fits.png")
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")
    return fig


# ── Fig 4: Bayesian Blend k evolution ─────────────────────────────────────────
def fig4_bayesian(rul):
    plt.style.use(STYLE)
    fig, ax = plt.subplots(figsize=FIG_SIZE)

    spans = np.linspace(0, 730, 500)

    scenarios = [
        ("Slower degrader (0.5x prior k)", LFP_PRIOR_K_DAY * 0.5,  "#2ecc71"),
        ("Fleet average (1.0x prior k)",   LFP_PRIOR_K_DAY * 1.0,  "#3498db"),
        ("Faster degrader (2.0x prior k)", LFP_PRIOR_K_DAY * 2.0,  "#e74c3c"),
    ]

    for label, k_fit, colour in scenarios:
        k_blend = (spans * k_fit + LFP_PRIOR_WEIGHT * LFP_PRIOR_K_DAY) / \
                  (spans + LFP_PRIOR_WEIGHT)
        ax.plot(spans, k_blend, color=colour, linewidth=2.2, label=label)

    ax.axhline(LFP_PRIOR_K_DAY, color="#888888", linestyle="--", linewidth=1.5,
               label=f"Prior k = {LFP_PRIOR_K_DAY:.2e} /day")

    # shaded band: current fleet position (85-100 days)
    ax.axvspan(85, 100, alpha=0.15, color="#f39c12",
               label="Fleet today (85-100 days)")
    ax.text(92, ax.get_ylim()[0] if ax.get_ylim()[0] > 0 else LFP_PRIOR_K_DAY * 0.62,
            "Fleet\ntoday", ha="center", va="bottom", fontsize=9,
            color="#d35400", fontweight="bold")

    ax.set_xlabel("Data span (days)", fontsize=12)
    ax.set_ylabel("k_day_blend (1/day)", fontsize=12)
    ax.set_title(
        "Bayesian Blend: Prior vs Fitted k (Day Axis)",
        fontsize=14, fontweight="bold"
    )
    ax.legend(fontsize=10)
    ax.yaxis.set_major_formatter(
        matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:.1e}")
    )
    fig.tight_layout()
    path = os.path.join(PLOTS_DIR, "fig4_bayesian_blend.png")
    fig.savefig(path, dpi=DPI)
    plt.close(fig)
    print(f"  Saved: {path}")
    return fig


# ── Fig 5: Degradation Driver Heatmap ─────────────────────────────────────────
def fig5_degradation_heatmap(rul):
    plt.style.use(STYLE)

    df = rul.copy()
    df["short_id"] = df["registration_number"].apply(short_id)
    df = df.sort_values("composite_degradation_score", ascending=False).reset_index(drop=True)

    heat_df = df[SLOPE_COLS].copy()
    heat_df.columns = SLOPE_LABELS
    heat_df.index = df["short_id"]

    # Fill NaN with 0 for heatmap
    heat_df = heat_df.fillna(0.0)

    fig, axes = plt.subplots(1, 2, figsize=(FIG_SIZE[0], max(12, len(df) * 0.28)),
                             gridspec_kw={"width_ratios": [6, 1]})

    sns.heatmap(
        heat_df,
        ax=axes[0],
        cmap="RdYlGn_r",
        vmin=0, vmax=1,
        annot=True, fmt=".2f",
        annot_kws={"size": 6},
        linewidths=0.3,
        cbar=False,
    )
    axes[0].set_title(
        "Degradation Driver Heatmap (Normalised 0-1)",
        fontsize=13, fontweight="bold"
    )
    axes[0].set_xlabel("")
    axes[0].set_ylabel("Vehicle ID", fontsize=10)
    axes[0].tick_params(axis="x", labelsize=9)
    axes[0].tick_params(axis="y", labelsize=7)

    # Composite score side bar
    comp_df = df[["composite_degradation_score"]].copy()
    comp_df.index = df["short_id"]
    sns.heatmap(
        comp_df,
        ax=axes[1],
        cmap="Reds",
        vmin=0, vmax=1,
        annot=True, fmt=".2f",
        annot_kws={"size": 6},
        linewidths=0.3,
        cbar=False,
    )
    axes[1].set_title("Composite", fontsize=9)
    axes[1].set_xlabel("")
    axes[1].set_ylabel("")
    axes[1].tick_params(axis="y", left=False, labelleft=False)
    axes[1].tick_params(axis="x", labelsize=8)

    fig.tight_layout()
    path = os.path.join(PLOTS_DIR, "fig5_degradation_heatmap.png")
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")
    return fig


# ── Fig 6: Anomaly Summary per Vehicle ────────────────────────────────────────
def fig6_anomaly_summary(rul):
    plt.style.use(STYLE)

    df = rul.copy()
    df["short_id"] = df["registration_number"].apply(short_id)
    df = df.sort_values("n_combined_anom", ascending=True).reset_index(drop=True)

    fig, ax = plt.subplots(figsize=(FIG_SIZE[0], max(10, len(df) * 0.22)))

    lefts = np.zeros(len(df))
    for col, label, colour in zip(ANOM_COLS, ANOM_LABELS, ANOM_COLORS):
        vals = df[col].fillna(0).values
        ax.barh(df["short_id"], vals, left=lefts,
                color=colour, label=label, edgecolor="white", linewidth=0.3,
                height=0.75)
        lefts += vals

    # Annotate top 5 by total anomalies
    top5 = df.nlargest(5, "n_combined_anom")
    for _, row in top5.iterrows():
        total = row["n_combined_anom"]
        ax.text(total + 0.3, row["short_id"], f"{int(total)}",
                va="center", fontsize=8, color="#222222")

    ax.set_xlabel("Number of anomaly flags", fontsize=12)
    ax.set_ylabel("Vehicle ID", fontsize=12)
    ax.set_title(
        "Anomaly Flags per Vehicle (Isolation Forest + CUSUM)",
        fontsize=14, fontweight="bold"
    )
    ax.legend(loc="lower right", fontsize=9)
    fig.tight_layout()
    path = os.path.join(PLOTS_DIR, "fig6_anomaly_summary.png")
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")
    return fig


# ── Fig 7: Composite Degradation Score Ranking ────────────────────────────────
def fig7_composite_ranking(rul):
    plt.style.use(STYLE)

    df = rul.copy()
    df["short_id"] = df["registration_number"].apply(short_id)
    df = df.dropna(subset=["composite_degradation_score"])
    df = df.sort_values("composite_degradation_score", ascending=True).reset_index(drop=True)

    colours = [FIT_COLOURS.get(q, "#888888") for q in df["fit_quality"]]

    fig, ax = plt.subplots(figsize=(14, max(10, len(df) * 0.26)))

    # Main composite bars
    ax.barh(df["short_id"], df["composite_degradation_score"],
            color=colours, edgecolor="white", linewidth=0.4,
            height=0.65, alpha=0.85, zorder=2)

    # Overlay individual normalised component scores as scatter dots
    component_styles = [
        ("soh_health_norm",   "SoH Health\n(EKF)", "o", "#1f77b4"),
        ("vsag_slope_norm",   "V-Sag",     "s", "#ff7f0e"),
        ("ir_slope_norm",     "IR",        "^", "#2ca02c"),
        ("energy_slope_norm", "Energy/km", "D", "#9467bd"),
        ("heat_slope_norm",   "Heat",      "P", "#8c564b"),
        ("spread_slope_norm", "Spread",    "X", "#e377c2"),
    ]
    for col, label, marker, mcolor in component_styles:
        if col in df.columns:
            ax.scatter(df[col].fillna(0), df["short_id"],
                       marker=marker, color=mcolor, s=28,
                       zorder=4, label=label, alpha=0.85)

    # Threshold lines
    ax.axvline(0.66, color="#e74c3c", linestyle="--", linewidth=1.2,
               label="High risk (0.66)", zorder=3)
    ax.axvline(0.33, color="#f39c12", linestyle="--", linewidth=1.2,
               label="Moderate risk (0.33)", zorder=3)

    # Annotate each bar: SoH + recommended RUL
    for _, row in df.iterrows():
        soh = row.get("current_soh", None)
        rul_yr = row.get("rul_years_recommended", None)
        score  = row["composite_degradation_score"]
        parts  = []
        if pd.notna(soh):
            parts.append(f"SoH {soh:.0f}%")
        if rul_yr is not None and pd.notna(rul_yr):
            parts.append(f"RUL {rul_yr:.1f}yr")
        if parts:
            ax.text(score + 0.01, row["short_id"], "  " + " | ".join(parts),
                    va="center", fontsize=6.5, color="#222222")

    ax.set_xlim(0, 1.15)
    ax.set_xlabel("Composite Degradation Score (0 = best, 1 = worst)", fontsize=12)
    ax.set_ylabel("Vehicle ID", fontsize=12)
    ax.set_title(
        "Fleet Composite Degradation Ranking\n"
        "Bars = weighted composite score  |  Dots = individual normalised components",
        fontsize=13, fontweight="bold"
    )

    # Legend: fit quality patches + component markers
    fit_patches = [mpatches.Patch(color=c, label=l, alpha=0.85)
                   for l, c in FIT_COLOURS.items()]
    component_handles = [
        plt.scatter([], [], marker=m, color=c, s=40, label=l)
        for _, l, m, c in component_styles
    ]
    threshold_lines = [
        plt.Line2D([0], [0], color="#e74c3c", linestyle="--", linewidth=1.2, label="High risk (0.66)"),
        plt.Line2D([0], [0], color="#f39c12", linestyle="--", linewidth=1.2, label="Moderate risk (0.33)"),
    ]
    ax.legend(
        handles=fit_patches + component_handles + threshold_lines,
        loc="lower right", fontsize=8,
        title="Fit Quality  |  Components  |  Thresholds",
        title_fontsize=8, ncol=2
    )

    fig.tight_layout()
    path = os.path.join(PLOTS_DIR, "fig7_composite_ranking.png")
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")
    return fig


# ── Fig 8: Composite Score Bar Chart (clean, no overlays) ─────────────────────
def fig8_composite_simple(rul):
    plt.style.use(STYLE)

    df = rul.copy()
    df["short_id"] = df["registration_number"].apply(short_id)
    df = df.dropna(subset=["composite_degradation_score"])
    df = df.sort_values("composite_degradation_score", ascending=True).reset_index(drop=True)

    colours = [FIT_COLOURS.get(q, "#888888") for q in df["fit_quality"]]

    fig, ax = plt.subplots(figsize=(12, max(10, len(df) * 0.26)))

    ax.barh(df["short_id"], df["composite_degradation_score"],
            color=colours, edgecolor="white", linewidth=0.4,
            height=0.65, alpha=0.90)

    ax.set_xlim(0, 1.0)
    ax.set_xlabel("Composite Degradation Score (0 = best, 1 = worst)", fontsize=12)
    ax.set_ylabel("Vehicle ID", fontsize=12)
    ax.set_title(
        "Fleet Composite Degradation Ranking",
        fontsize=14, fontweight="bold"
    )

    fit_patches = [mpatches.Patch(color=c, label=l, alpha=0.90)
                   for l, c in FIT_COLOURS.items()]
    ax.legend(handles=fit_patches, loc="lower right", fontsize=9,
              title="Fit Quality", title_fontsize=10)

    fig.tight_layout()
    path = os.path.join(PLOTS_DIR, "fig8_composite_simple.png")
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")
    return fig


# ── Fig 9: Composite Score Vertical Bar Chart ─────────────────────────────────
def fig9_composite_vertical(rul):
    plt.style.use(STYLE)

    df = rul.copy()
    df["short_id"] = df["registration_number"].apply(short_id)
    df = df.dropna(subset=["composite_degradation_score"])
    df = df.sort_values("composite_degradation_score", ascending=False).reset_index(drop=True)

    colours = [FIT_COLOURS.get(q, "#888888") for q in df["fit_quality"]]

    fig, ax = plt.subplots(figsize=(max(14, len(df) * 0.30), 8))

    ax.bar(df["short_id"], df["composite_degradation_score"],
           color=colours, edgecolor="white", linewidth=0.4,
           width=0.7, alpha=0.90)

    ax.set_ylim(0, 1.0)
    ax.set_ylabel("Composite Degradation Score (0 = best, 1 = worst)", fontsize=12)
    ax.set_xlabel("Vehicle ID", fontsize=12)
    ax.set_title(
        "Fleet Composite Degradation Ranking",
        fontsize=14, fontweight="bold"
    )
    ax.tick_params(axis="x", rotation=90, labelsize=7)

    fit_patches = [mpatches.Patch(color=c, label=l, alpha=0.90)
                   for l, c in FIT_COLOURS.items()]
    ax.legend(handles=fit_patches, loc="upper right", fontsize=9,
              title="Fit Quality", title_fontsize=10)

    fig.tight_layout()
    path = os.path.join(PLOTS_DIR, "fig9_composite_vertical.png")
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")
    return fig


# ── Fig 10: RUL Calendar-Day (Prior-Blended) Vertical Bar Chart ───────────────
def fig10_rul_day_simple(rul):
    plt.style.use(STYLE)

    df = rul.copy()
    df["short_id"] = df["registration_number"].apply(short_id)
    df = df.dropna(subset=["rul_years_exp_day"])
    df["rul_plot"] = df["rul_years_exp_day"].clip(upper=10.0)
    df = df.sort_values("rul_plot", ascending=False).reset_index(drop=True)

    colours = [FIT_COLOURS.get(q, "#888888") for q in df["fit_quality"]]

    fig, ax = plt.subplots(figsize=(max(14, len(df) * 0.30), 8))

    ax.bar(df["short_id"], df["rul_plot"],
           color=colours, edgecolor="white", linewidth=0.4,
           width=0.7, alpha=0.90)

    ax.set_ylim(0, 10.0)
    ax.set_ylabel("Remaining Useful Life (years)", fontsize=12)
    ax.set_xlabel("Vehicle ID", fontsize=12)
    ax.set_title(
        "Fleet RUL — Calendar-Day Model (Prior-Blended)",
        fontsize=14, fontweight="bold"
    )
    ax.tick_params(axis="x", rotation=90, labelsize=7)

    fit_patches = [mpatches.Patch(color=c, label=l, alpha=0.90)
                   for l, c in FIT_COLOURS.items()]
    ax.legend(handles=fit_patches, loc="upper right", fontsize=9,
              title="Fit Quality", title_fontsize=10)

    fig.tight_layout()
    path = os.path.join(PLOTS_DIR, "fig10_rul_day_simple.png")
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")
    return fig


# ── Fig 11: Reconstruction Error Distribution (Train vs Test) ─────────────────
def fig11_neural_error_dist(neural):
    plt.style.use(STYLE)
    fig, ax = plt.subplots(figsize=FIG_SIZE)

    train = neural[neural["split"] == "train"]["reconstruction_err"].dropna()
    test  = neural[neural["split"] == "test"]["reconstruction_err"].dropna()

    # Anomaly threshold = 95th percentile of train
    threshold = np.percentile(train, 95)

    sns.kdeplot(train, ax=ax, color="#1f77b4", fill=True, alpha=0.35, label=f"Train  (n={len(train)})")
    sns.kdeplot(test,  ax=ax, color="#ff7f0e", fill=True, alpha=0.35, label=f"Test   (n={len(test)})")

    ax.axvline(threshold, color="#e74c3c", linestyle="--", linewidth=1.8,
               label=f"Anomaly threshold (95th train pct = {threshold:.4f})")

    ax.set_xlabel("Reconstruction Error (MSE)", fontsize=12)
    ax.set_ylabel("Density", fontsize=12)
    ax.set_title(
        "Neural Reconstruction Error Distribution — Train vs Test",
        fontsize=14, fontweight="bold"
    )
    ax.legend(fontsize=10)
    fig.tight_layout()

    path = os.path.join(PLOTS_DIR, "fig11_neural_error_dist.png")
    fig.savefig(path, dpi=DPI)
    plt.close(fig)
    print(f"  Saved: {path}")
    return fig


# ── Fig 12: Per-Vehicle Neural Anomaly Count (Vertical Bar) ───────────────────
def fig12_neural_anomaly_bar(rul):
    plt.style.use(STYLE)

    df = rul.copy()
    df["short_id"] = df["registration_number"].apply(short_id)
    df = df.dropna(subset=["n_neural_anomalies"])
    df = df.sort_values("n_neural_anomalies", ascending=False).reset_index(drop=True)

    # Colour by neural_anomaly_pct (white → red)
    pct_vals = df["neural_anomaly_pct"].fillna(0).values
    norm  = plt.Normalize(vmin=0, vmax=max(pct_vals.max(), 1))
    cmap  = plt.cm.Reds
    colours = [cmap(norm(v)) for v in pct_vals]

    fig, ax = plt.subplots(figsize=(max(14, len(df) * 0.30), 8))

    bars = ax.bar(df["short_id"], df["n_neural_anomalies"],
                  color=colours, edgecolor="white", linewidth=0.4,
                  width=0.7)

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, fraction=0.025, pad=0.01)
    cbar.set_label("Neural Anomaly %", fontsize=11)

    ax.set_ylabel("Number of Neural Anomaly Sessions", fontsize=12)
    ax.set_xlabel("Vehicle ID", fontsize=12)
    ax.set_title(
        "Per-Vehicle Neural Anomaly Count (sorted worst → best)",
        fontsize=14, fontweight="bold"
    )
    ax.tick_params(axis="x", rotation=90, labelsize=7)
    fig.tight_layout()

    path = os.path.join(PLOTS_DIR, "fig12_neural_anomaly_bar.png")
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")
    return fig


# ── Fig 13: Reconstruction Error vs EFC Cumulative ────────────────────────────
def fig13_neural_error_vs_efc(neural):
    plt.style.use(STYLE)
    fig, ax = plt.subplots(figsize=FIG_SIZE)

    normal  = neural[neural["is_anomaly"] == False]
    anomaly = neural[neural["is_anomaly"] == True]

    x_col = "date" if "date" in neural.columns else "cycle_number"
    ax.scatter(normal[x_col],  normal["reconstruction_err"],
               s=12, alpha=0.4, color="#aaaaaa", label="Normal", zorder=2)
    ax.scatter(anomaly[x_col], anomaly["reconstruction_err"],
               s=20, alpha=0.7, color="#e74c3c", label="Anomaly", zorder=3)

    if x_col == "date":
        ax.xaxis.set_major_formatter(matplotlib.dates.DateFormatter("%b %Y"))
        ax.tick_params(axis="x", rotation=30)

    ax.set_xlabel("Date" if x_col == "date" else "Cycle Number", fontsize=12)
    ax.set_ylabel("Reconstruction Error (MSE)", fontsize=12)
    ax.set_title(
        "Reconstruction Error over Time",
        fontsize=14, fontweight="bold"
    )
    ax.legend(fontsize=10)
    fig.tight_layout()

    path = os.path.join(PLOTS_DIR, "fig13_neural_error_vs_efc.png")
    fig.savefig(path, dpi=DPI)
    plt.close(fig)
    print(f"  Saved: {path}")
    return fig


# ── Fig 14: Anomaly Score Over Time (Top 6 Vehicles) ──────────────────────────
def fig14_neural_anomaly_timeline(neural, rul):
    plt.style.use(STYLE)

    # Top 6 vehicles by n_neural_anomalies
    top6_regs = (
        rul.dropna(subset=["n_neural_anomalies"])
           .nlargest(6, "n_neural_anomalies")["registration_number"]
           .tolist()
    )

    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    axes_flat = axes.flatten()

    x_col = "date" if "date" in neural.columns else "seq_index"

    for idx, reg in enumerate(top6_regs):
        ax  = axes_flat[idx]
        vdf = neural[neural["registration_number"] == reg].sort_values(x_col)

        anomaly = vdf[vdf["is_anomaly"] == True]

        # Raw score (light) + rolling median trend (bold) to cut noise
        ax.plot(vdf[x_col], vdf["anomaly_pct"],
                color="#4682b4", linewidth=0.8, alpha=0.4, zorder=2)
        roll = vdf["anomaly_pct"].rolling(5, min_periods=1, center=True).median()
        ax.plot(vdf[x_col], roll,
                color="#1f4e8c", linewidth=1.8, alpha=0.9, zorder=3, label="Trend (5-session median)")
        ax.scatter(anomaly[x_col], anomaly["anomaly_pct"],
                   s=30, color="#e74c3c", zorder=4, label="Anomaly")

        ax.axhline(95, color="#e74c3c", linestyle="--", linewidth=1.2,
                   label="Threshold (95)")

        ax.set_title(short_id(reg), fontsize=11, fontweight="bold")
        ax.set_xlabel("Date" if x_col == "date" else "Sequence Index", fontsize=8)
        ax.set_ylabel("Anomaly %ile", fontsize=8)
        ax.set_ylim(0, 105)
        ax.tick_params(labelsize=7)
        if x_col == "date":
            ax.xaxis.set_major_formatter(matplotlib.dates.DateFormatter("%b %y"))
            ax.tick_params(axis="x", rotation=30)
        ax.legend(fontsize=6, loc="upper left")

    # Hide unused subplots if fewer than 6 vehicles
    for idx in range(len(top6_regs), 6):
        axes_flat[idx].set_visible(False)

    fig.suptitle(
        "Neural Anomaly Score Over Time — Top-6 Vehicles (by anomaly count)",
        fontsize=14, fontweight="bold", y=1.01
    )
    fig.tight_layout()

    path = os.path.join(PLOTS_DIR, "fig14_neural_anomaly_timeline.png")
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")
    return fig


# ── Fig 15: Neural vs Isolation Forest Agreement ──────────────────────────────
def fig15_neural_vs_if(rul):
    plt.style.use(STYLE)

    df = rul.dropna(subset=["n_neural_anomalies", "n_if_anomalies"]).copy()
    df["short_id"] = df["registration_number"].apply(short_id)

    scores = df["composite_degradation_score"].fillna(0)
    norm   = plt.Normalize(vmin=scores.min(), vmax=scores.max())
    cmap   = plt.cm.viridis

    fig, ax = plt.subplots(figsize=FIG_SIZE)

    sc = ax.scatter(df["n_if_anomalies"], df["n_neural_anomalies"],
                    c=scores, cmap=cmap, norm=norm,
                    s=60, alpha=0.80, edgecolors="white", linewidths=0.5, zorder=3)

    # Agreement diagonal
    max_val = max(df["n_if_anomalies"].max(), df["n_neural_anomalies"].max()) * 1.05
    ax.plot([0, max_val], [0, max_val], color="#888888", linestyle="--",
            linewidth=1.2, label="Perfect agreement", zorder=2)

    # Label top-5 disagreement outliers (largest Euclidean distance from diagonal)
    df["_diag_dist"] = abs(df["n_neural_anomalies"] - df["n_if_anomalies"]) / math.sqrt(2)
    top5 = df.nlargest(5, "_diag_dist")
    for _, row in top5.iterrows():
        ax.annotate(
            row["short_id"],
            xy=(row["n_if_anomalies"], row["n_neural_anomalies"]),
            xytext=(6, 3), textcoords="offset points",
            fontsize=7, color="#222222"
        )

    cbar = fig.colorbar(sc, ax=ax, fraction=0.025, pad=0.01)
    cbar.set_label("Composite Degradation Score", fontsize=11)

    ax.set_xlabel("Isolation Forest Anomaly Count", fontsize=12)
    ax.set_ylabel("Neural Model Anomaly Count", fontsize=12)
    ax.set_title(
        "Neural vs Isolation Forest Anomaly Agreement (per vehicle)",
        fontsize=14, fontweight="bold"
    )
    ax.legend(fontsize=10)
    fig.tight_layout()

    path = os.path.join(PLOTS_DIR, "fig15_neural_vs_if.png")
    fig.savefig(path, dpi=DPI)
    plt.close(fig)
    print(f"  Saved: {path}")
    return fig


# ── Fig 16: Neural Anomaly % vs Composite Degradation Score ───────────────────
def fig16_neural_vs_composite(rul):
    plt.style.use(STYLE)

    df = rul.dropna(subset=["neural_anomaly_pct", "composite_degradation_score"]).copy()
    df["short_id"] = df["registration_number"].apply(short_id)

    colours = [FIT_COLOURS.get(q, "#888888") for q in df["fit_quality"]]

    fig, ax = plt.subplots(figsize=FIG_SIZE)

    for fq, colour in FIT_COLOURS.items():
        sub = df[df["fit_quality"] == fq]
        if sub.empty:
            continue
        ax.scatter(sub["composite_degradation_score"], sub["neural_anomaly_pct"],
                   s=55, alpha=0.80, color=colour, label=fq,
                   edgecolors="white", linewidths=0.5, zorder=3)

    ax.set_xlabel("Composite Degradation Score", fontsize=12)
    ax.set_ylabel("Neural Anomaly %", fontsize=12)
    ax.set_title(
        "Neural Anomaly Rate vs Composite Degradation Score",
        fontsize=14, fontweight="bold"
    )
    legend_patches = [mpatches.Patch(color=c, label=l) for l, c in FIT_COLOURS.items()]
    ax.legend(handles=legend_patches, title="Fit Quality", fontsize=9, title_fontsize=10)
    fig.tight_layout()

    path = os.path.join(PLOTS_DIR, "fig16_neural_vs_composite.png")
    fig.savefig(path, dpi=DPI)
    plt.close(fig)
    print(f"  Saved: {path}")
    return fig


# ── Fig 17: Isolation Forest Score Distribution ───────────────────────────────
def fig17_if_score_dist(anom):
    plt.style.use(STYLE)

    df = anom[anom["session_type"] == "discharge"].copy()
    normal  = df[df["if_anomaly"] == False]["if_score"].dropna()
    anomaly = df[df["if_anomaly"] == True]["if_score"].dropna()

    fig, ax = plt.subplots(figsize=FIG_SIZE)

    sns.kdeplot(normal,  ax=ax, color="#1f77b4", fill=True, alpha=0.35,
                label=f"Normal  (n={len(normal):,})")
    sns.kdeplot(anomaly, ax=ax, color="#e74c3c", fill=True, alpha=0.35,
                label=f"Anomaly (n={len(anomaly):,})")

    ax.axvline(0, color="#333333", linestyle="--", linewidth=1.8,
               label="Decision boundary (score = 0)")

    ax.set_xlabel("Isolation Forest Anomaly Score", fontsize=12)
    ax.set_ylabel("Density", fontsize=12)
    ax.set_title(
        "Isolation Forest Score Distribution — Normal vs Anomaly Sessions",
        fontsize=14, fontweight="bold"
    )
    ax.legend(fontsize=10)
    fig.tight_layout()

    path = os.path.join(PLOTS_DIR, "fig17_if_score_dist.png")
    fig.savefig(path, dpi=DPI)
    plt.close(fig)
    print(f"  Saved: {path}")
    return fig


# ── Fig 18: CUSUM Alarm Heatmap (per vehicle × alarm type) ────────────────────
def fig18_cusum_heatmap(rul):
    plt.style.use(STYLE)

    CUSUM_COLS   = ["n_cusum_soh", "n_cusum_epk", "n_cusum_heat", "n_cusum_spread"]
    CUSUM_LABELS = ["SoH", "Energy/km", "Heat Rate", "Cell Spread"]

    df = rul.copy()
    df["short_id"]    = df["registration_number"].apply(short_id)
    df["total_cusum"] = df[CUSUM_COLS].fillna(0).sum(axis=1)
    df = df.sort_values("total_cusum", ascending=False).reset_index(drop=True)

    heat_df = df[CUSUM_COLS].fillna(0).copy()
    heat_df.columns = CUSUM_LABELS
    heat_df.index   = df["short_id"]

    # Only keep vehicles with at least one CUSUM alarm for readability
    heat_df = heat_df[heat_df.sum(axis=1) > 0]

    fig, ax = plt.subplots(figsize=(10, max(8, len(heat_df) * 0.30)))

    sns.heatmap(
        heat_df,
        ax=ax,
        cmap="YlOrRd",
        linewidths=0.4,
        annot=True, fmt=".0f",
        annot_kws={"size": 8},
        cbar_kws={"label": "Alarm count"},
    )
    ax.set_title(
        "CUSUM Alarm Counts per Vehicle × Alarm Type\n(vehicles with ≥1 alarm, sorted by total)",
        fontsize=13, fontweight="bold"
    )
    ax.set_xlabel("CUSUM Channel", fontsize=11)
    ax.set_ylabel("Vehicle ID", fontsize=11)
    ax.tick_params(axis="y", labelsize=8)
    fig.tight_layout()

    path = os.path.join(PLOTS_DIR, "fig18_cusum_heatmap.png")
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")
    return fig


# ── Fig 19: Anomaly Timeline — IF + CUSUM (Top 6 vehicles) ────────────────────
def fig19_anomaly_timeline(anom, rul):
    plt.style.use(STYLE)

    top6_regs = (
        rul.dropna(subset=["n_combined_anom"])
           .nlargest(6, "n_combined_anom")["registration_number"]
           .tolist()
    )

    disc = anom[anom["session_type"] == "discharge"].copy()

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes_flat = axes.flatten()

    for idx, reg in enumerate(top6_regs):
        ax  = axes_flat[idx]
        vdf = disc[disc["registration_number"] == reg].sort_values("date")

        if vdf.empty:
            ax.set_visible(False)
            continue

        # SoH trajectory
        ax.plot(vdf["date"], vdf["soh_smooth"],
                color="#4682b4", linewidth=1.5, alpha=0.85, zorder=2, label="SoH (smoothed)")

        # IF anomaly points
        if_anom = vdf[vdf["if_anomaly"] == True]
        ax.scatter(if_anom["date"], if_anom["soh_smooth"],
                   s=45, color="#e74c3c", zorder=4, label="IF anomaly", marker="o")

        # CUSUM alarm points (any channel)
        cusum_anom = vdf[vdf["cusum_alarm"] == True]
        ax.scatter(cusum_anom["date"], cusum_anom["soh_smooth"],
                   s=55, color="#ff7f0e", zorder=5, label="CUSUM alarm",
                   marker="^", alpha=0.85)

        ax.axhline(EOL_SOH, color="red", linestyle="--", linewidth=0.9, alpha=0.6)

        ax.set_title(short_id(reg), fontsize=11, fontweight="bold")
        ax.set_xlabel("Date", fontsize=8)
        ax.set_ylabel("SoH (%)", fontsize=8)
        ax.xaxis.set_major_formatter(matplotlib.dates.DateFormatter("%b %y"))
        ax.tick_params(labelsize=7)
        ax.tick_params(axis="x", rotation=30)
        ax.legend(fontsize=7, loc="lower left")

    for idx in range(len(top6_regs), 6):
        axes_flat[idx].set_visible(False)

    fig.suptitle(
        "Anomaly Detection Timeline — Top-6 Vehicles (IF + CUSUM overlaid on SoH)",
        fontsize=14, fontweight="bold", y=1.01
    )
    fig.tight_layout()

    path = os.path.join(PLOTS_DIR, "fig19_anomaly_timeline.png")
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")
    return fig


# ── Fig 20: EKF SoH Trace with Uncertainty Band (top 9 vehicles) ───────────────
def fig20_ekf_soh_trace(ekf: pd.DataFrame, rul: pd.DataFrame, n_vehicles: int = 9):
    """
    EKF-estimated SoH over time with 1-sigma uncertainty band.
    Overlays raw capacity_soh observations as scatter points.
    Shows top vehicles by composite degradation score (or all if < n_vehicles).
    """
    plt.style.use(STYLE)

    # Pick vehicles: prefer worst by composite score, else all available
    if "composite_degradation_score" in rul.columns:
        plot_regs = (
            rul.dropna(subset=["composite_degradation_score"])
               .nlargest(n_vehicles, "composite_degradation_score")["registration_number"]
               .tolist()
        )
    else:
        plot_regs = ekf["registration_number"].unique()[:n_vehicles].tolist()

    plot_regs = [r for r in plot_regs if r in ekf["registration_number"].values]
    if not plot_regs:
        print("  [fig20] No EKF data for selected vehicles — skipping.")
        return None

    ncols = 3
    nrows = math.ceil(len(plot_regs) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(18, 5 * nrows))
    axes_flat = axes.flatten() if hasattr(axes, "flatten") else [axes]

    for idx, reg in enumerate(plot_regs):
        ax  = axes_flat[idx]
        vdf = ekf[ekf["registration_number"] == reg].sort_values("date")

        if vdf.empty:
            ax.set_visible(False)
            continue

        # EKF SoH trace
        ax.plot(vdf["date"], vdf["ekf_soh"],
                color="#2ca02c", linewidth=1.8, zorder=3, label="EKF SoH")

        # ±1σ uncertainty band
        if "ekf_soh_std" in vdf.columns:
            ax.fill_between(
                vdf["date"],
                vdf["ekf_soh"] - vdf["ekf_soh_std"],
                vdf["ekf_soh"] + vdf["ekf_soh_std"],
                alpha=0.25, color="#2ca02c", zorder=2, label="±1σ"
            )

        # Raw capacity_soh observations
        obs_mask = vdf["capacity_soh_obs"].notna()
        if obs_mask.any():
            ax.scatter(vdf.loc[obs_mask, "date"], vdf.loc[obs_mask, "capacity_soh_obs"],
                       s=20, color="#1f77b4", alpha=0.7, zorder=4,
                       label="capacity_soh (obs)")

        # EOL line
        ax.axhline(EOL_SOH, color="red", linestyle="--", linewidth=0.9, alpha=0.6)

        ax.set_title(short_id(reg), fontsize=11, fontweight="bold")
        ax.set_xlabel("Date", fontsize=8)
        ax.set_ylabel("SoH (%)", fontsize=8)
        ax.xaxis.set_major_formatter(matplotlib.dates.DateFormatter("%b %y"))
        ax.tick_params(labelsize=7)
        ax.tick_params(axis="x", rotation=30)
        ax.legend(fontsize=7, loc="lower left")

        # Annotate current EKF SoH + RUL
        last = vdf.iloc[-1]
        ann_text = f"EKF: {last['ekf_soh']:.1f}%"
        if pd.notna(last.get("ekf_rul_days")):
            ann_text += f"\nRUL: {int(last['ekf_rul_days'])} days"
        ax.annotate(ann_text, xy=(0.02, 0.06), xycoords="axes fraction",
                    fontsize=7, color="#2ca02c",
                    bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.7))

    for idx in range(len(plot_regs), len(axes_flat)):
        axes_flat[idx].set_visible(False)

    fig.suptitle(
        "EKF SoH Tracking — Top Vehicles by Degradation Score\n"
        "(green trace = filtered estimate, band = ±1σ, dots = observed capacity_soh)",
        fontsize=13, fontweight="bold", y=1.01
    )
    fig.tight_layout()

    path = os.path.join(PLOTS_DIR, "fig20_ekf_soh_trace.png")
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")
    return fig


# ── PDF export ─────────────────────────────────────────────────────────────────
def save_pdf(fig_paths):
    pdf_path = os.path.join(PLOTS_DIR, "rul_presentation.pdf")
    with PdfPages(pdf_path) as pdf:
        for path in fig_paths:
            img = plt.imread(path)
            fig, ax = plt.subplots(figsize=(16.54, 11.69))  # A4 landscape inches
            ax.imshow(img)
            ax.axis("off")
            fig.tight_layout(pad=0)
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)
    print(f"  Saved PDF: {pdf_path}")


# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    print("Loading data...")
    rul, trends, anom, neural, ekf = load_data()
    print(f"  rul_report        : {rul.shape[0]} vehicles, {rul.shape[1]} columns")
    print(f"  soh_trends        : {trends.shape[0]} sessions")
    print(f"  anomaly           : {anom.shape[0]} sessions")
    if neural is not None:
        print(f"  neural_predictions: {neural.shape[0]} sequences")
    else:
        print("  neural_predictions: not found — figs 11/13/14 will be skipped")
    if ekf is None:
        print("  ekf_soh           : not found — fig20 will be skipped (run ekf_soh.py)")

    print("\nGenerating figures...")
    fig1_fleet_soh(trends, rul)
    fig2_rul_rankings(rul)
    fig3_exp_fits(trends, rul)
    fig4_bayesian(rul)
    fig5_degradation_heatmap(rul)
    fig6_anomaly_summary(rul)
    fig7_composite_ranking(rul)
    fig8_composite_simple(rul)
    fig9_composite_vertical(rul)
    fig10_rul_day_simple(rul)

    fig_paths = [
        os.path.join(PLOTS_DIR, "fig1_fleet_soh_trajectories.png"),
        os.path.join(PLOTS_DIR, "fig2_rul_rankings.png"),
        os.path.join(PLOTS_DIR, "fig3_exponential_fits.png"),
        os.path.join(PLOTS_DIR, "fig4_bayesian_blend.png"),
        os.path.join(PLOTS_DIR, "fig5_degradation_heatmap.png"),
        os.path.join(PLOTS_DIR, "fig6_anomaly_summary.png"),
        os.path.join(PLOTS_DIR, "fig7_composite_ranking.png"),
        os.path.join(PLOTS_DIR, "fig8_composite_simple.png"),
        os.path.join(PLOTS_DIR, "fig9_composite_vertical.png"),
        os.path.join(PLOTS_DIR, "fig10_rul_day_simple.png"),
    ]

    # Neural model figures (require neural_predictions.csv)
    if neural is not None:
        fig11_neural_error_dist(neural)
        fig_paths.append(os.path.join(PLOTS_DIR, "fig11_neural_error_dist.png"))

    if "n_neural_anomalies" in rul.columns:
        fig12_neural_anomaly_bar(rul)
        fig_paths.append(os.path.join(PLOTS_DIR, "fig12_neural_anomaly_bar.png"))

    if neural is not None:
        fig13_neural_error_vs_efc(neural)
        fig_paths.append(os.path.join(PLOTS_DIR, "fig13_neural_error_vs_efc.png"))

    if neural is not None and "n_neural_anomalies" in rul.columns:
        fig14_neural_anomaly_timeline(neural, rul)
        fig_paths.append(os.path.join(PLOTS_DIR, "fig14_neural_anomaly_timeline.png"))

    if "n_neural_anomalies" in rul.columns and "n_if_anomalies" in rul.columns:
        fig15_neural_vs_if(rul)
        fig_paths.append(os.path.join(PLOTS_DIR, "fig15_neural_vs_if.png"))

    if "neural_anomaly_pct" in rul.columns and "composite_degradation_score" in rul.columns:
        fig16_neural_vs_composite(rul)
        fig_paths.append(os.path.join(PLOTS_DIR, "fig16_neural_vs_composite.png"))

    # Isolation Forest + CUSUM figures
    if "if_score" in anom.columns:
        fig17_if_score_dist(anom)
        fig_paths.append(os.path.join(PLOTS_DIR, "fig17_if_score_dist.png"))

    if "n_cusum_soh" in rul.columns:
        fig18_cusum_heatmap(rul)
        fig_paths.append(os.path.join(PLOTS_DIR, "fig18_cusum_heatmap.png"))

    if "cusum_alarm" in anom.columns and "n_combined_anom" in rul.columns:
        fig19_anomaly_timeline(anom, rul)
        fig_paths.append(os.path.join(PLOTS_DIR, "fig19_anomaly_timeline.png"))

    # EKF SoH trace (requires ekf_soh.py to have been run first)
    if ekf is not None:
        fig20_ekf_soh_trace(ekf, rul)
        fig_paths.append(os.path.join(PLOTS_DIR, "fig20_ekf_soh_trace.png"))

    print("\nBuilding PDF...")
    save_pdf(fig_paths)

    print(f"\nSaved {len(fig_paths)} figures to {PLOTS_DIR}/")


if __name__ == "__main__":
    main()
