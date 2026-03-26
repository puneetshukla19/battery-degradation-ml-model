"""
verify_exp.py
=============

Two jobs in one script:

1. EFC vs Day RUL comparison
   ---------------------------
   Both models already exist in rul_report.csv:
     rul_years_exp      <- EFC-axis blended model  (k blend uses efc_total + 200 EFC prior)
     rul_years_exp_day  <- Day-axis blended model  (k blend uses data_span_days + 180 day prior)
   This section prints a side-by-side table and saves it as efc_vs_day_rul.csv.
   NOTE: EFC-axis RUL appears wildly inflated (24-82 yr) vs day-axis (4-11 yr) because
   avg_efc_per_day is tiny (~0.03-0.08 EFC/day) at this fleet age, so the EFC->days
   conversion magnifies the uncertainty. Day-axis is the headline number.

2. Exponential decay verification plots
   ---------------------------------------
   For the 3 "indicative" vehicles + 3 "insufficient_data" vehicles with the most data:
   Each vehicle gets a 2-panel figure:
     Left  - Linear y-scale: scatter of actual SoH + fitted exp curve (full range to EOL)
     Right - Semi-log y-scale: same data + curve.
             KEY: true exponential decay is a STRAIGHT LINE on a log-y axis.
             If the fitted curve is straight on the right panel = confirmed exponential.

Output:
  BASE/plots/efc_vs_day_rul.csv
  BASE/plots/verify_<reg>_linear.png   (one per vehicle)
  BASE/plots/verify_<reg>_semilog.png  (one per vehicle -- combined 2-panel)
  BASE/plots/verify_all_6.png          (all 6 vehicles in one 3x2 grid, 2 panels each)
"""

import os
import math
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

from config import ARTIFACTS_DIR, PLOTS_DIR, EOL_SOH

LFP_PRIOR_K_DAY    = math.log(100.0 / EOL_SOH) / 3652.0
LFP_PRIOR_WEIGHT   = 180.0
STYLE = "seaborn-v0_8-whitegrid"

# ── Load data ──────────────────────────────────────────────────────────────────
rul    = pd.read_csv(os.path.join(ARTIFACTS_DIR, "rul_report.csv"))
trends = pd.read_csv(os.path.join(ARTIFACTS_DIR, "soh_trends.csv"))
trends = trends[trends["session_type"] == "discharge"].copy()


# =============================================================================
# PART 1: EFC vs Day RUL comparison
# =============================================================================

print("=" * 70)
print("PART 1 — EFC vs Day-Axis Blended RUL Comparison")
print("=" * 70)

efc_day_cols = [
    "registration_number", "fit_quality",
    "data_span_days", "efc_total", "avg_efc_per_day",
    "exp_k_day_blended", "rul_years_exp_day",   # day-axis
    "exp_k_blended",     "rul_years_exp",        # EFC-axis
]
cmp = rul[efc_day_cols].copy()

# Show how blended k was formed for each vehicle
cmp["k_day_prior_contrib_%"] = (
    LFP_PRIOR_WEIGHT /
    (cmp["data_span_days"].fillna(0) + LFP_PRIOR_WEIGHT) * 100
).round(1)

print(cmp.sort_values("fit_quality").to_string(index=False))

out_csv = os.path.join(ARTIFACTS_DIR, "efc_vs_day_rul.csv")
cmp.to_csv(out_csv, index=False)
print(f"\nSaved comparison: {out_csv}")

print("\nNote: rul_years_exp (EFC) is inflated because avg_efc_per_day ~0.03-0.08")
print("      at fleet age ~95 days. Day-axis (rul_years_exp_day) is more reliable.")


# =============================================================================
# PART 2: Exponential decay verification plots
# =============================================================================

print("\n" + "=" * 70)
print("PART 2 — Exponential Decay Verification (linear + semi-log)")
print("=" * 70)

# Pick vehicles
indicative    = rul[rul["fit_quality"] == "indicative"].copy()
insuff        = (rul[rul["fit_quality"] == "insufficient_data"]
                 .dropna(subset=["exp_A_day", "exp_k_day_blended"])
                 .nlargest(3, "data_span_days"))

vehicles = pd.concat([indicative, insuff], ignore_index=True)
print(f"\nVehicles selected ({len(vehicles)} total):")
print(vehicles[["registration_number", "fit_quality", "data_span_days",
                "exp_A_day", "exp_k_day_blended", "rul_years_exp_day"]].to_string(index=False))


def short_id(reg):
    return str(reg)[-7:]


def plot_vehicle_exp(row, ax_lin, ax_log, vdf):
    """Plot linear + semi-log panels for one vehicle."""
    reg       = row["registration_number"]
    A         = row["exp_A_day"]
    k         = row["exp_k_day_blended"]
    span      = row["data_span_days"]
    rul_yr    = row["rul_years_exp_day"]
    fit_qual  = row["fit_quality"]

    # ── Derived ────────────────────────────────────────────────────────────────
    if A > EOL_SOH and k > 0:
        t_eol = math.log(A / EOL_SOH) / k
    else:
        t_eol = span + 365 * 6

    # Extend axis well past EOL so curvature is obvious
    t_max = t_eol * 1.6
    t_curve = np.linspace(0.1, t_max, 600)
    soh_curve = A * np.exp(-k * t_curve)

    # ── Data scatter ───────────────────────────────────────────────────────────
    vdf_v = vdf[vdf["registration_number"] == reg].sort_values("date_days")
    t_data   = vdf_v["date_days"].values
    soh_data = vdf_v["soh_smooth"].values

    label_curve = f"A={A:.1f}, k={k:.2e}\nRUL={rul_yr:.1f} yr"

    for ax, yscale in [(ax_lin, "linear"), (ax_log, "log")]:
        ax.scatter(t_data, soh_data, s=14, alpha=0.6, color="#4682b4",
                   zorder=4, label="Observed SoH")
        ax.plot(t_curve, soh_curve, color="#e74c3c", linewidth=2,
                zorder=3, label=label_curve)

        # EOL line
        ax.axhline(EOL_SOH, color="red", linestyle="--", linewidth=1.0, alpha=0.7)
        ax.axvline(t_eol,   color="grey", linestyle=":", linewidth=1.0,
                   label=f"t_EOL={t_eol/365:.1f} yr")

        # Data span marker
        ax.axvline(span, color="#555", linestyle="-.", linewidth=1.0,
                   label=f"Data today ({span:.0f} d)")

        ax.set_xlabel("Days", fontsize=9)
        ax.set_xlim(0, t_max)

        if yscale == "linear":
            ax.set_ylim(max(60, EOL_SOH - 10), min(105, A + 3))
            ax.set_ylabel("SoH (%)", fontsize=9)
            ax.set_title(f"{short_id(reg)} — Linear scale\n({fit_qual})",
                         fontsize=9, fontweight="bold")
        else:
            ax.set_yscale("log")
            ax.set_ylim(max(50, EOL_SOH - 15), min(105, A + 5))
            ax.yaxis.set_major_formatter(ticker.ScalarFormatter())
            ax.set_ylabel("SoH (%, log scale)", fontsize=9)
            ax.set_title(
                f"{short_id(reg)} — Semi-log scale\n"
                "Straight line = confirmed exponential",
                fontsize=9, fontweight="bold"
            )

        ax.legend(fontsize=7, loc="upper right")
        ax.tick_params(labelsize=8)


# ── Fig A: one combined 6-vehicle grid (3 rows x 4 cols: lin|log per vehicle) ─
plt.style.use(STYLE)
n = len(vehicles)   # 6
fig, axes = plt.subplots(n, 2, figsize=(14, n * 3.5))

for idx, (_, row) in enumerate(vehicles.iterrows()):
    ax_lin = axes[idx, 0]
    ax_log = axes[idx, 1]
    plot_vehicle_exp(row, ax_lin, ax_log, trends)

fig.suptitle(
    "Exponential Decay Verification: 3 Indicative + 3 Insufficient-Data Vehicles\n"
    "Left: linear scale | Right: semi-log (straight line = true exponential)",
    fontsize=12, fontweight="bold", y=1.005
)
fig.tight_layout()
out_all = os.path.join(PLOTS_DIR, "verify_exp_all6.png")
fig.savefig(out_all, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"\nSaved: {out_all}")


# ── Fig B: individual 2-panel PNGs per vehicle ────────────────────────────────
for _, row in vehicles.iterrows():
    reg = row["registration_number"]
    fig2, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    plot_vehicle_exp(row, ax1, ax2, trends)
    fig2.suptitle(
        f"Exp Decay Verification — {short_id(reg)}\n"
        f"fit_quality: {row['fit_quality']}  |  span: {row['data_span_days']:.0f} days",
        fontsize=11, fontweight="bold"
    )
    fig2.tight_layout()
    fname = os.path.join(PLOTS_DIR, f"verify_exp_{short_id(reg)}.png")
    fig2.savefig(fname, dpi=150, bbox_inches="tight")
    plt.close(fig2)
    print(f"  Saved: {fname}")


print("\nDone.")
print(f"All outputs in: {PLOTS_DIR}")
