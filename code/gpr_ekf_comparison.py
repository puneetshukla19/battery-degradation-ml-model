"""
gpr_ekf_comparison.py
Two diagnostic plots comparing GPR predicted SoH against EKF SoH.

Outputs
-------
  plots/gpr_vs_ekf_per_vehicle.png  — per-vehicle panel: GPR + EKF + QG obs vs time
  plots/gpr_vs_ekf_scatter.png      — fleet scatter: EKF SoH (x) vs GPR pred (y) + OLS slope
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.lines import Line2D
from scipy import stats

from config import ARTIFACTS_DIR, PLOTS_DIR

GPR_CSV = os.path.join(ARTIFACTS_DIR, "gpr_predictions.csv")
EKF_CSV = os.path.join(ARTIFACTS_DIR, "ekf_soh.csv")

# ── Load ───────────────────────────────────────────────────────────────────────
print("Loading data ...")
gpr = pd.read_csv(GPR_CSV, low_memory=False)
ekf = pd.read_csv(EKF_CSV, low_memory=False)

for df in [gpr, ekf]:
    try:
        df["start_dt"] = (
            pd.to_datetime(df["start_time"], unit="ms", utc=True)
            .dt.tz_convert("Asia/Kolkata")
        )
    except Exception:
        df["start_dt"] = pd.to_datetime(df["start_time"], errors="coerce")

gpr["is_quality_gated"] = gpr["is_quality_gated"].map(
    lambda v: str(v).strip().lower() in ("true", "1")
)

merged = gpr.merge(
    ekf[["registration_number", "session_id", "ekf_soh", "ekf_soh_std"]],
    on=["registration_number", "session_id"],
    how="left",
)
print(f"  {len(merged):,} sessions   |   {merged['ekf_soh'].notna().sum():,} have EKF SoH")

vehicles = sorted(merged["registration_number"].unique())
n_veh = len(vehicles)

# ── Plot 1: Per-vehicle panel ──────────────────────────────────────────────────
print("Building per-vehicle panel ...")
NCOLS = 6
NROWS = int(np.ceil(n_veh / NCOLS))

fig, axes = plt.subplots(NROWS, NCOLS, figsize=(NCOLS * 4.0, NROWS * 3.0))
axes = np.array(axes).flatten()

fig.suptitle(
    "GPR Predicted SoH vs EKF SoH — Per Vehicle\n"
    "Blue: GPR   |   Orange: EKF (±1σ shaded)   |   Red dots: Quality-gated cycle observations",
    fontsize=11, fontweight="bold", y=1.003,
)

for i, reg in enumerate(vehicles):
    ax  = axes[i]
    vdf = merged[merged["registration_number"] == reg].sort_values("start_dt")
    ekf_rows = vdf[vdf["ekf_soh"].notna()].sort_values("start_dt")
    qg_rows  = vdf[vdf["is_quality_gated"]].sort_values("start_dt")

    # GPR SoH — all sessions (thin)
    ax.plot(vdf["start_dt"], vdf["gpr_soh_pred"],
            c="#4E79A7", lw=0.85, alpha=0.65)

    # EKF SoH — charging sessions with ±1σ band
    if len(ekf_rows) > 1:
        ax.plot(ekf_rows["start_dt"], ekf_rows["ekf_soh"],
                c="#F28E2B", lw=1.4, alpha=0.9)
        ax.fill_between(
            ekf_rows["start_dt"],
            ekf_rows["ekf_soh"] - ekf_rows["ekf_soh_std"],
            ekf_rows["ekf_soh"] + ekf_rows["ekf_soh_std"],
            alpha=0.13, color="#F28E2B",
        )

    # Quality-gated observations
    if len(qg_rows) > 0:
        ax.scatter(qg_rows["start_dt"], qg_rows["cycle_soh"],
                   c="#E15759", s=7, zorder=6, alpha=0.8)

    # Title: vehicle + MAE between GPR and EKF
    both = vdf[vdf["ekf_soh"].notna() & vdf["gpr_soh_pred"].notna()]
    if len(both) >= 2:
        mae_ekf = np.mean(np.abs(both["gpr_soh_pred"] - both["ekf_soh"]))
        ax.set_title(f"{reg}\nGPR–EKF MAE={mae_ekf:.2f}%", fontsize=6.3, pad=2)
    else:
        ax.set_title(reg, fontsize=7, pad=2)

    y_min = max(79, min(
        vdf["gpr_soh_pred"].min() if vdf["gpr_soh_pred"].notna().any() else 95,
        ekf_rows["ekf_soh"].min() if len(ekf_rows) > 0 else 95,
    ) - 1)
    ax.set_ylim(y_min, 102)
    ax.axhline(80, c="red", lw=0.6, linestyle=":", alpha=0.45)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b%y"))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    ax.tick_params(labelsize=4.8, axis="x", rotation=30)
    ax.tick_params(labelsize=4.8, axis="y")
    ax.set_ylabel("SoH (%)", fontsize=5.5)
    ax.grid(True, alpha=0.22)

for j in range(i + 1, len(axes)):
    axes[j].set_visible(False)

leg_els = [
    Line2D([0], [0], color="#4E79A7", lw=1.5, label="GPR Predicted SoH"),
    Line2D([0], [0], color="#F28E2B", lw=1.5, label="EKF SoH  (shaded ±1σ)"),
    Line2D([0], [0], marker="o", color="w", markerfacecolor="#E15759", ms=7,
           label="Quality-gated cycle obs"),
]
fig.legend(handles=leg_els, loc="lower center", ncol=3, fontsize=9,
           bbox_to_anchor=(0.5, -0.012), framealpha=0.95)

fig.tight_layout()
out1 = os.path.join(PLOTS_DIR, "gpr_vs_ekf_per_vehicle.png")
fig.savefig(out1, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"  Saved: {out1}")


# ── Plot 2: Scatterplot — EKF SoH vs GPR predicted SoH ───────────────────────
print("Building scatter plot ...")

sc_all = merged[merged["ekf_soh"].notna() & merged["gpr_soh_pred"].notna()].copy()
sc_qg  = sc_all[sc_all["is_quality_gated"]].copy()

def _ols_overlay(ax, x, y, color="black", lw=2):
    slope, intercept, r, p, se = stats.linregress(x, y)
    x_fit = np.linspace(x.min(), x.max(), 300)
    label = (
        f"OLS  slope = {slope:.3f}   R² = {r**2:.3f}\n"
        f"y = {slope:.3f}·x + {intercept:.2f}"
    )
    ax.plot(x_fit, slope * x_fit + intercept, "-", c=color, lw=lw, zorder=8, label=label)
    return slope, r**2

fig, axes = plt.subplots(1, 2, figsize=(14, 6.5))
fig.suptitle("EKF SoH vs GPR Predicted SoH", fontsize=13, fontweight="bold")

# ── Left: all sessions with EKF overlap, coloured by vehicle ──────────────────
ax = axes[0]
unique_regs = sorted(sc_all["registration_number"].unique())
cmap_tab    = plt.get_cmap("tab20", len(unique_regs))
reg_color   = {r: cmap_tab(i) for i, r in enumerate(unique_regs)}
pt_colors   = [reg_color[r] for r in sc_all["registration_number"]]

ax.scatter(sc_all["ekf_soh"], sc_all["gpr_soh_pred"],
           c=pt_colors, alpha=0.22, s=5, rasterized=True)

slope_all, r2_all = _ols_overlay(ax, sc_all["ekf_soh"].values, sc_all["gpr_soh_pred"].values)

lo = min(sc_all["ekf_soh"].min(), sc_all["gpr_soh_pred"].min()) - 0.5
hi = max(sc_all["ekf_soh"].max(), sc_all["gpr_soh_pred"].max()) + 0.5
ax.plot([lo, hi], [lo, hi], "--", c="#888888", lw=1.3, alpha=0.75, label="Ideal 1:1")
ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
ax.set_xlabel("EKF SoH — reference (%)", fontsize=11)
ax.set_ylabel("GPR Predicted SoH (%)", fontsize=11)
ax.set_title(f"All sessions with EKF overlap  (n = {len(sc_all):,})", fontsize=10)
ax.legend(fontsize=9, loc="upper left")
ax.grid(True, alpha=0.3)
ax.set_aspect("equal", adjustable="box")

# ── Right: quality-gated only, coloured by cum_efc ────────────────────────────
ax2 = axes[1]
efc_vals = sc_qg["cum_efc"].fillna(0).values
sc2 = ax2.scatter(
    sc_qg["ekf_soh"], sc_qg["gpr_soh_pred"],
    c=efc_vals, cmap="plasma", alpha=0.55, s=20, zorder=4,
)
cb = plt.colorbar(sc2, ax=ax2, label="Cumulative EFC", pad=0.02)
cb.ax.tick_params(labelsize=8)

if len(sc_qg) > 2:
    slope_qg, r2_qg = _ols_overlay(
        ax2, sc_qg["ekf_soh"].values, sc_qg["gpr_soh_pred"].values
    )

lo_q = min(sc_qg["ekf_soh"].min(), sc_qg["gpr_soh_pred"].min()) - 0.5
hi_q = max(sc_qg["ekf_soh"].max(), sc_qg["gpr_soh_pred"].max()) + 0.5
ax2.plot([lo_q, hi_q], [lo_q, hi_q], "--", c="#888888", lw=1.3, alpha=0.75, label="Ideal 1:1")
ax2.set_xlim(lo_q, hi_q); ax2.set_ylim(lo_q, hi_q)
ax2.set_xlabel("EKF SoH — reference (%)", fontsize=11)
ax2.set_ylabel("GPR Predicted SoH (%)", fontsize=11)
ax2.set_title(f"Quality-gated sessions only  (n = {len(sc_qg):,})", fontsize=10)
ax2.legend(fontsize=9, loc="upper left")
ax2.grid(True, alpha=0.3)
ax2.set_aspect("equal", adjustable="box")

fig.tight_layout()
out2 = os.path.join(PLOTS_DIR, "gpr_vs_ekf_scatter.png")
fig.savefig(out2, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"  Saved: {out2}")
print("Done.")
