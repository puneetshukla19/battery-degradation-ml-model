"""
cell_diagnostics.py
===================
Identifies which physical battery pack (subsystem) and cell needs replacing
across the fleet, using the per-row probe and subsystem number fields in the
BMS data.

Signals used
------------
  min_cell_voltage_number              — index of the weakest cell within its subsystem
  min_cell_voltage_subsystem_number    — which subsystem (pack) hosts the weakest cell
  max_cell_voltage_number              — index of the strongest cell (for spread context)
  max_cell_voltage_subsystem_number    — subsystem of the strongest cell
  temperature_highest_probe_number     — probe running hottest
  temperature_highest_subsystem_number — subsystem running hottest
  insulation_resistance                — isolation fault indicator
  subsystem_voltage                    — per-subsystem pack voltage (imbalance)
  subsystem_current                    — per-subsystem current

Method
------
1. Load cycles.csv (session metadata) + BMS raw data
2. Stamp each BMS row with its session using merge_asof per vehicle
3. Per session: compute mode of weak_subsystem, weak_cell, hot_subsystem
4. Per vehicle, aggregate across sessions:
   - Frequency each subsystem appears as weakest / hottest (replacement score)
   - Trend: is the weakest cell's min_cell_voltage declining over time?
   - Pack voltage imbalance: std of subsystem_voltage
   - Insulation resistance trend
5. Output cell_health_ranking.csv + 5 diagnostic plots

Outputs
-------
  artifacts/cell_health_ranking.csv   — per-vehicle subsystem replacement ranking
  plots/cell_weak_subsystem_heatmap.png   — fleet heatmap: frequency of each subsystem being weakest
  plots/cell_min_voltage_trend.png        — weakest-cell voltage trend over time per vehicle
  plots/cell_hot_subsystem_heatmap.png    — fleet heatmap: frequency of each subsystem being hottest
  plots/cell_insulation_trend.png         — insulation resistance trend per vehicle
  plots/cell_pack_imbalance.png           — per-vehicle subsystem voltage std (pack imbalance)
"""

import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tqdm import tqdm

from config import DATA_DIR, ARTIFACTS_DIR, PLOTS_DIR, NOMINAL_CAPACITY_AH

BMS_FILE    = os.path.join(DATA_DIR, "bms_full_ultratech_intangles_more_cols_full.csv")
CYCLES_FILE = os.path.join(ARTIFACTS_DIR, "cycles.csv")
OUT_CSV     = os.path.join(ARTIFACTS_DIR, "cell_health_ranking.csv")

CACHE_FILE  = os.path.join(ARTIFACTS_DIR, "_cell_diag_cache.csv")
DPI         = 150
STYLE       = "seaborn-v0_8-whitegrid"

os.makedirs(PLOTS_DIR, exist_ok=True)

# ══════════════════════════════════════════════════════════════════════════════
# 1. Load data
# ══════════════════════════════════════════════════════════════════════════════

print("Loading cycles.csv ...")
cycles = pd.read_csv(CYCLES_FILE, low_memory=False)
cycles["start_time"] = pd.to_numeric(cycles["start_time"], errors="coerce")
cycles["end_time"]   = pd.to_numeric(cycles["end_time"],   errors="coerce")
cycles = cycles.dropna(subset=["start_time", "end_time", "registration_number"])
print(f"  Sessions: {len(cycles):,}  |  Vehicles: {cycles['registration_number'].nunique()}")

BMS_CELL_COLS = [
    "registration_number", "gps_time",
    "min_cell_voltage", "max_cell_voltage",
    "min_cell_voltage_number", "max_cell_voltage_number",
    "min_cell_voltage_subsystem_number", "max_cell_voltage_subsystem_number",
    "temperature_highest", "temperature_lowest",
    "temperature_highest_probe_number", "temperature_highest_subsystem_number",
    "temperature_lowest_probe_number", "temperature_lowest_subsystem_number",
    "insulation_resistance",
    "subsystem_voltage", "subsystem_number", "subsystem_current",
]

if not os.path.exists(CACHE_FILE):
    print("\nLoading BMS raw data (cell/subsystem columns) ...")
    bms = pd.read_csv(
        BMS_FILE,
        usecols=[c for c in BMS_CELL_COLS if c != "gps_time"] + ["gps_time"],
        low_memory=False,
        dtype={"registration_number": "str"},
    )
    bms["gps_time"] = pd.to_numeric(bms["gps_time"], errors="coerce")
    bms = bms.dropna(subset=["gps_time", "registration_number"])
    bms["gps_time"] = bms["gps_time"].astype("int64")
    print(f"  Rows: {len(bms):,}  |  Vehicles: {bms['registration_number'].nunique()}")
else:
    bms = None   # not needed — will load from cache


# ══════════════════════════════════════════════════════════════════════════════
# 2. Stamp each BMS row with its session (merge_asof per vehicle)
#    then compute per-session aggregates
# ══════════════════════════════════════════════════════════════════════════════

def _mode_int(s):
    m = s.dropna()
    return int(m.mode().iloc[0]) if len(m) > 0 else -1

def _consistency(s):
    """Fraction of rows where the modal value appears."""
    m = s.dropna()
    if len(m) == 0:
        return np.nan
    modal = m.mode().iloc[0]
    return (m == modal).mean()


if os.path.exists(CACHE_FILE):
    print(f"\nLoading cached session cell stats from {CACHE_FILE} ...")
    sess_cell = pd.read_csv(CACHE_FILE)
else:
    print("\nStamping BMS rows with sessions (vectorized per vehicle) ...")
    vehicles = sorted(
        set(cycles["registration_number"].unique()) &
        set(bms["registration_number"].unique())
    )

    sess_lkp = (
        cycles[["registration_number", "session_id", "start_time", "end_time"]]
        .dropna()
        .copy()
    )

    all_sess = []
    for reg in tqdm(vehicles, desc="Vehicles"):
        bms_v  = bms[bms["registration_number"] == reg].sort_values("gps_time").copy()
        sess_v = sess_lkp[sess_lkp["registration_number"] == reg].sort_values("start_time").copy()

        if bms_v.empty or sess_v.empty:
            continue

        merged = pd.merge_asof(
            bms_v,
            sess_v[["start_time", "end_time", "session_id"]].rename(
                columns={"start_time": "_ss", "end_time": "_se"}),
            left_on="gps_time", right_on="_ss",
            direction="backward",
        )
        merged = merged[
            merged["session_id"].notna() &
            (merged["gps_time"] <= merged["_se"])
        ].copy()

        if merged.empty:
            continue

        grp = merged.groupby("session_id")
        s = pd.DataFrame({
            "registration_number":      reg,
            "weak_subsystem_id":        grp["min_cell_voltage_subsystem_number"].agg(_mode_int),
            "weak_cell_id":             grp["min_cell_voltage_number"].agg(_mode_int),
            "weak_subsystem_consistency": grp["min_cell_voltage_subsystem_number"].agg(_consistency),
            "hot_subsystem_id":         grp["temperature_highest_subsystem_number"].agg(_mode_int),
            "hot_probe_id":             grp["temperature_highest_probe_number"].agg(_mode_int),
            "min_cell_voltage_mean":    grp["min_cell_voltage"].mean(),
            "max_cell_voltage_mean":    grp["max_cell_voltage"].mean(),
            "cell_spread_mean":         (grp["max_cell_voltage"].mean() - grp["min_cell_voltage"].mean()),
            "temp_highest_mean":        grp["temperature_highest"].mean(),
            "insulation_mean":          grp["insulation_resistance"].mean(),
            "subsystem_voltage_std":    grp["subsystem_voltage"].std(),
        }).reset_index()
        all_sess.append(s)

    sess_cell = pd.concat(all_sess, ignore_index=True)
    sess_cell.to_csv(CACHE_FILE, index=False)
    print(f"  Cached to {CACHE_FILE}")

# Merge with session metadata (timestamp, session_type)
sess_meta = cycles[["registration_number", "session_id", "start_time",
                     "session_type", "current_mean"]].copy()
sess_meta["start_time"] = pd.to_numeric(sess_meta["start_time"], errors="coerce")
sess_cell = sess_cell.merge(sess_meta, on=["registration_number", "session_id"], how="left")
sess_cell["date"] = pd.to_datetime(sess_cell["start_time"], unit="ms", utc=True).dt.tz_convert("Asia/Kolkata").dt.date

print(f"  Session cell stats: {len(sess_cell):,} rows")


# ══════════════════════════════════════════════════════════════════════════════
# 3. Per-vehicle subsystem ranking
#    For each vehicle:
#      - Which subsystem appears most often as the weakest (min_cell_voltage)?
#      - Which subsystem appears most often as the hottest?
#      - Is the weakest-cell voltage declining over time?
#      - What is the pack voltage imbalance (subsystem_voltage_std)?
# ══════════════════════════════════════════════════════════════════════════════

print("\nBuilding per-vehicle subsystem health rankings ...")

vehicle_rankings = []

for reg, vdf in tqdm(sess_cell.groupby("registration_number"), desc="Vehicles"):
    # Only discharge sessions for degradation analysis
    disc = vdf[vdf["session_type"] == "discharge"].copy() if "session_type" in vdf else vdf.copy()

    # ── Weakest subsystem frequency ──────────────────────────────────────────
    weak_counts = (
        disc["weak_subsystem_id"]
        .value_counts()
        .rename_axis("subsystem_id")
        .reset_index(name="n_sessions_weakest")
    )
    weak_counts["pct_sessions_weakest"] = (
        weak_counts["n_sessions_weakest"] / len(disc) * 100
    ).round(1)
    weak_counts["registration_number"] = reg
    weak_counts["signal"] = "voltage"

    # ── Hottest subsystem frequency ───────────────────────────────────────────
    hot_counts = (
        disc["hot_subsystem_id"]
        .value_counts()
        .rename_axis("subsystem_id")
        .reset_index(name="n_sessions_hottest")
    )
    hot_counts["pct_sessions_hottest"] = (
        hot_counts["n_sessions_hottest"] / len(disc) * 100
    ).round(1)

    # ── Mean min_cell_voltage per subsystem ───────────────────────────────────
    sub_voltage = (
        disc.groupby("weak_subsystem_id")["min_cell_voltage_mean"]
        .mean()
        .rename_axis("subsystem_id")
        .reset_index(name="mean_min_cell_v")
    )

    # ── Merge and score ───────────────────────────────────────────────────────
    rank_df = weak_counts.merge(hot_counts[["subsystem_id", "pct_sessions_hottest"]],
                                on="subsystem_id", how="outer")
    rank_df = rank_df.merge(sub_voltage, on="subsystem_id", how="outer")
    rank_df["registration_number"] = reg

    # Composite replacement score:
    # 60% weight on % sessions weakest + 20% on % sessions hottest
    # + 20% on how low the mean min cell voltage is (normalised, inverted)
    v_min = rank_df["mean_min_cell_v"].min()
    v_max = rank_df["mean_min_cell_v"].max()
    v_range = v_max - v_min if v_max > v_min else 1.0
    rank_df["v_score"] = (v_max - rank_df["mean_min_cell_v"]) / v_range * 100

    rank_df["pct_sessions_weakest"]  = rank_df["pct_sessions_weakest"].fillna(0)
    rank_df["pct_sessions_hottest"]  = rank_df["pct_sessions_hottest"].fillna(0)
    rank_df["v_score"]               = rank_df["v_score"].fillna(0)

    rank_df["replacement_score"] = (
        0.60 * rank_df["pct_sessions_weakest"] +
        0.20 * rank_df["pct_sessions_hottest"] +
        0.20 * rank_df["v_score"]
    ).round(2)

    rank_df = rank_df.sort_values("replacement_score", ascending=False).reset_index(drop=True)
    rank_df["rank"] = rank_df.index + 1

    # ── Pack-level stats for this vehicle ─────────────────────────────────────
    rank_df["mean_insulation_ohm"]  = vdf["insulation_mean"].mean()
    rank_df["mean_subsystem_v_std"] = vdf["subsystem_voltage_std"].mean()
    rank_df["n_sessions_total"]     = len(disc)

    vehicle_rankings.append(rank_df)

ranking = pd.concat(vehicle_rankings, ignore_index=True)
ranking.to_csv(OUT_CSV, index=False)
print(f"\n  Saved: {OUT_CSV}")

# ── Console summary: top replacement candidates ───────────────────────────────
print("\n" + "=" * 72)
print("  TOP REPLACEMENT CANDIDATES (highest replacement_score, rank=1 per vehicle)")
print("=" * 72)
top1 = (
    ranking[ranking["rank"] == 1]
    .sort_values("replacement_score", ascending=False)
    [["registration_number", "subsystem_id", "replacement_score",
      "pct_sessions_weakest", "pct_sessions_hottest", "mean_min_cell_v",
      "mean_insulation_ohm", "n_sessions_total"]]
)
print(top1.to_string(index=False))

# ── Insulation resistance summary ─────────────────────────────────────────────
ins_veh = (
    sess_cell.groupby("registration_number")["insulation_mean"].mean()
    .sort_values()
)
print("\n  Insulation resistance — vehicles with lowest mean (Ohm):")
print(ins_veh.head(10).to_string())


# ══════════════════════════════════════════════════════════════════════════════
# 4. Plots
# ══════════════════════════════════════════════════════════════════════════════

vehicles_sorted = top1["registration_number"].tolist()

# ── Plot 1: Weak subsystem heatmap (fleet × subsystem) ───────────────────────
print("\nGenerating plots ...")

# Build pivot: rows = vehicles, cols = subsystem_id, values = pct_sessions_weakest
pivot_weak = (
    ranking[ranking["signal"] == "voltage"]
    .pivot_table(index="registration_number", columns="subsystem_id",
                 values="pct_sessions_weakest", aggfunc="first", fill_value=0)
)

with plt.style.context(STYLE):
    fig, ax = plt.subplots(figsize=(max(10, len(pivot_weak.columns) * 0.6),
                                    max(8, len(pivot_weak) * 0.35)))
    im = ax.imshow(pivot_weak.values, aspect="auto", cmap="YlOrRd", vmin=0, vmax=100)
    ax.set_xticks(range(len(pivot_weak.columns)))
    ax.set_xticklabels([f"Sub {c}" for c in pivot_weak.columns], rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(pivot_weak.index)))
    ax.set_yticklabels(pivot_weak.index, fontsize=7)
    plt.colorbar(im, ax=ax, label="% discharge sessions where subsystem is weakest")
    ax.set_title("Fleet Cell Health — Frequency of Each Subsystem Being Weakest\n"
                 "(darker = more often the weakest subsystem — higher replacement priority)",
                 fontsize=12, fontweight="bold")
    fig.tight_layout()
    out = os.path.join(PLOTS_DIR, "cell_weak_subsystem_heatmap.png")
    fig.savefig(out, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out}")


# ── Plot 2: Pivot — hot subsystem heatmap ────────────────────────────────────
pivot_hot = (
    sess_cell[sess_cell["session_type"] == "discharge"]
    .groupby(["registration_number", "hot_subsystem_id"])
    .size()
    .reset_index(name="count")
)
total_per_veh = (
    sess_cell[sess_cell["session_type"] == "discharge"]
    .groupby("registration_number")
    .size()
    .rename("total")
)
pivot_hot = pivot_hot.merge(total_per_veh, on="registration_number")
pivot_hot["pct"] = pivot_hot["count"] / pivot_hot["total"] * 100
pivot_hot_wide = pivot_hot.pivot_table(
    index="registration_number", columns="hot_subsystem_id",
    values="pct", fill_value=0
)

with plt.style.context(STYLE):
    fig, ax = plt.subplots(figsize=(max(10, len(pivot_hot_wide.columns) * 0.6),
                                    max(8, len(pivot_hot_wide) * 0.35)))
    im = ax.imshow(pivot_hot_wide.values, aspect="auto", cmap="YlOrRd", vmin=0, vmax=100)
    ax.set_xticks(range(len(pivot_hot_wide.columns)))
    ax.set_xticklabels([f"Sub {c}" for c in pivot_hot_wide.columns], rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(pivot_hot_wide.index)))
    ax.set_yticklabels(pivot_hot_wide.index, fontsize=7)
    plt.colorbar(im, ax=ax, label="% discharge sessions where subsystem is hottest")
    ax.set_title("Fleet Thermal Health — Frequency of Each Subsystem Running Hottest\n"
                 "(darker = thermal hotspot — potential cooling or cell degradation issue)",
                 fontsize=12, fontweight="bold")
    fig.tight_layout()
    out = os.path.join(PLOTS_DIR, "cell_hot_subsystem_heatmap.png")
    fig.savefig(out, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out}")


# ── Plot 3: Min cell voltage trend over time — top 10 vehicles by score ───────
top_vehicles = top1.head(10)["registration_number"].tolist()

with plt.style.context(STYLE):
    fig, axes = plt.subplots(2, 5, figsize=(20, 8), sharey=False)
    fig.suptitle("Weakest Cell Voltage Trend — Top 10 Vehicles by Replacement Score",
                 fontsize=13, fontweight="bold")
    for ax, reg in zip(axes.flat, top_vehicles):
        vdf = sess_cell[
            (sess_cell["registration_number"] == reg) &
            (sess_cell["session_type"] == "discharge")
        ].copy()
        if vdf.empty:
            ax.set_title(reg, fontsize=8)
            continue
        vdf = vdf.sort_values("start_time")
        vdf["date_num"] = pd.to_datetime(vdf["start_time"], unit="ms").dt.to_period("W").dt.start_time
        weekly = vdf.groupby("date_num")["min_cell_voltage_mean"].mean().reset_index()
        ax.plot(weekly["date_num"], weekly["min_cell_voltage_mean"],
                color="#d62728", lw=1.5, marker="o", markersize=3)
        ax.set_title(reg, fontsize=8, fontweight="bold")
        ax.set_xlabel("Date", fontsize=7)
        ax.set_ylabel("Mean min cell V (V)", fontsize=7)
        ax.tick_params(axis="x", rotation=30, labelsize=6)
        ax.tick_params(axis="y", labelsize=7)
    for ax in axes.flat[len(top_vehicles):]:
        ax.set_visible(False)
    fig.tight_layout()
    out = os.path.join(PLOTS_DIR, "cell_min_voltage_trend.png")
    fig.savefig(out, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out}")


# ── Plot 4: Insulation resistance trend per vehicle ───────────────────────────
with plt.style.context(STYLE):
    fig, axes = plt.subplots(2, 5, figsize=(20, 8), sharey=False)
    fig.suptitle("Insulation Resistance Trend — Top 10 Vehicles by Replacement Score",
                 fontsize=13, fontweight="bold")
    for ax, reg in zip(axes.flat, top_vehicles):
        vdf = sess_cell[sess_cell["registration_number"] == reg].copy()
        vdf = vdf.dropna(subset=["insulation_mean"]).sort_values("start_time")
        if vdf.empty:
            ax.set_title(reg, fontsize=8)
            continue
        vdf["date_num"] = pd.to_datetime(vdf["start_time"], unit="ms").dt.to_period("W").dt.start_time
        weekly = vdf.groupby("date_num")["insulation_mean"].mean().reset_index()
        ax.plot(weekly["date_num"], weekly["insulation_mean"],
                color="#9467bd", lw=1.5, marker="o", markersize=3)
        ax.axhline(500, color="#d62728", ls="--", lw=1.2, label="500 Ohm warn")
        ax.set_title(reg, fontsize=8, fontweight="bold")
        ax.set_xlabel("Date", fontsize=7)
        ax.set_ylabel("Insulation (Ohm)", fontsize=7)
        ax.tick_params(axis="x", rotation=30, labelsize=6)
        ax.tick_params(axis="y", labelsize=7)
    for ax in axes.flat[len(top_vehicles):]:
        ax.set_visible(False)
    fig.tight_layout()
    out = os.path.join(PLOTS_DIR, "cell_insulation_trend.png")
    fig.savefig(out, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out}")


# ── Plot 5: Pack voltage imbalance per vehicle (subsystem_voltage_std) ─────────
imbalance = (
    sess_cell[sess_cell["session_type"] == "discharge"]
    .groupby("registration_number")["subsystem_voltage_std"]
    .mean()
    .sort_values(ascending=False)
    .dropna()
)

with plt.style.context(STYLE):
    fig, ax = plt.subplots(figsize=(14, 5))
    colors = ["#d62728" if v > imbalance.quantile(0.75) else "#1f77b4"
              for v in imbalance.values]
    ax.bar(range(len(imbalance)), imbalance.values, color=colors, edgecolor="white")
    ax.axhline(imbalance.quantile(0.75), color="#ff7f0e", ls="--", lw=1.5,
               label=f"p75 = {imbalance.quantile(0.75):.2f} V")
    ax.set_xticks(range(len(imbalance)))
    ax.set_xticklabels(imbalance.index, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Mean subsystem voltage std (V)", fontsize=10)
    ax.set_title("Pack Voltage Imbalance per Vehicle\n"
                 "(high std = subsystems not in equilibrium — cell degradation or connection issue)",
                 fontsize=12, fontweight="bold")
    ax.legend(fontsize=9)
    fig.tight_layout()
    out = os.path.join(PLOTS_DIR, "cell_pack_imbalance.png")
    fig.savefig(out, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out}")


# ══════════════════════════════════════════════════════════════════════════════
# 5. ML feature summary — what gets added to cycles.csv by data_prep_1.py
# ══════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 72)
print("  NEW SESSION-LEVEL FEATURES ADDED TO cycles.csv (via data_prep_1.py)")
print("=" * 72)
print("""
  weak_subsystem_id          — subsystem most often showing min_cell_voltage in session
  weak_cell_id               — cell index most often showing min_cell_voltage
  weak_subsystem_consistency — fraction of rows where modal subsystem is weakest (0–1)
  hot_subsystem_id           — subsystem most often showing max temperature in session
  hot_probe_id               — temperature probe most often showing max temperature
  subsystem_voltage_std      — std of subsystem voltages within session (pack imbalance)

  These feed into anomaly.py's LightGBM and Isolation Forest models as:
    - Recurring weak_subsystem_id across sessions → degraded pack
    - Low weak_subsystem_consistency → spread degradation (multiple weak cells)
    - Rising subsystem_voltage_std → growing pack imbalance
    - Persistent hot_subsystem_id → thermal issue in a specific pack
""")

print("Done.")
