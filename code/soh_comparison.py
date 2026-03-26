"""
soh_comparison.py
=================
Standalone analysis: compare capacity_soh derived from two current/voltage sources.

  Source A (existing): voltage / current from the main BMS table  (cycles.csv)
  Source B (new):      hves1_voltage_level / hves1_current from bms_ultratech_current_full.csv

Method
------
Per vehicle, we use merge_asof to stamp each hves1 row with the session it belongs to
(nearest session start_time, backward direction), then discard rows outside the session
end_time. We then coulomb-count within each session, derive capacity_soh_B, and compare
with the existing capacity_soh_A.

ref_capacity_ah note
--------------------
Config uses NOMINAL_CAPACITY_AH = 436 Ah (2P × 218 Ah cells, 180S pack).
  Pack energy check: 436 Ah × 648 V (true nominal at 3.6 V/cell × 180S) = 282.5 kWh  ✓
  The 630 V in config is a mid-SoC conservative value.
  The data-driven p90 approach for ref_capacity_ah is better than the hard nominal.

Outputs
-------
  plots/soh_comparison_discharge.png    — SOH distribution + delta (discharge)
  plots/soh_comparison_charging.png     — SOH distribution + delta (charging)
  plots/soh_comparison_scatter.png      — scatter: SOH_A vs SOH_B per session
  plots/soh_ref_capacity_dist.png       — ref_capacity_ah distribution vs nominal
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

from config import ARTIFACTS_DIR, PLOTS_DIR, DATA_DIR, NOMINAL_CAPACITY_AH, NOMINAL_VOLTAGE_V

CURRENT_FILE     = os.path.join(DATA_DIR, "bms_ultratech_current_full.csv")
CYCLES_FILE      = os.path.join(ARTIFACTS_DIR, "cycles.csv")

MAX_DT_MIN       = 5.0          # same cap as data_prep_1.py
REGEN_DT_MAX_SEC = 5.0
CHARGE_A         = -50.0        # same threshold as config
CHARGE_EFF       = 0.97
MIN_SOC_RANGE_DISC = 15.0
MIN_SOC_RANGE_CHG  = 10.0
DPI = 150
STYLE = "seaborn-v0_8-whitegrid"


# ══════════════════════════════════════════════════════════════════════════════
# 1. Load data
# ══════════════════════════════════════════════════════════════════════════════

print("Loading cycles.csv ...")
cycles = pd.read_csv(CYCLES_FILE, low_memory=False)
cycles["start_time"] = pd.to_numeric(cycles["start_time"], errors="coerce")
cycles["end_time"]   = pd.to_numeric(cycles["end_time"],   errors="coerce")
cycles = cycles.dropna(subset=["start_time", "end_time", "registration_number"])
print(f"  Sessions: {len(cycles):,}  |  Vehicles: {cycles['registration_number'].nunique()}")

print("\nLoading supplementary current table ...")
curr = pd.read_csv(
    CURRENT_FILE,
    usecols=["registration_number", "timestamp", "hves1_voltage_level", "hves1_current"],
    dtype={"hves1_voltage_level": "float32", "hves1_current": "float32",
           "registration_number": "str"},
)
curr["timestamp"] = pd.to_numeric(curr["timestamp"], errors="coerce")
curr = curr.dropna(subset=["timestamp", "registration_number",
                            "hves1_voltage_level", "hves1_current"])
curr["timestamp"] = curr["timestamp"].astype("int64")
print(f"  Rows: {len(curr):,}  |  Vehicles: {curr['registration_number'].nunique()}")


# ══════════════════════════════════════════════════════════════════════════════
# 2. Vectorized session assignment + coulomb counting
#    Strategy: merge_asof stamps every hves1 row with the session whose
#    start_time <= timestamp (backward), then filter timestamp <= end_time.
#    Coulomb counting runs on the resulting grouped frame.
# ══════════════════════════════════════════════════════════════════════════════

CACHE_FILE = os.path.join(ARTIFACTS_DIR, "_soh_comparison_cache.csv")

print("\nAssigning sessions to hves1 rows (vectorized per vehicle) ...")

vehicles = sorted(
    set(cycles["registration_number"].unique()) &
    set(curr["registration_number"].unique())
)

# Session lookup table: only the columns needed
sess_lkp = (
    cycles[["registration_number", "session_id", "start_time", "end_time",
            "session_type", "current_mean"]]
    .dropna(subset=["start_time", "end_time"])
    .copy()
)

if os.path.exists(CACHE_FILE):
    print(f"  Loading cached session stats from {CACHE_FILE} ...")
    sess_stats_all = pd.read_csv(CACHE_FILE)
    cycles = cycles.merge(sess_stats_all, on=["registration_number", "session_id"], how="left")
    matched = (cycles["n_rows_B"] > 0).sum()
    print(f"  Sessions with hves1 data: {matched:,} / {len(cycles):,}")
else:
    all_sess_stats = []

for reg in tqdm(vehicles, desc="Vehicles") if not os.path.exists(CACHE_FILE) else []:
    curr_v = curr[curr["registration_number"] == reg].sort_values("timestamp").copy()
    sess_v = sess_lkp[sess_lkp["registration_number"] == reg].sort_values("start_time").copy()

    if curr_v.empty or sess_v.empty:
        continue

    # Stamp each hves1 row with the last session that started <= its timestamp
    merged = pd.merge_asof(
        curr_v,
        sess_v[["start_time", "end_time", "session_id", "session_type"]].rename(
            columns={"start_time": "_sess_start", "end_time": "_sess_end"}),
        left_on="timestamp",
        right_on="_sess_start",
        direction="backward",
    )

    # Keep only rows that fall within [start, end] of their matched session
    merged = merged[
        merged["session_id"].notna() &
        (merged["timestamp"] <= merged["_sess_end"])
    ].copy()

    if merged.empty:
        continue

    # Per-session coulomb counting
    merged = merged.sort_values(["session_id", "timestamp"])
    dt_hr       = merged.groupby("session_id")["timestamp"].diff().fillna(0) / 3_600_000
    dt_hr       = dt_hr.clip(0, MAX_DT_MIN / 60)
    dt_hr_regen = dt_hr.clip(0, REGEN_DT_MAX_SEC / 3_600)

    curr_vals = merged["hves1_current"].values
    disc_mask = curr_vals > 0
    chg_mask  = curr_vals < CHARGE_A

    merged["_dq_disc"]  = np.where(disc_mask,  curr_vals * dt_hr.values,              0.0)
    merged["_dq_chg"]   = np.where(chg_mask,   np.abs(curr_vals * dt_hr_regen.values), 0.0)

    sess_stats = merged.groupby("session_id").agg(
        capacity_ah_disc_B  = ("_dq_disc",           "sum"),
        capacity_ah_chg_B   = ("_dq_chg",            "sum"),
        voltage_mean_B      = ("hves1_voltage_level", "mean"),
        n_rows_B            = ("timestamp",           "count"),
    ).reset_index()
    sess_stats["registration_number"] = reg

    all_sess_stats.append(sess_stats)

if not os.path.exists(CACHE_FILE):
    print("  Merging session stats ...")
    if all_sess_stats:
        sess_stats_all = pd.concat(all_sess_stats, ignore_index=True)
        sess_stats_all.to_csv(CACHE_FILE, index=False)
        print(f"  Cached to {CACHE_FILE}")
        cycles = cycles.merge(sess_stats_all, on=["registration_number", "session_id"], how="left")
        matched = (cycles["n_rows_B"] > 0).sum()
        print(f"  Sessions with hves1 data: {matched:,} / {len(cycles):,}")
    else:
        print("  WARNING: no hves1 sessions matched.")
        cycles["capacity_ah_disc_B"]  = np.nan
        cycles["capacity_ah_chg_B"]   = np.nan
        cycles["voltage_mean_B"]      = np.nan
        cycles["n_rows_B"]            = 0


# ══════════════════════════════════════════════════════════════════════════════
# 3. Derive capacity_soh_B
# ══════════════════════════════════════════════════════════════════════════════

print("\nRecomputing Source A with fixed 436 Ah reference ...")
# cycles.csv was generated with data-driven p90 ref (~197 Ah). Recompute here
# using block_capacity_ah / block_soc_diff already in cycles.csv so we can
# immediately compare without re-running data_prep_1.py.
cycles["capacity_soh_disc_A436"] = np.nan
if "block_id" in cycles.columns and "block_capacity_ah" in cycles.columns:
    blk_A = (
        cycles[
            (cycles["session_type"] == "discharge") &
            (cycles["current_mean"] > 0) &
            cycles["block_id"].notna() &
            (cycles["block_soc_diff"] < 0) &
            (cycles["block_capacity_ah"] > 0)
        ]
        .drop_duplicates(subset=["registration_number", "block_id"])
        [["registration_number", "block_id", "block_capacity_ah", "block_soc_diff"]]
        .copy()
    )
    blk_A["dod"]      = blk_A["block_soc_diff"].abs()
    blk_A["norm_cap"] = (blk_A["block_capacity_ah"] / (blk_A["dod"] / 100.0)).replace(
        [np.inf, -np.inf], np.nan)
    quality_A436 = (blk_A["dod"] >= MIN_SOC_RANGE_DISC) & blk_A["norm_cap"].notna()
    blk_A["soh_A436"] = np.nan
    blk_A.loc[quality_A436, "soh_A436"] = (
        (blk_A.loc[quality_A436, "norm_cap"] / NOMINAL_CAPACITY_AH * 100).clip(0, 100)
    )
    blk_A_idx = blk_A.set_index(["registration_number", "block_id"])["soh_A436"]
    disc_rows_A = cycles[
        (cycles["session_type"] == "discharge") &
        (cycles["current_mean"] > 0) &
        cycles["block_id"].notna() &
        (cycles["block_soc_diff"] < 0)
    ]
    cycles.loc[disc_rows_A.index, "capacity_soh_disc_A436"] = [
        blk_A_idx.get(k, np.nan)
        for k in zip(disc_rows_A["registration_number"], disc_rows_A["block_id"])
    ]
    a436 = cycles["capacity_soh_disc_A436"].dropna()
    print(f"  capacity_soh_disc_A436: n={len(a436):,}  mean={a436.mean():.2f}%  "
          f"median={a436.median():.2f}%  std={a436.std():.2f}%")

print("\nDeriving capacity_soh_B ...")

# ── Discharge — block-level (matches Source A: aggregate Ah per block) ─────────
cycles["capacity_soh_disc_B"] = np.nan

has_blocks = "block_id" in cycles.columns and cycles["block_id"].notna().any()
if has_blocks:
    disc_rows = cycles[
        (cycles["session_type"] == "discharge") &
        (cycles["current_mean"] > 0) &
        cycles["block_id"].notna() &
        cycles["capacity_ah_disc_B"].notna()
    ].copy()

    block_B = (
        disc_rows.groupby(["registration_number", "block_id"])
        .agg(
            block_ah_B     = ("capacity_ah_disc_B", "sum"),
            block_soc_diff = ("block_soc_diff",     "first"),
            ref_cap        = ("ref_capacity_ah",    "first"),
        )
        .reset_index()
    )
    block_B["dod_B"]      = block_B["block_soc_diff"].abs()
    block_B["norm_cap_B"] = (
        block_B["block_ah_B"] / (block_B["dod_B"] / 100.0)
    ).replace([np.inf, -np.inf], np.nan)

    quality_blk = (
        (block_B["dod_B"]      >= MIN_SOC_RANGE_DISC) &
        (block_B["block_ah_B"] >  0) &
        block_B["norm_cap_B"].notna()
    )
    block_B["capacity_soh_disc_B"] = np.nan
    block_B.loc[quality_blk, "capacity_soh_disc_B"] = (
        (block_B.loc[quality_blk, "norm_cap_B"] /
         block_B.loc[quality_blk, "ref_cap"] * 100).clip(0, 100)
    )

    block_soh_B_idx = block_B.set_index(
        ["registration_number", "block_id"])["capacity_soh_disc_B"]
    disc_sess = cycles[
        (cycles["session_type"] == "discharge") &
        (cycles["current_mean"] > 0) &
        cycles["block_id"].notna() &
        (cycles["block_soc_diff"] < 0)
    ]
    keys = list(zip(disc_sess["registration_number"], disc_sess["block_id"]))
    cycles.loc[disc_sess.index, "capacity_soh_disc_B"] = [
        block_soh_B_idx.get(k, np.nan) for k in keys
    ]
    print(f"  Discharge B (block): {quality_blk.sum():,} quality blocks | "
          f"norm_cap_B p90={block_B.loc[quality_blk,'norm_cap_B'].quantile(0.9):.1f} Ah")
else:
    # Fallback: session-level
    dod = cycles["soc_diff"].abs()
    disc_quality = (
        (cycles["session_type"] == "discharge") &
        (cycles["current_mean"] > 0) & (cycles["soc_diff"] < 0) &
        (dod >= MIN_SOC_RANGE_DISC) & cycles["capacity_ah_disc_B"].notna()
    )
    norm_disc_B = (cycles["capacity_ah_disc_B"] / (dod / 100.0)).replace(
        [np.inf, -np.inf], np.nan)
    cycles.loc[disc_quality, "capacity_soh_disc_B"] = (
        (norm_disc_B[disc_quality] /
         cycles.loc[disc_quality, "ref_capacity_ah"] * 100).clip(0, 100)
    )

# ── Charging — session-level (same as Source A) ───────────────────────────────
chg_soc = cycles["soc_range"].abs()
chg_quality = (
    (cycles["session_type"] == "charging") &
    (cycles["soc_diff"] > 0) &
    (chg_soc >= MIN_SOC_RANGE_CHG) &
    cycles["capacity_ah_chg_B"].notna()
)
norm_chg_B = (
    (cycles["capacity_ah_chg_B"] * CHARGE_EFF) / (chg_soc / 100.0)
).replace([np.inf, -np.inf], np.nan)
cycles["capacity_soh_chg_B"] = np.nan
cycles.loc[chg_quality, "capacity_soh_chg_B"] = (
    (norm_chg_B[chg_quality] / cycles.loc[chg_quality, "ref_capacity_ah"] * 100)
    .clip(0, 100)
)


# ══════════════════════════════════════════════════════════════════════════════
# 3b. Adjusted discharge SOH — account for unmeasured idle/parking SoC drops
#
#     For each discharge block, consecutive sessions may have a gap where the
#     vehicle was parked (BMS off). During this idle period the SoC drops
#     (parasitic loads, BMS recalibration) but no current is measured, so
#     block_capacity_ah is under-counted while block_soc_diff captures the full
#     drop. We add synthetic Ah = (idle_soc_gap% / 100) × NOMINAL_CAPACITY_AH
#     to bridge each inter-session gap.
#
#     ref_capacity_ah is hard-fixed at NOMINAL_CAPACITY_AH (436 Ah) — not the
#     data-driven p90 — because once idle Ah are restored the p90 should
#     converge to the true nominal anyway.
# ══════════════════════════════════════════════════════════════════════════════

print("\nAdjusting block_ah_B for unmeasured idle SoC gaps ...")
cycles["capacity_soh_disc_B_adj"] = np.nan

_has_soc_bounds = ("soc_start" in cycles.columns and "soc_end" in cycles.columns)

if has_blocks and _has_soc_bounds:
    disc_adj = cycles[
        (cycles["session_type"] == "discharge") &
        (cycles["current_mean"] > 0) &
        cycles["block_id"].notna() &
        cycles["capacity_ah_disc_B"].notna() &
        cycles["soc_end"].notna() &
        cycles["soc_start"].notna()
    ].copy()

    # Order sessions within each block chronologically
    disc_adj = disc_adj.sort_values(["registration_number", "block_id", "start_time"])

    # Inter-session idle gap: how much SoC was lost between consecutive sessions
    # prev_soc_end − next_soc_start > 0 means SoC dropped during unmeasured idle
    disc_adj["_prev_soc_end"] = disc_adj.groupby(
        ["registration_number", "block_id"])["soc_end"].shift(1)
    disc_adj["_idle_soc_gap"] = (
        disc_adj["_prev_soc_end"] - disc_adj["soc_start"]
    ).clip(lower=0)   # negative = SoC went up (charging between sessions) — ignore
    disc_adj["_idle_ah"] = disc_adj["_idle_soc_gap"] / 100.0 * NOMINAL_CAPACITY_AH

    # Block-level aggregation
    block_adj = disc_adj.groupby(["registration_number", "block_id"]).agg(
        block_ah_B     = ("capacity_ah_disc_B", "sum"),
        idle_ah_total  = ("_idle_ah",           "sum"),
        block_soc_diff = ("block_soc_diff",     "first"),
    ).reset_index()

    block_adj["block_ah_B_adj"] = block_adj["block_ah_B"] + block_adj["idle_ah_total"]
    block_adj["dod_B"]          = block_adj["block_soc_diff"].abs()

    block_adj["norm_cap_B_adj"] = (
        block_adj["block_ah_B_adj"] / (block_adj["dod_B"] / 100.0)
    ).replace([np.inf, -np.inf], np.nan)

    quality_adj = (
        (block_adj["dod_B"]          >= MIN_SOC_RANGE_DISC) &
        (block_adj["block_ah_B_adj"] >  0) &
        block_adj["norm_cap_B_adj"].notna()
    )
    block_adj["capacity_soh_disc_B_adj"] = np.nan
    # Fixed ref at nominal — not the per-vehicle data-driven p90
    block_adj.loc[quality_adj, "capacity_soh_disc_B_adj"] = (
        (block_adj.loc[quality_adj, "norm_cap_B_adj"] / NOMINAL_CAPACITY_AH * 100)
        .clip(0, 100)
    )

    # Map block SOH back to every session row in the block
    blk_soh_adj_idx = block_adj.set_index(
        ["registration_number", "block_id"])["capacity_soh_disc_B_adj"]
    disc_sess_adj = cycles[
        (cycles["session_type"] == "discharge") &
        (cycles["current_mean"] > 0) &
        cycles["block_id"].notna() &
        (cycles["block_soc_diff"] < 0)
    ]
    keys_adj = list(zip(disc_sess_adj["registration_number"], disc_sess_adj["block_id"]))
    cycles.loc[disc_sess_adj.index, "capacity_soh_disc_B_adj"] = [
        blk_soh_adj_idx.get(k, np.nan) for k in keys_adj
    ]

    n_gaps = (disc_adj["_idle_ah"] > 0).sum()
    total_idle_ah = disc_adj["_idle_ah"].sum()
    print(f"  Inter-session gaps with idle SoC drop: {n_gaps:,}")
    print(f"  Total synthetic Ah added:  {total_idle_ah:,.1f} Ah")
    print(f"  Mean idle Ah per block:    {block_adj['idle_ah_total'].mean():.1f} Ah")
    print(f"  norm_cap_B_adj p90 (quality blocks): "
          f"{block_adj.loc[quality_adj, 'norm_cap_B_adj'].quantile(0.9):.1f} Ah")
    print(f"  ref fixed at {NOMINAL_CAPACITY_AH} Ah (not data-driven p90)")

    soh_adj_vals = cycles["capacity_soh_disc_B_adj"].dropna()
    print(f"  capacity_soh_disc_B_adj: n={len(soh_adj_vals):,}  "
          f"mean={soh_adj_vals.mean():.2f}%  "
          f"median={soh_adj_vals.median():.2f}%  "
          f"std={soh_adj_vals.std():.2f}%")
elif not _has_soc_bounds:
    print("  WARNING: soc_start/soc_end not in cycles.csv — skipping adjustment.")
else:
    print("  WARNING: block_id not available — skipping adjustment.")


# ══════════════════════════════════════════════════════════════════════════════
# 4. Summary statistics
# ══════════════════════════════════════════════════════════════════════════════

print("\n" + "═" * 68)
print("  capacity_soh comparison: Source A (BMS current) vs Source B (hves1)")
print("═" * 68)

for label, stype, a_col, b_col in [
    ("Discharge", "discharge", "capacity_soh",  "capacity_soh_disc_B"),
    ("Charging ", "charge",    "capacity_soh",  "capacity_soh_chg_B"),
]:
    src_mask = cycles["capacity_soh_source"] == stype
    a = cycles.loc[src_mask, a_col].dropna()
    b = cycles.loc[src_mask, b_col].dropna()
    both = cycles.loc[src_mask & cycles[a_col].notna() & cycles[b_col].notna()]
    delta = (both[b_col] - both[a_col]).dropna()

    print(f"\n  {label}")
    print(f"    Source A (BMS)   n={len(a):,}  mean={a.mean():.2f}%  "
          f"std={a.std():.2f}%  median={a.median():.2f}%")
    print(f"    Source B (hves1) n={len(b):,}  mean={b.mean():.2f}%  "
          f"std={b.std():.2f}%  median={b.median():.2f}%")
    if len(delta):
        print(f"    B − A            n={len(delta):,}  mean={delta.mean():+.2f}%  "
              f"MAE={delta.abs().mean():.2f}%  within ±5%: {(delta.abs()<=5).mean():.1%}")

ref_per_veh = cycles.groupby("registration_number")["ref_capacity_ah"].first().dropna()
true_nominal_v = 282_000 / NOMINAL_CAPACITY_AH
print(f"\n  ref_capacity_ah (data-driven p90, capped at {NOMINAL_CAPACITY_AH} Ah):")
print(f"    Fleet: mean={ref_per_veh.mean():.1f} Ah  min={ref_per_veh.min():.1f} Ah  "
      f"max={ref_per_veh.max():.1f} Ah")
print(f"    Nominal cap {NOMINAL_CAPACITY_AH} Ah x {true_nominal_v:.0f} V = 282.0 kWh")
print(f"    (config NOMINAL_VOLTAGE_V={NOMINAL_VOLTAGE_V} V is conservative; "
      f"true 180S nominal ~ {true_nominal_v:.0f} V)")

# Adjusted discharge summary
if cycles["capacity_soh_disc_B_adj"].notna().any():
    disc_src = cycles["capacity_soh_source"] == "discharge"
    a_d   = cycles.loc[disc_src, "capacity_soh"].dropna()
    b_d   = cycles.loc[disc_src, "capacity_soh_disc_B"].dropna()
    b_adj = cycles.loc[disc_src, "capacity_soh_disc_B_adj"].dropna()
    both_adj = cycles.loc[
        disc_src &
        cycles["capacity_soh"].notna() &
        cycles["capacity_soh_disc_B_adj"].notna()
    ]
    delta_adj = (both_adj["capacity_soh_disc_B_adj"] - both_adj["capacity_soh"]).dropna()

    print(f"\n  Discharge — effect of idle Ah correction (ref fixed at {NOMINAL_CAPACITY_AH} Ah):")
    print(f"    Source A (BMS, data-driven ref): "
          f"n={len(a_d):,}  mean={a_d.mean():.2f}%  median={a_d.median():.2f}%")
    print(f"    Source B (block, no adj):        "
          f"n={len(b_d):,}  mean={b_d.mean():.2f}%  median={b_d.median():.2f}%")
    print(f"    Source B (+ idle Ah, ref=436):   "
          f"n={len(b_adj):,}  mean={b_adj.mean():.2f}%  median={b_adj.median():.2f}%")
    if len(delta_adj):
        print(f"    B_adj - A:  n={len(delta_adj):,}  mean={delta_adj.mean():+.2f}%  "
              f"MAE={delta_adj.abs().mean():.2f}%  within +/-5%: {(delta_adj.abs()<=5).mean():.1%}")


# ══════════════════════════════════════════════════════════════════════════════
# 5. Plots
# ══════════════════════════════════════════════════════════════════════════════

os.makedirs(PLOTS_DIR, exist_ok=True)
BINS = 40


def _hist_compare(ax, a_vals, b_vals, title, label_a, label_b):
    lo = max(0,   min(a_vals.min() if len(a_vals) else 0,
                      b_vals.min() if len(b_vals) else 0) - 2)
    hi = min(110, max(a_vals.max() if len(a_vals) else 100,
                      b_vals.max() if len(b_vals) else 100) + 2)
    bins = np.linspace(lo, hi, BINS)
    ax.hist(a_vals, bins=bins, alpha=0.55, color="#1f77b4", label=label_a, density=True)
    ax.hist(b_vals, bins=bins, alpha=0.55, color="#ff7f0e", label=label_b, density=True)
    for vals, col in [(a_vals, "#1f77b4"), (b_vals, "#ff7f0e")]:
        if len(vals):
            ax.axvline(vals.mean(), color=col, lw=2.0, ls="--")
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.set_xlabel("capacity_soh (%)")
    ax.set_ylabel("Density")
    ax.legend(fontsize=9)


def _delta_hist(ax, delta, title):
    ax.hist(delta, bins=BINS, color="#2ca02c", alpha=0.75, density=True)
    ax.axvline(0, color="k", lw=1.2, ls=":")
    if len(delta):
        ax.axvline(delta.mean(), color="#d62728", lw=2.0, ls="--",
                   label=f"mean {delta.mean():+.2f}%")
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.set_xlabel("Δ capacity_soh  B − A (%)")
    ax.set_ylabel("Density")
    ax.legend(fontsize=9)


# ── Plot 1: Discharge ─────────────────────────────────────────────────────────
disc_src = cycles["capacity_soh_source"] == "discharge"
a_disc   = cycles.loc[disc_src, "capacity_soh"].dropna()
b_disc   = cycles.loc[disc_src, "capacity_soh_disc_B"].dropna()
both_disc = cycles.loc[disc_src & cycles["capacity_soh"].notna() &
                        cycles["capacity_soh_disc_B"].notna()]
delta_disc = (both_disc["capacity_soh_disc_B"] - both_disc["capacity_soh"]).dropna()

with plt.style.context(STYLE):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Discharge Sessions — capacity_soh: BMS vs hves1_current",
                 fontsize=13, fontweight="bold")
    _hist_compare(axes[0], a_disc, b_disc, "SOH Distribution",
                  f"Source A BMS   n={len(a_disc):,}",
                  f"Source B hves1 n={len(b_disc):,}")
    _delta_hist(axes[1], delta_disc, "Delta (B − A)")
    fig.tight_layout()
    out = os.path.join(PLOTS_DIR, "soh_comparison_discharge.png")
    fig.savefig(out, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"\n  Saved: {out}")


# ── Plot 2: Charging ──────────────────────────────────────────────────────────
chg_src   = cycles["capacity_soh_source"] == "charge"
a_chg     = cycles.loc[chg_src, "capacity_soh"].dropna()
b_chg     = cycles.loc[chg_src, "capacity_soh_chg_B"].dropna()
both_chg  = cycles.loc[chg_src & cycles["capacity_soh"].notna() &
                        cycles["capacity_soh_chg_B"].notna()]
delta_chg = (both_chg["capacity_soh_chg_B"] - both_chg["capacity_soh"]).dropna()

with plt.style.context(STYLE):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Charging Sessions — capacity_soh: BMS vs hves1_current",
                 fontsize=13, fontweight="bold")
    _hist_compare(axes[0], a_chg, b_chg, "SOH Distribution",
                  f"Source A BMS   n={len(a_chg):,}",
                  f"Source B hves1 n={len(b_chg):,}")
    _delta_hist(axes[1], delta_chg, "Delta (B − A)")
    fig.tight_layout()
    out = os.path.join(PLOTS_DIR, "soh_comparison_charging.png")
    fig.savefig(out, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out}")


# ── Plot 3: Scatter ───────────────────────────────────────────────────────────
with plt.style.context(STYLE):
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle("Session-level Scatter: SOH Source A vs Source B",
                 fontsize=13, fontweight="bold")

    for ax, a_col, b_col, title in [
        (axes[0], "capacity_soh", "capacity_soh_disc_B", "Discharge"),
        (axes[1], "capacity_soh", "capacity_soh_chg_B",  "Charging"),
    ]:
        xy = cycles[[a_col, b_col]].dropna()
        if xy.empty:
            ax.set_title(f"{title} — no overlap data")
            continue
        ax.scatter(xy[a_col], xy[b_col], s=12, alpha=0.30, color="#1f77b4")
        lims = [max(0, min(xy[a_col].min(), xy[b_col].min()) - 1),
                min(105, max(xy[a_col].max(), xy[b_col].max()) + 1)]
        ax.plot(lims, lims, "k--", lw=1.2, label="y = x (perfect agreement)")
        corr = xy[a_col].corr(xy[b_col])
        ax.set_title(f"{title}  n={len(xy):,}  r={corr:.3f}",
                     fontsize=11, fontweight="bold")
        ax.set_xlabel("capacity_soh — Source A (BMS)")
        ax.set_ylabel("capacity_soh — Source B (hves1)")
        ax.legend(fontsize=9)
        ax.set_xlim(lims); ax.set_ylim(lims)

    fig.tight_layout()
    out = os.path.join(PLOTS_DIR, "soh_comparison_scatter.png")
    fig.savefig(out, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out}")


# ── Plot 4: ref_capacity_ah distribution ──────────────────────────────────────
with plt.style.context(STYLE):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("ref_capacity_ah: Data-driven p90 per Vehicle vs Nominal",
                 fontsize=13, fontweight="bold")

    ref_per_veh = cycles.groupby("registration_number")["ref_capacity_ah"].first().dropna()
    axes[0].hist(ref_per_veh, bins=20, color="#9467bd", alpha=0.8, edgecolor="white")
    axes[0].axvline(NOMINAL_CAPACITY_AH, color="#d62728", lw=2.0, ls="--",
                    label=f"Hard cap = {NOMINAL_CAPACITY_AH} Ah")
    axes[0].axvline(ref_per_veh.mean(), color="#ff7f0e", lw=2.0, ls="--",
                    label=f"Fleet mean = {ref_per_veh.mean():.1f} Ah")
    axes[0].set_title("Per-vehicle ref_capacity_ah", fontsize=11, fontweight="bold")
    axes[0].set_xlabel("ref_capacity_ah (Ah)")
    axes[0].set_ylabel("Vehicle count")
    axes[0].legend(fontsize=9)

    # Implied pack energy vs voltage assumption
    v_arr = np.linspace(600, 680, 200)
    axes[1].plot(v_arr, NOMINAL_CAPACITY_AH * v_arr / 1000,
                 color="#d62728", lw=2, label=f"Cap at {NOMINAL_CAPACITY_AH} Ah")
    axes[1].plot(v_arr, ref_per_veh.mean() * v_arr / 1000,
                 color="#ff7f0e", lw=2, ls="--",
                 label=f"Fleet mean {ref_per_veh.mean():.1f} Ah")
    axes[1].axhline(282, color="#2ca02c", lw=1.5, ls=":",
                    label="Rated 282 kWh")
    axes[1].axvline(282_000 / NOMINAL_CAPACITY_AH, color="#9467bd", lw=1.5, ls=":",
                    label=f"True nominal ≈ {282_000/NOMINAL_CAPACITY_AH:.0f} V")
    axes[1].set_title("Implied pack energy vs voltage", fontsize=11, fontweight="bold")
    axes[1].set_xlabel("Pack voltage (V)")
    axes[1].set_ylabel("Pack energy (kWh)")
    axes[1].legend(fontsize=9)

    fig.tight_layout()
    out = os.path.join(PLOTS_DIR, "soh_ref_capacity_dist.png")
    fig.savefig(out, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out}")

# ── Plot 5: Source A — effect of fixing ref to 436 Ah ────────────────────────
if cycles["capacity_soh_disc_A436"].notna().any():
    disc_src = cycles["capacity_soh_source"] == "discharge"
    a_orig = cycles.loc[disc_src, "capacity_soh"].dropna()
    a_436  = cycles["capacity_soh_disc_A436"].dropna()

    with plt.style.context(STYLE):
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        fig.suptitle(
            "Source A Discharge SOH — p90 ref (~197 Ah) vs Fixed 436 Ah ref",
            fontsize=13, fontweight="bold"
        )
        bins_all = np.linspace(0, 105, BINS)
        axes[0].hist(a_orig, bins=bins_all, alpha=0.55, color="#1f77b4",
                     label=f"A orig (p90 ref ~197 Ah)  n={len(a_orig):,}", density=True)
        axes[0].hist(a_436,  bins=bins_all, alpha=0.55, color="#d62728",
                     label=f"A fixed (436 Ah ref)       n={len(a_436):,}", density=True)
        for vals, col in [(a_orig, "#1f77b4"), (a_436, "#d62728")]:
            if len(vals):
                axes[0].axvline(vals.mean(), color=col, lw=2.0, ls="--")
        axes[0].set_title("SOH Distribution", fontsize=11, fontweight="bold")
        axes[0].set_xlabel("capacity_soh (%)")
        axes[0].set_ylabel("Density")
        axes[0].legend(fontsize=9)

        delta_A = (cycles.loc[disc_src & cycles["capacity_soh_disc_A436"].notna(),
                               "capacity_soh_disc_A436"] -
                   cycles.loc[disc_src & cycles["capacity_soh_disc_A436"].notna(),
                               "capacity_soh"]).dropna()
        _delta_hist(axes[1], delta_A, "Delta  A_fixed436 − A_orig")
        print(f"\n  A fixed436 vs A orig: mean={delta_A.mean():+.2f}%  "
              f"MAE={delta_A.abs().mean():.2f}%")

        fig.tight_layout()
        out = os.path.join(PLOTS_DIR, "soh_A_ref_fix_effect.png")
        fig.savefig(out, dpi=DPI, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved: {out}")

# ── Plot 6: Three-way discharge comparison (A_fixed vs B vs B_adj) ───────────
if cycles["capacity_soh_disc_B_adj"].notna().any():
    disc_src = cycles["capacity_soh_source"] == "discharge"
    a_d   = cycles.loc[disc_src, "capacity_soh"].dropna()
    b_d   = cycles.loc[disc_src, "capacity_soh_disc_B"].dropna()
    b_adj = cycles.loc[disc_src, "capacity_soh_disc_B_adj"].dropna()
    both_adj = cycles.loc[
        disc_src &
        cycles["capacity_soh"].notna() &
        cycles["capacity_soh_disc_B_adj"].notna()
    ]
    delta_adj = (both_adj["capacity_soh_disc_B_adj"] - both_adj["capacity_soh"]).dropna()

    with plt.style.context(STYLE):
        fig, axes = plt.subplots(1, 3, figsize=(19, 5))
        fig.suptitle(
            "Discharge SOH — Effect of Idle Ah Correction  (ref fixed at 436 Ah)",
            fontsize=13, fontweight="bold"
        )

        # Panel 1: A vs B_adj distribution
        bins_all = np.linspace(0, 105, BINS)
        axes[0].hist(a_d,   bins=bins_all, alpha=0.55, color="#1f77b4",
                     label=f"A (BMS, p90-ref)  n={len(a_d):,}", density=True)
        axes[0].hist(b_adj, bins=bins_all, alpha=0.55, color="#2ca02c",
                     label=f"B adj (436 Ah ref) n={len(b_adj):,}", density=True)
        axes[0].hist(b_d,   bins=bins_all, alpha=0.35, color="#ff7f0e",
                     label=f"B orig (no adj)    n={len(b_d):,}", density=True)
        for vals, col in [(a_d, "#1f77b4"), (b_adj, "#2ca02c"), (b_d, "#ff7f0e")]:
            if len(vals):
                axes[0].axvline(vals.mean(), color=col, lw=2.0, ls="--")
        axes[0].set_title("SOH Distribution (3-way)", fontsize=11, fontweight="bold")
        axes[0].set_xlabel("capacity_soh (%)")
        axes[0].set_ylabel("Density")
        axes[0].legend(fontsize=8)

        # Panel 2: Delta B_adj - A
        _delta_hist(axes[1], delta_adj, "Delta B_adj - A  (idle Ah corrected)")

        # Panel 3: Scatter A vs B_adj
        xy = cycles[["capacity_soh", "capacity_soh_disc_B_adj"]].dropna()
        if not xy.empty:
            axes[2].scatter(xy["capacity_soh"], xy["capacity_soh_disc_B_adj"],
                            s=10, alpha=0.25, color="#2ca02c")
            lims = [max(0,   min(xy["capacity_soh"].min(),
                                 xy["capacity_soh_disc_B_adj"].min()) - 1),
                    min(105, max(xy["capacity_soh"].max(),
                                 xy["capacity_soh_disc_B_adj"].max()) + 1)]
            axes[2].plot(lims, lims, "k--", lw=1.2, label="y = x")
            corr = xy["capacity_soh"].corr(xy["capacity_soh_disc_B_adj"])
            axes[2].set_title(f"Scatter A vs B_adj  n={len(xy):,}  r={corr:.3f}",
                              fontsize=11, fontweight="bold")
            axes[2].set_xlabel("Source A (BMS, p90-ref)")
            axes[2].set_ylabel("Source B adj (idle Ah, ref=436 Ah)")
            axes[2].legend(fontsize=9)
            axes[2].set_xlim(lims); axes[2].set_ylim(lims)

        fig.tight_layout()
        out = os.path.join(PLOTS_DIR, "soh_discharge_idle_adj.png")
        fig.savefig(out, dpi=DPI, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved: {out}")

print("\nDone.")
