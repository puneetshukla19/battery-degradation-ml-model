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
    # Regen dt cap only applies inside discharge sessions (short regen spikes)
    dt_hr_regen = dt_hr.clip(0, REGEN_DT_MAX_SEC / 3_600)

    curr_vals  = merged["hves1_current"].values
    is_chg_ses = (merged["session_type"] == "charging").values

    disc_mask  = curr_vals > 0

    # Discharge sessions:  regen = current < CHARGE_A (-50 A), capped dt to 5 s
    #   — keeps regen braking separate from noise/trickle in a moving session
    # Charging sessions:   count ALL negative current (< 0), full dt
    #   — trickle / slow phases (0 to -50 A) must be included for accuracy
    regen_mask   = (~is_chg_ses) & (curr_vals < CHARGE_A)
    plugin_mask  = is_chg_ses    & (curr_vals < 0)

    merged["_dq_disc"]  = np.where(disc_mask,   curr_vals * dt_hr.values,               0.0)
    merged["_dq_chg"]   = np.where(regen_mask,  np.abs(curr_vals * dt_hr_regen.values), 0.0)
    merged["_dq_plugin"] = np.where(plugin_mask, np.abs(curr_vals * dt_hr.values),       0.0)

    sess_grp = merged.groupby("session_id")
    sess_stats = pd.DataFrame({
        "capacity_ah_disc_B" : sess_grp["_dq_disc"].sum(),
        # regen Ah (discharge sessions) + plugin/trickle Ah (charging sessions)
        "capacity_ah_chg_B"  : sess_grp["_dq_chg"].sum() + sess_grp["_dq_plugin"].sum(),
        "voltage_mean_B"     : sess_grp["hves1_voltage_level"].mean(),
        "n_rows_B"           : sess_grp["timestamp"].count(),
    }).reset_index()
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

print("\nDeriving Source C — energy-derived capacity SOH ...")
# ══════════════════════════════════════════════════════════════════════════════
# Source C: capacity derived from energy_kwh / voltage_mean
#
#   capacity_ah_C = energy_kwh × 1000 / voltage_mean
#
# This bypasses both current sensors entirely. It uses the pack energy
# (already computed in data_prep_1.py from V × I integration) and the
# mean pack voltage to back-calculate Ah. Since energy is computed from
# the BMS voltage × current product, it is less sensitive to the raw
# current value — the voltage measurement is generally much more accurate
# than current sensors on EV packs.
#
# Discharge (block-level, same structure as Source A):
#   block_energy_kwh = sum of session energy_kwh across the block
#   block_cap_C      = block_energy_kwh × 1000 / voltage_mean_block
#   norm_cap_C       = block_cap_C / (block_soc_diff / 100)
#   capacity_soh_disc_C = (norm_cap_C / 436) × 100, clipped [0, 100]
#
# Charging (session-level):
#   cap_C_chg = energy_kwh × 1000 / voltage_mean × CHARGE_EFF
#   norm_cap_C_chg = cap_C_chg / (soc_range / 100)
#   capacity_soh_chg_C = (norm_cap_C_chg / 436) × 100, clipped [0, 100]
# ══════════════════════════════════════════════════════════════════════════════

TRUE_NOMINAL_V = 282_000.0 / NOMINAL_CAPACITY_AH   # ~647 V — pack energy / nominal cap

cycles["capacity_soh_disc_C"] = np.nan
cycles["capacity_soh_chg_C"]  = np.nan

has_energy  = "energy_kwh"    in cycles.columns
has_voltage = "voltage_mean"  in cycles.columns

if has_energy and has_voltage and "block_id" in cycles.columns:
    # ── Discharge block-level ─────────────────────────────────────────────────
    disc_C = cycles[
        (cycles["session_type"] == "discharge") &
        (cycles["current_mean"] > 0) &
        cycles["block_id"].notna() &
        (cycles["block_soc_diff"] < 0) &
        (cycles["energy_kwh"]   > 0) &
        (cycles["voltage_mean"] > 0)
    ].copy()

    # Ah per session from energy: energy_kwh × 1000 / V_mean
    disc_C["_cap_ah_C"] = disc_C["energy_kwh"] * 1000.0 / disc_C["voltage_mean"]

    blk_C = disc_C.groupby(["registration_number", "block_id"]).agg(
        block_cap_C    = ("_cap_ah_C",      "sum"),
        block_v_mean   = ("voltage_mean",   "mean"),
        block_soc_diff = ("block_soc_diff", "first"),
    ).reset_index()

    blk_C["dod_C"]      = blk_C["block_soc_diff"].abs()
    blk_C["norm_cap_C"] = (
        blk_C["block_cap_C"] / (blk_C["dod_C"] / 100.0)
    ).replace([np.inf, -np.inf], np.nan)

    quality_C = (
        (blk_C["dod_C"]      >= MIN_SOC_RANGE_DISC) &
        (blk_C["block_cap_C"] > 0) &
        blk_C["norm_cap_C"].notna()
    )
    blk_C["soh_disc_C"] = np.nan
    blk_C.loc[quality_C, "soh_disc_C"] = (
        (blk_C.loc[quality_C, "norm_cap_C"] / NOMINAL_CAPACITY_AH * 100).clip(0, 100)
    )

    blk_C_idx = blk_C.set_index(["registration_number", "block_id"])["soh_disc_C"]
    disc_rows_C = cycles[
        (cycles["session_type"] == "discharge") &
        (cycles["current_mean"] > 0) &
        cycles["block_id"].notna() &
        (cycles["block_soc_diff"] < 0)
    ]
    cycles.loc[disc_rows_C.index, "capacity_soh_disc_C"] = [
        blk_C_idx.get(k, np.nan)
        for k in zip(disc_rows_C["registration_number"], disc_rows_C["block_id"])
    ]

    c_disc_vals = cycles["capacity_soh_disc_C"].dropna()
    norm_c = blk_C.loc[quality_C, "norm_cap_C"]
    print(f"  Source C discharge: n={len(c_disc_vals):,}  "
          f"mean={c_disc_vals.mean():.2f}%  median={c_disc_vals.median():.2f}%  "
          f"std={c_disc_vals.std():.2f}%")
    print(f"  norm_cap_C  p50={norm_c.median():.1f}  p90={norm_c.quantile(0.9):.1f}  "
          f"mean={norm_c.mean():.1f}  > 436 Ah: {(norm_c > NOMINAL_CAPACITY_AH).mean()*100:.1f}%")

    # ── Charging session-level ────────────────────────────────────────────────
    chg_C = cycles[
        (cycles["session_type"] == "charging") &
        (cycles["soc_diff"] > 0) &
        (cycles["energy_kwh"]   != 0) &   # negative for charging — use abs below
        (cycles["voltage_mean"] > 0)
    ].copy()

    chg_C["soc_rng"]     = chg_C["soc_range"].abs()
    chg_C                = chg_C[chg_C["soc_rng"] >= MIN_SOC_RANGE_CHG]
    chg_C["_cap_ah_C"]   = chg_C["energy_kwh"].abs() * 1000.0 / chg_C["voltage_mean"]
    chg_C["norm_cap_C"]  = (
        chg_C["_cap_ah_C"] * CHARGE_EFF / (chg_C["soc_rng"] / 100.0)
    ).replace([np.inf, -np.inf], np.nan)

    quality_C_chg = chg_C["norm_cap_C"].notna() & (chg_C["_cap_ah_C"] > 0)
    cycles.loc[chg_C.index[quality_C_chg], "capacity_soh_chg_C"] = (
        (chg_C.loc[quality_C_chg, "norm_cap_C"] / NOMINAL_CAPACITY_AH * 100).clip(0, 100)
    )

    c_chg_vals    = cycles["capacity_soh_chg_C"].dropna()
    norm_c_chg    = chg_C.loc[quality_C_chg, "norm_cap_C"]
    print(f"  Source C charging:  n={len(c_chg_vals):,}  "
          f"mean={c_chg_vals.mean():.2f}%  median={c_chg_vals.median():.2f}%  "
          f"std={c_chg_vals.std():.2f}%")
    print(f"  norm_cap_C_chg p50={norm_c_chg.median():.1f}  p90={norm_c_chg.quantile(0.9):.1f}  "
          f"mean={norm_c_chg.mean():.1f}  > 436 Ah: {(norm_c_chg > NOMINAL_CAPACITY_AH).mean()*100:.1f}%")
else:
    print("  WARNING: energy_kwh or voltage_mean missing from cycles.csv — skipping Source C.")

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

# Source C summary
if cycles["capacity_soh_disc_C"].notna().any() or cycles["capacity_soh_chg_C"].notna().any():
    print(f"\n  Source C (energy / voltage — independent of current sensors):")
    for label, col in [("Discharge", "capacity_soh_disc_C"), ("Charging ", "capacity_soh_chg_C")]:
        v = cycles[col].dropna()
        if len(v):
            print(f"    {label}: n={len(v):,}  mean={v.mean():.2f}%  "
                  f"median={v.median():.2f}%  std={v.std():.2f}%")


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

# ── Plot 7: Four-way comparison: A_orig / A_436 / B_adj / C (energy) ─────────
if cycles["capacity_soh_disc_C"].notna().any():
    disc_src = cycles["capacity_soh_source"] == "discharge"
    chg_src  = cycles["capacity_soh_source"] == "charge"

    rows = [
        ("A orig (BMS, p90 ref ~197 Ah)",  cycles.loc[disc_src, "capacity_soh"].dropna(),            "#aec7e8", "--"),
        ("A fixed (BMS, 436 Ah ref)",       cycles["capacity_soh_disc_A436"].dropna(),                "#1f77b4", "-"),
        ("B hves1 + idle adj (436 Ah)",     cycles.loc[disc_src, "capacity_soh_disc_B_adj"].dropna(), "#ff7f0e", "-"),
        ("C energy/voltage (436 Ah)",       cycles["capacity_soh_disc_C"].dropna(),                   "#2ca02c", "-"),
    ]

    VAR_BINS_CMP = np.concatenate([
        np.arange(0,  50,  5.0),
        np.arange(50, 80,  2.0),
        np.arange(80, 100, 0.5),
        [100.0],
    ])
    widths_cmp = np.diff(VAR_BINS_CMP)

    with plt.style.context(STYLE):
        fig, axes = plt.subplots(1, 2, figsize=(17, 6))
        fig.suptitle(
            "Discharge SOH — All Sources Compared  (ref = 436 Ah)\n"
            "Variable bin width: 5 pp below 50%  |  2 pp 50-80%  |  0.5 pp 80-100%",
            fontsize=12, fontweight="bold",
        )
        ax = axes[0]
        for label, vals, color, ls in rows:
            if len(vals) == 0:
                continue
            counts, _ = np.histogram(vals.clip(0, 100), bins=VAR_BINS_CMP)
            density    = counts / (counts.sum() * widths_cmp)
            ax.step(VAR_BINS_CMP[:-1], density, where="post",
                    color=color, lw=2.0, ls=ls,
                    label=f"{label}  n={len(vals):,}  mean={vals.mean():.1f}%")
            ax.axvline(vals.mean(), color=color, lw=1.0, ls=":", alpha=0.7)
        ax.axvspan(80, 100, alpha=0.06, color="#999999", zorder=0)
        ax.axvline(80, color="#999999", lw=0.8, ls="--", alpha=0.5)
        ax.set_title("SOH Distribution by Source", fontsize=11, fontweight="bold")
        ax.set_xlabel("capacity_soh (%)")
        ax.set_ylabel("Probability density")
        ax.set_xlim(0, 101)
        ax.legend(fontsize=8, loc="upper left")

        ax2 = axes[1]
        labels_bar = [r[0].split("(")[0].strip() for r in rows if len(r[1]) > 0]
        means_bar  = [r[1].mean()                for r in rows if len(r[1]) > 0]
        pct80_bar  = [(r[1] >= 80).mean()*100    for r in rows if len(r[1]) > 0]
        colors_bar = [r[2]                       for r in rows if len(r[1]) > 0]
        x  = np.arange(len(labels_bar))
        w  = 0.38
        b1 = ax2.bar(x - w/2, means_bar, width=w, color=colors_bar, alpha=0.85,
                     label="Mean SOH (%)", edgecolor="white")
        b2 = ax2.bar(x + w/2, pct80_bar, width=w, color=colors_bar, alpha=0.45,
                     label="% sessions >= 80%", edgecolor="white", hatch="//")
        for bar, val in zip(b1, means_bar):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.8,
                     f"{val:.1f}%", ha="center", va="bottom", fontsize=8, fontweight="bold")
        for bar, val in zip(b2, pct80_bar):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.8,
                     f"{val:.1f}%", ha="center", va="bottom", fontsize=8)
        ax2.set_xticks(x)
        ax2.set_xticklabels(labels_bar, fontsize=8, rotation=10)
        ax2.set_ylabel("Value (%)")
        ax2.set_title("Mean SOH vs % sessions >= 80%", fontsize=11, fontweight="bold")
        ax2.legend(fontsize=9)
        ax2.set_ylim(0, 115)
        fig.tight_layout()
        out = os.path.join(PLOTS_DIR, "soh_all_sources_discharge.png")
        fig.savefig(out, dpi=DPI, bbox_inches="tight")
        plt.close(fig)
        print(f"\n  Saved: {out}")

    # Charging 4-way
    chg_quality_mask = (
        (cycles["session_type"] == "charging") &
        (cycles["soc_diff"] > 0) &
        (cycles["soc_range"].abs() >= MIN_SOC_RANGE_CHG) &
        (cycles["capacity_ah_charge_total"] > 0) &
        (cycles["voltage_mean"] > 0)
    )
    norm_chg_A436 = (
        (cycles.loc[chg_quality_mask, "capacity_ah_charge_total"] * CHARGE_EFF) /
        (cycles.loc[chg_quality_mask, "soc_range"].abs() / 100.0) /
        NOMINAL_CAPACITY_AH * 100
    ).clip(0, 100).dropna()

    chg_rows = [
        ("A BMS (p90 ref)",       cycles.loc[chg_src, "capacity_soh"].dropna(),        "#aec7e8", "--"),
        ("A BMS (436 Ah ref)",    norm_chg_A436,                                        "#1f77b4", "-"),
        ("B hves1 (436 Ah ref)",  cycles.loc[chg_src, "capacity_soh_chg_B"].dropna(),  "#ff7f0e", "-"),
        ("C energy/V (436 Ah)",   cycles["capacity_soh_chg_C"].dropna(),                "#2ca02c", "-"),
    ]

    with plt.style.context(STYLE):
        fig, axes = plt.subplots(1, 2, figsize=(17, 6))
        fig.suptitle(
            "Charging SOH — All Sources Compared  (ref = 436 Ah)\n"
            "Variable bin width: 5 pp below 50%  |  2 pp 50-80%  |  0.5 pp 80-100%",
            fontsize=12, fontweight="bold",
        )
        ax = axes[0]
        for label, vals, color, ls in chg_rows:
            if len(vals) == 0:
                continue
            counts, _ = np.histogram(vals.clip(0, 100), bins=VAR_BINS_CMP)
            density    = counts / (counts.sum() * widths_cmp)
            ax.step(VAR_BINS_CMP[:-1], density, where="post",
                    color=color, lw=2.0, ls=ls,
                    label=f"{label}  n={len(vals):,}  mean={vals.mean():.1f}%")
            ax.axvline(vals.mean(), color=color, lw=1.0, ls=":", alpha=0.7)
        ax.axvspan(80, 100, alpha=0.06, color="#999999", zorder=0)
        ax.axvline(80, color="#999999", lw=0.8, ls="--", alpha=0.5)
        ax.set_title("Charging SOH Distribution by Source", fontsize=11, fontweight="bold")
        ax.set_xlabel("capacity_soh (%)")
        ax.set_ylabel("Probability density")
        ax.set_xlim(0, 101)
        ax.legend(fontsize=8, loc="upper left")

        ax2 = axes[1]
        labels_bar = [r[0].split("(")[0].strip() for r in chg_rows if len(r[1]) > 0]
        means_bar  = [r[1].mean()                for r in chg_rows if len(r[1]) > 0]
        pct80_bar  = [(r[1] >= 80).mean()*100    for r in chg_rows if len(r[1]) > 0]
        colors_bar = [r[2]                       for r in chg_rows if len(r[1]) > 0]
        x  = np.arange(len(labels_bar))
        b1 = ax2.bar(x - w/2, means_bar, width=w, color=colors_bar, alpha=0.85,
                     label="Mean SOH (%)", edgecolor="white")
        b2 = ax2.bar(x + w/2, pct80_bar, width=w, color=colors_bar, alpha=0.45,
                     label="% sessions >= 80%", edgecolor="white", hatch="//")
        for bar, val in zip(b1, means_bar):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.8,
                     f"{val:.1f}%", ha="center", va="bottom", fontsize=8, fontweight="bold")
        for bar, val in zip(b2, pct80_bar):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.8,
                     f"{val:.1f}%", ha="center", va="bottom", fontsize=8)
        ax2.set_xticks(x)
        ax2.set_xticklabels(labels_bar, fontsize=8, rotation=10)
        ax2.set_ylabel("Value (%)")
        ax2.set_title("Mean SOH vs % sessions >= 80%", fontsize=11, fontweight="bold")
        ax2.legend(fontsize=9)
        ax2.set_ylim(0, 115)
        fig.tight_layout()
        out = os.path.join(PLOTS_DIR, "soh_all_sources_charging.png")
        fig.savefig(out, dpi=DPI, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved: {out}")

# ── Plot 8: Variable-bin SOH distribution — Source C (energy/voltage) ────────
# Coarse bins below 80 % so the dense 80-100 % region can be shown finely.
#   0 – 50 %   →  5 % wide bins   (broad, few sessions here)
#   50 – 80 %  →  2 % wide bins   (medium detail)
#   80 – 100 % →  0.5 % wide bins (fine — where most sessions cluster)
VAR_BINS = np.concatenate([
    np.arange(0,  50,  5.0),
    np.arange(50, 80,  2.0),
    np.arange(80, 100, 0.5),
    [100.0],
])

disc_src = cycles["capacity_soh_source"] == "discharge"
chg_src  = cycles["capacity_soh_source"] == "charge"

disc_vals = cycles["capacity_soh_disc_C"].dropna().values
chg_vals  = cycles["capacity_soh_chg_C"].dropna().values

with plt.style.context(STYLE):
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle(
        "SOH Distribution — Source C: energy_kwh / voltage_mean  (ref = 436 Ah)\n"
        "Variable bin width: 5 pp below 50%  |  2 pp from 50-80%  |  0.5 pp from 80-100%",
        fontsize=12, fontweight="bold",
    )

    for ax, vals, title, color, n_label in [
        (axes[0], disc_vals, "Discharge SOH  (block-level, energy / voltage)", "#2ca02c", "discharge"),
        (axes[1], chg_vals,  "Charging SOH   (session-level, energy / voltage)", "#9467bd", "charging"),
    ]:
        # Counts per bin (variable width) — use density so bar height is
        # probability density, making area ∝ fraction of sessions.
        counts, edges = np.histogram(vals, bins=VAR_BINS)
        widths = np.diff(edges)
        density = counts / (counts.sum() * widths)   # prob density

        # Colour-code bars: grey below 80 %, colour in 80-100 %
        bar_colors = ["#bbbbbb" if e < 80 else color for e in edges[:-1]]

        bars = ax.bar(edges[:-1], density, width=widths, align="edge",
                      color=bar_colors, edgecolor="white", linewidth=0.4)

        # Vertical lines for mean and median
        ax.axvline(np.mean(vals),   color="#333333", lw=1.8, ls="--",
                   label=f"Mean   {np.mean(vals):.1f} %")
        ax.axvline(np.median(vals), color="#555555", lw=1.8, ls=":",
                   label=f"Median {np.median(vals):.1f} %")

        # Shade the 80-100 % region
        ax.axvspan(80, 100, alpha=0.07, color=color, zorder=0)
        ax.axvline(80, color=color, lw=1.0, ls="--", alpha=0.5)

        # Annotations
        pct_above_80 = (vals >= 80).mean() * 100
        pct_above_90 = (vals >= 90).mean() * 100
        ax.text(0.97, 0.95, f"≥ 80 %: {pct_above_80:.1f} % of sessions\n"
                             f"≥ 90 %: {pct_above_90:.1f} % of sessions",
                transform=ax.transAxes, ha="right", va="top", fontsize=9,
                bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8))

        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.set_xlabel("capacity_soh  (%)", fontsize=10)
        ax.set_ylabel("Probability density", fontsize=10)
        ax.set_xlim(0, 101)
        ax.legend(fontsize=9)
        ax.tick_params(axis="x", labelsize=8)

        # Secondary x-axis annotation marking the 80 % boundary
        ax.annotate("← coarse bins  |  fine bins →",
                    xy=(80, 0), xycoords=("data", "axes fraction"),
                    xytext=(0, -28), textcoords="offset points",
                    ha="center", fontsize=7.5, color="#666666",
                    arrowprops=None)

    fig.tight_layout()
    out = os.path.join(PLOTS_DIR, "soh_energy_distribution.png")
    fig.savefig(out, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"\n  Saved: {out}")

print("\nDone.")
