"""
fleet_power_stats.py
--------------------
Pulls real observed voltage, current, and power values from BMS data
across different driving situations. Run once, prints a table for the presentation.
"""

import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pandas as pd
import numpy as np
from config import DATA_DIR, ARTIFACTS_DIR

BMS_FILE    = os.path.join(DATA_DIR, "bms_full_ultratech_intangles_more_cols_full.csv")
CYCLES_FILE = os.path.join(ARTIFACTS_DIR, "cycles.csv")

# ── 1. Session-level stats from cycles.csv ────────────────────────────────────
print("Loading cycles.csv ...")
cyc = pd.read_csv(CYCLES_FILE, low_memory=False)
cyc.columns = cyc.columns.str.strip().str.lower().str.replace(" ", "_")

disc = cyc[cyc["session_type"] == "discharge"].copy()
disc = disc[disc["current_mean"] > 0]   # true discharge only

print(f"\nDischarge sessions: {len(disc):,}  across {disc['registration_number'].nunique()} vehicles\n")

# ── 2. Sample raw BMS rows in chunks — collect voltage, current, derive power ─
print("Sampling raw BMS data (chunked) ...")
chunks = []
CHUNK  = 200_000
MAX_ROWS = 2_000_000   # sample up to 2M rows

for chunk in pd.read_csv(BMS_FILE, low_memory=False, chunksize=CHUNK):
    chunk.columns = chunk.columns.str.strip().str.lower().str.replace(" ", "_")
    chunks.append(chunk)
    if sum(len(c) for c in chunks) >= MAX_ROWS:
        break

raw = pd.concat(chunks, ignore_index=True)
print(f"Sampled {len(raw):,} raw BMS rows")
print("Columns:", [c for c in raw.columns if any(k in c for k in ["volt","curr","soc","temp","speed","power"])])

# Find voltage and current columns
volt_col = next((c for c in raw.columns if "voltage" in c and "cell" not in c and "limit" not in c), None)
curr_col = next((c for c in raw.columns if "current" in c and "limit" not in c and "charge" not in c), None)
print(f"\nUsing: voltage='{volt_col}'  current='{curr_col}'")

raw[volt_col] = pd.to_numeric(raw[volt_col], errors="coerce")
raw[curr_col] = pd.to_numeric(raw[curr_col], errors="coerce")

# Filter to valid discharge rows
raw_disc = raw[
    (raw[volt_col].between(400, 760)) &
    (raw[curr_col] > 20)
].copy()
raw_disc["power_kw"] = raw_disc[volt_col] * raw_disc[curr_col] / 1000

print(f"Valid discharge rows: {len(raw_disc):,}\n")

# ── 3. Define situations by current percentile ────────────────────────────────
p25  = raw_disc[curr_col].quantile(0.25)
p50  = raw_disc[curr_col].quantile(0.50)
p75  = raw_disc[curr_col].quantile(0.75)
p90  = raw_disc[curr_col].quantile(0.90)
p99  = raw_disc[curr_col].quantile(0.99)

situations = {
    "Idle / very light"  : raw_disc[raw_disc[curr_col].between(20,  p25)],
    "Gentle city driving": raw_disc[raw_disc[curr_col].between(p25, p50)],
    "Normal driving"     : raw_disc[raw_disc[curr_col].between(p50, p75)],
    "Hard driving (p75–p90)" : raw_disc[raw_disc[curr_col].between(p75, p90)],
    "Peak / acceleration (p90+)": raw_disc[raw_disc[curr_col] > p90],
}

print(f"{'Situation':<30} {'Current (A)':>14} {'Voltage (V)':>14} {'Power (kW)':>12} {'Rows':>8}")
print("-" * 82)
for name, grp in situations.items():
    if len(grp) == 0:
        continue
    c_med = grp[curr_col].median()
    c_lo  = grp[curr_col].quantile(0.10)
    c_hi  = grp[curr_col].quantile(0.90)
    v_med = grp[volt_col].median()
    p_med = grp["power_kw"].median()
    p_lo  = grp["power_kw"].quantile(0.10)
    p_hi  = grp["power_kw"].quantile(0.90)
    print(f"{name:<30} {c_lo:>5.0f}-{c_hi:<5.0f}A  {v_med:>7.0f}V  {p_lo:>5.0f}-{p_hi:<5.0f}kW  {len(grp):>8,}")

# ── 4. Overall fleet discharge stats ─────────────────────────────────────────
print("\n-- Overall fleet discharge (raw rows) --")
print(f"  Voltage : min={raw_disc[volt_col].min():.0f}V  "
      f"p10={raw_disc[volt_col].quantile(0.10):.0f}V  "
      f"median={raw_disc[volt_col].median():.0f}V  "
      f"p90={raw_disc[volt_col].quantile(0.90):.0f}V  "
      f"max={raw_disc[volt_col].max():.0f}V")
print(f"  Current : min={raw_disc[curr_col].min():.0f}A  "
      f"p10={raw_disc[curr_col].quantile(0.10):.0f}A  "
      f"median={raw_disc[curr_col].median():.0f}A  "
      f"p90={raw_disc[curr_col].quantile(0.90):.0f}A  "
      f"max={raw_disc[curr_col].max():.0f}A")
print(f"  Power   : min={raw_disc['power_kw'].min():.0f}kW  "
      f"p10={raw_disc['power_kw'].quantile(0.10):.0f}kW  "
      f"median={raw_disc['power_kw'].median():.0f}kW  "
      f"p90={raw_disc['power_kw'].quantile(0.90):.0f}kW  "
      f"max={raw_disc['power_kw'].max():.0f}kW")

# ── 5. Session-level from cycles.csv ─────────────────────────────────────────
if "voltage_mean" in disc.columns and "current_mean" in disc.columns:
    disc["power_mean_kw"] = disc["voltage_mean"] * disc["current_mean"] / 1000
    print("\n-- Session averages (from cycles.csv) --")
    print(f"  voltage_mean : {disc['voltage_mean'].mean():.1f}V  "
          f"(range {disc['voltage_mean'].min():.0f}-{disc['voltage_mean'].max():.0f}V)")
    print(f"  current_mean : {disc['current_mean'].mean():.1f}A  "
          f"(range {disc['current_mean'].min():.0f}-{disc['current_mean'].max():.0f}A)")
    if "current_max" in disc.columns:
        print(f"  current_max  : median={disc['current_max'].median():.0f}A  "
              f"p90={disc['current_max'].quantile(0.90):.0f}A  "
              f"max={disc['current_max'].max():.0f}A")
    print(f"  power_mean   : {disc['power_mean_kw'].mean():.1f}kW  "
          f"(range {disc['power_mean_kw'].min():.0f}–{disc['power_mean_kw'].max():.0f}kW)")
    if "energy_per_km" in disc.columns:
        epk = disc["energy_per_km"].dropna()
        # sanity: a bus should be 1-5 kWh/km — filter obvious outliers
        epk_sane = epk[epk.between(0.5, 10)]
        print(f"  energy/km    : median={epk.median():.3f} kWh/km  (all rows, inc. bad odometer)")
        print(f"  energy/km    : median={epk_sane.median():.3f} kWh/km  ({len(epk_sane)} sane rows 0.5-10 kWh/km)")
        if len(epk_sane) > 0:
            print(f"  implied range: {282/epk_sane.median():.0f} km at median consumption")
        # show odometer coverage
        if "odometer_km" in disc.columns:
            odo = disc["odometer_km"].dropna()
            print(f"  odometer_km  : median={odo.median():.1f} km  p10={odo.quantile(0.1):.1f}  p90={odo.quantile(0.9):.1f}  (per session)")
        if "soc_diff" in disc.columns:
            soc = disc["soc_diff"].dropna()
            print(f"  soc_diff     : median={soc.median():.1f}%  (negative = discharge)")
