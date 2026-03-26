import sys, os, warnings
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) if "__file__" in dir() else os.getcwd())
# Force UTF-8 output on Windows so special chars don't crash the console
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import numpy as np
import pandas as pd
from tqdm import tqdm
from config import (
    BMS_FILE, GPS_FILE, VCU_FILE, CYCLES_CSV, SEQ_NPY, SEQ_META,
    VOLTAGE_RANGE, CURRENT_RANGE, CELL_V_RANGE, TEMP_RANGE, SOH_MIN,
    DISCHARGE_A, CHARGE_A, MIN_SESSION_MIN, MIN_BMS_ROWS, MAX_DT_MIN, TRIP_GAP_MIN,
    NUM_BINS, SEQ_FEATURES, SCALAR_FEATURES,
    IR_THRESHOLD_MOHM, LOW_SOC_PCT, BATTERY_CAPACITY_KWH, NOMINAL_CAPACITY_AH, NOMINAL_VOLTAGE_V,
    REGEN_SPEED_KPH, GPS_GAP_MAX_SEC, ODO_GAP_MAX_SEC, EPK_MAX_KWH_KM,
)

# ── New / overriding constants ──────────────────────────────────────────────────
SMOOTH_WINDOW    = 1           # no smoothing; session continuity handled by _merge_discharge_gaps()

BMS_GAP_MAX_SEC  = 30          # BMS records ≈10 s apart; allow up to 30 s staleness

# Cell voltage health thresholds (runtime flags, separate from CELL_V_RANGE data filter)
CELL_V_WARN_LO   = 3.0         # V — below = undervoltage stress
CELL_V_WARN_HI   = 3.5         # V — above = overvoltage stress
CELL_SPREAD_WARN = 0.02        # V — spread > this = notable imbalance

# Depot detection
MIN_DEPOT_SPEED_KPH   = 5.0    # kph — below this = candidate depot/idle position
DEPOT_IDLE_MIN_DETECT = 30     # minutes gap before a stationary cluster counts as depot

# capacity_soh quality gates (same as data_prep.py)
MIN_SOC_RANGE_PCT = 10.0
MIN_REF_SESSIONS  = 5
BMS_RATE_PER_HR   = 360
MIN_DATA_DENSITY  = BMS_RATE_PER_HR * 0.20   # 72 rows/hr

# Route stop/start intensity threshold
HIGH_STOP_INTENSITY = 4.0      # stops/hr — above this = high frequency stop-start route

REGEN_DT_MAX_SEC = 5.0         # braking is momentary; cap regen dt window at 5 s

MERGE_GAP_SEC        = 300   # seconds — max idle/stop gap to bridge between discharge sessions
                              # 5 min covers coasting + traffic light stops within a trip
MIN_SOC_RANGE_DISC_PCT = 15.0  # discharge quality gate; raised from 5% — at 5% SoC drop,
                                # ±1% BMS quantization noise = ±20% SoH error; 15% keeps error ≤7%
                                # (charging keeps MIN_SOC_RANGE_PCT = 10% from config)

# ── Column lists ────────────────────────────────────────────────────────────────
GPS_LOAD_COLS = ["registration_number", "gps_time", "latitude", "longitude",
                 "altitude", "head", "speed"]

VCU_LOAD_COLS = ["registration_number", "gps_time", "vcu_odometer"]

BMS_LOAD_COLS = [
    "registration_number", "gps_time", "event_datetime", "vendor", "spv",
    "voltage", "current", "soc", "soh",
    "battery_operating_state",
    "status_heating_control", "status_cooling_control",
    "status_charge_relay_off", "status_charge_relay_on",
    "status_precharge_relay", "status_positive_relay", "status_negative_relay",
    "max_discharge_power_limit", "max_charge_power_limit",
    "max_discharge_current_limit", "max_charge_current_limit",
    "min_cell_voltage", "max_cell_voltage",
    "min_cell_voltage_number", "max_cell_voltage_number",
    "min_cell_voltage_subsystem_number", "max_cell_voltage_subsystem_number",
    "temperature_lowest", "temperature_highest",
    "temperature_lowest_probe_number", "temperature_highest_probe_number",
    "temperature_lowest_subsystem_number", "temperature_highest_subsystem_number",
    "insulation_resistance",
    "subsystem_voltage", "subsystem_number", "subsystem_total_number",
    "subsystem_current",
]
BMS_STR_COLS   = {"registration_number", "gps_time", "event_datetime", "vendor", "spv"}
BMS_FLOAT_COLS = [c for c in BMS_LOAD_COLS if c not in BMS_STR_COLS]


# ══════════════════════════════════════════════════════════════════════════════
# GEO HELPERS
# ══════════════════════════════════════════════════════════════════════════════

_R_EARTH = 6371.0  # km

def _haversine_km(lat1, lon1, lat2, lon2):
    """Vectorised haversine distance (km). All inputs may be scalars or arrays."""
    lat1, lon1, lat2, lon2 = (np.radians(np.asarray(x, dtype=float))
                               for x in (lat1, lon1, lat2, lon2))
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a    = np.sin(dlat / 2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2)**2
    return 2 * _R_EARTH * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


def _bearing_deg(lat1, lon1, lat2, lon2):
    """
    Compass bearing (0–360°) from (lat1,lon1) to (lat2,lon2).
    Vectorised; returns NaN where start == end.
    """
    lat1, lon1, lat2, lon2 = (np.radians(np.asarray(x, dtype=float))
                               for x in (lat1, lon1, lat2, lon2))
    dlon = lon2 - lon1
    x    = np.sin(dlon) * np.cos(lat2)
    y    = np.cos(lat1) * np.sin(lat2) - np.sin(lat1) * np.cos(lat2) * np.cos(dlon)
    bearing = (np.degrees(np.arctan2(x, y)) + 360) % 360
    same_point = (np.abs(lat2 - lat1) < 1e-9) & (np.abs(lon2 - lon1) < 1e-9)
    return np.where(same_point, np.nan, bearing)


def _angle_diff(a, b):
    """Absolute angular difference (0–180°) between two bearing arrays. NaN-safe."""
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    diff = np.where(np.isnan(a) | np.isnan(b), np.nan, np.abs(a - b) % 360)
    return np.where(np.isnan(diff), np.nan, np.where(diff > 180, 360 - diff, diff))


# ══════════════════════════════════════════════════════════════════════════════
# STEP 0 — LOAD RAW DATA
# ══════════════════════════════════════════════════════════════════════════════

def load_gps(path: str) -> dict:
    """Load GPS data. Returns {reg: DataFrame sorted by gps_time}."""
    print(f"Loading GPS from {path} ...")
    existing = set(pd.read_csv(path, nrows=0).columns)
    cols = [c for c in GPS_LOAD_COLS if c in existing]
    missing = set(GPS_LOAD_COLS) - existing
    if missing:
        print(f"  Note: GPS columns not found: {sorted(missing)}")

    dtypes = {"speed": "float32", "latitude": "float32", "longitude": "float32",
              "altitude": "float32", "head": "float32",
              "registration_number": "str"}

    df = pd.read_csv(path, usecols=cols, dtype=dtypes)
    df["gps_time"] = pd.to_numeric(df["gps_time"], errors="coerce")
    df = df.dropna(subset=["gps_time", "registration_number", "latitude", "longitude"])
    df["gps_time"] = df["gps_time"].astype("int64")

    for col in ("head", "altitude"):
        if col not in df.columns:
            df[col] = np.nan
    df["head"] = df["head"] % 360

    gps_by_veh = {reg: grp.sort_values("gps_time").reset_index(drop=True)
                  for reg, grp in df.groupby("registration_number", sort=False)}

    print(f"  GPS rows: {len(df):,}  |  Vehicles: {len(gps_by_veh)}")
    return gps_by_veh


def load_vcu(path: str) -> dict:
    """Load VCU odometer data. Returns {reg: DataFrame sorted by gps_time}."""
    print(f"Loading VCU (odometer) from {path} ...")
    existing = set(pd.read_csv(path, nrows=0).columns)
    cols = [c for c in VCU_LOAD_COLS if c in existing]

    df = pd.read_csv(path, usecols=cols, dtype={"vcu_odometer": "float64"}, low_memory=False)
    df["gps_time"] = pd.to_numeric(df["gps_time"], errors="coerce")
    df = df.dropna(subset=["gps_time", "registration_number", "vcu_odometer"])

    vcu_by_veh = {reg: grp.sort_values("gps_time").reset_index(drop=True)
                  for reg, grp in df.groupby("registration_number", sort=False)}

    print(f"  VCU rows: {len(df):,}  |  Vehicles: {len(vcu_by_veh)}")
    return vcu_by_veh


def load_bms(path: str) -> tuple:
    """Load BMS data, apply range filters. Returns (bms_by_veh, bms_val_cols, fleet_thr)."""
    print(f"Loading BMS from {path} ...")
    vlo, vhi = VOLTAGE_RANGE
    ilo, ihi = CURRENT_RANGE
    clo, chi = CELL_V_RANGE
    tlo, thi = TEMP_RANGE

    existing   = set(pd.read_csv(path, nrows=0).columns)
    load_cols  = [c for c in BMS_LOAD_COLS  if c in existing]
    float_cols = [c for c in BMS_FLOAT_COLS if c in existing]
    missing    = set(BMS_LOAD_COLS) - existing
    if missing:
        print(f"  Note: {len(missing)} BMS columns missing: {sorted(missing)}")

    df = pd.read_csv(path, usecols=load_cols,
                     dtype={c: "float32" for c in float_cols}, low_memory=False)
    df["gps_time"] = pd.to_numeric(df["gps_time"], errors="coerce")
    df = df.dropna(subset=["gps_time", "registration_number", "voltage"])

    mask = (df["voltage"].between(vlo, vhi) &
            df["current"].between(ilo, ihi) &
            df["soh"].ge(SOH_MIN))
    for col, lo, hi in [("min_cell_voltage", clo, chi),
                         ("max_cell_voltage", clo, chi),
                         ("temperature_lowest",  tlo, thi),
                         ("temperature_highest", tlo, thi)]:
        if col in df.columns:
            mask &= df[col].between(lo, hi)
    df = df[mask].reset_index(drop=True)

    print(f"  Clean rows: {len(df):,}  |  Vehicles: {df['registration_number'].nunique()}")

    disc_mask = df["current"] > DISCHARGE_A
    if disc_mask.any():
        all_v = df.loc[disc_mask, "voltage"].values
        all_c = df.loc[disc_mask, "current"].values
        fleet_thr = dict(
            vsag_mild     = float(np.quantile(all_v, 0.25)),
            high_curr_thr = float(np.quantile(all_c, 0.75)),
            ir_thr_ohm    = IR_THRESHOLD_MOHM / 1000,
        )
        print(f"  Fleet thresholds (from {len(all_v):,} discharge rows):")
        print(f"    Vsag threshold (p25 discharge): <{fleet_thr['vsag_mild']:.1f} V")
        print(f"    High current: >{fleet_thr['high_curr_thr']:.1f} A  |  IR threshold: "
              f"{fleet_thr['ir_thr_ohm']*1000:.0f} mOhm")
    else:
        fleet_thr = {}

    bms_val_cols = [c for c in df.columns if c not in ("registration_number", "gps_time")]
    bms_by_veh = {reg: grp.sort_values("gps_time").reset_index(drop=True)
                  for reg, grp in df.groupby("registration_number", sort=False)}

    print(f"  BMS rows loaded: {len(df):,}")
    return bms_by_veh, bms_val_cols, fleet_thr


# ══════════════════════════════════════════════════════════════════════════════
# STEP 1 — GPS ← VCU ASOF JOIN (odometer per GPS record)
# ══════════════════════════════════════════════════════════════════════════════

def join_vcu_onto_gps(gps: pd.DataFrame, vcu: pd.DataFrame) -> pd.DataFrame:
    """Per-vehicle asof join: nearest VCU record within ODO_GAP_MAX_SEC seconds."""
    print("Joining VCU odometer onto GPS (asof per vehicle) ...")
    tolerance_ms = ODO_GAP_MAX_SEC * 1000
    parts = []

    for reg, gps_v in gps.groupby("registration_number"):
        vcu_v = vcu[vcu["registration_number"] == reg].sort_values("gps_time")
        if vcu_v.empty:
            gps_v = gps_v.copy()
            gps_v["vcu_odometer"] = np.nan
            parts.append(gps_v)
            continue

        merged = pd.merge_asof(
            gps_v.sort_values("gps_time"),
            vcu_v[["gps_time", "vcu_odometer"]],
            on="gps_time",
            tolerance=tolerance_ms,
            direction="nearest",
        )
        parts.append(merged)

    result = (pd.concat(parts)
                .sort_values(["registration_number", "gps_time"])
                .reset_index(drop=True))

    match_rate = result["vcu_odometer"].notna().mean()
    print(f"  Output rows: {len(result):,}  |  VCU match rate: {match_rate:.1%}")

    # Fleet-level VCU coverage diagnostic
    odo_cov = result.groupby("registration_number")["vcu_odometer"].apply(
        lambda s: s.notna().mean())
    print(f"  Fleet VCU coverage: median={odo_cov.median():.1%}  "
          f">=90%: {(odo_cov >= 0.9).sum()}/{len(odo_cov)} vehicles")

    # ── Odometer backward-jump anomalies ──────────────────────────────────────
    issues = []
    for reg, grp in result.groupby("registration_number"):
        odo = grp["vcu_odometer"].dropna()
        n_back = (odo.diff() < -1.0).sum()
        if n_back:
            issues.append(f"    {reg}: {n_back} drops >1 km")
    if issues:
        print(f"  Odometer backward-jump anomalies ({len(issues)} vehicles):")
        for s in issues:
            print(s)

    return result


# ══════════════════════════════════════════════════════════════════════════════
# STEP 2 — INBOUND / OUTBOUND TRIP DIRECTION
# ══════════════════════════════════════════════════════════════════════════════

def detect_depot(gps_by_veh: dict) -> dict:
    """
    Find depot location per vehicle.

    Accepts gps_by_veh dict {reg: DataFrame sorted by gps_time} — avoids
    building a fleet-wide DataFrame just for depot detection.

    Strategy:
      1. Collect GPS points where speed < MIN_DEPOT_SPEED_KPH AND they follow
         a time gap > DEPOT_IDLE_MIN_DETECT minutes (end-of-trip parking).
      2. Spatial median of those points = depot.
      3. Fallback: median of all near-stationary points if step 1 yields < 3.

    Returns {reg: (depot_lat, depot_lon, depot_alt_or_nan)}
    """
    depots = {}
    gap_ms = DEPOT_IDLE_MIN_DETECT * 60 * 1000

    for reg, grp in gps_by_veh.items():
        # grp is already sorted by gps_time from load_gps
        dt  = grp["gps_time"].diff().fillna(0)

        # Stationary after a long idle gap → likely returning to depot
        at_stop    = grp["speed"].fillna(0) < MIN_DEPOT_SPEED_KPH
        after_gap  = at_stop & (dt > gap_ms)
        candidates = grp[after_gap]

        if len(candidates) < 3:
            candidates = grp[grp["speed"].fillna(0) < MIN_DEPOT_SPEED_KPH]
        if candidates.empty:
            candidates = grp.head(5)

        dlat = candidates["latitude"].median()
        dlon = candidates["longitude"].median()
        dalt = candidates["altitude"].median() if candidates["altitude"].notna().any() else np.nan
        depots[reg] = (dlat, dlon, dalt)

    print(f"  Detected depot for {len(depots)} vehicle(s)")
    return depots


def label_trip_direction(gps_vcu: pd.DataFrame, depots: dict) -> pd.DataFrame:
    """
    Add per-row columns:
      dist_from_depot_km    — haversine distance to vehicle's depot
      head_aligns_outbound  — True if vehicle heading points away from depot (±90°)
      trip_direction        — 'outbound' | 'inbound' | 'unknown'
      is_loaded             — True for inbound (trucks return loaded)
      altitude_trend        — 'ascending' | 'descending' | 'flat' (5-point rolling)

    Algorithm:
      For each trip segment (gap > TRIP_GAP_MIN):
        1. Compute distance from depot per GPS point.
        2. Find argmax (furthest point = turnaround).
        3. Points before turnaround → outbound; after → inbound.
        4. If no clear turnaround (max at boundary): use heading alignment.

    Note: Outbound = zero load (empty trucks departing depot).
          Inbound  = loaded  (trucks returning to depot with cargo).
    """
    df = gps_vcu.copy()
    df["dist_from_depot_km"]   = np.nan
    df["head_aligns_outbound"] = np.nan
    df["trip_direction"]       = "unknown"
    df["is_loaded"]            = False
    df["altitude_trend"]       = "flat"

    trip_gap_ms = TRIP_GAP_MIN * 60 * 1000

    for reg, grp_orig in df.groupby("registration_number"):
        if reg not in depots:
            continue
        depot_lat, depot_lon, depot_alt = depots[reg]
        grp = grp_orig.sort_values("gps_time")
        idx = grp.index

        # Distance from depot (vectorised)
        dist = _haversine_km(grp["latitude"].values, grp["longitude"].values,
                             depot_lat, depot_lon)
        df.loc[idx, "dist_from_depot_km"] = dist

        # Heading alignment
        if grp["head"].notna().any():
            brg = _bearing_deg(
                np.full(len(grp), depot_lat), np.full(len(grp), depot_lon),
                grp["latitude"].values, grp["longitude"].values,
            )
            adiff = _angle_diff(grp["head"].values, brg)
            # cast to float (1.0/0.0/NaN) to stay compatible with float64 column
            df.loc[idx, "head_aligns_outbound"] = np.where(
                np.isnan(adiff), np.nan, (adiff < 90.0).astype(float)
            )

        # Altitude trend (5-point rolling gradient)
        if grp["altitude"].notna().sum() > 5:
            alt_grad = grp["altitude"].rolling(5, min_periods=2).mean().diff()
            trend = pd.cut(alt_grad,
                           bins=[-np.inf, -0.3, 0.3, np.inf],
                           labels=["descending", "flat", "ascending"])
            df.loc[idx, "altitude_trend"] = trend.astype(str).values

        # Segment loop
        dt_ms  = grp["gps_time"].diff().fillna(0).values
        seg_id = np.cumsum(dt_ms > trip_gap_ms)

        for seg in np.unique(seg_id):
            seg_mask    = seg_id == seg
            seg_indices = idx[seg_mask]
            seg_dist    = dist[seg_mask]
            n           = len(seg_dist)

            if n < 3:
                continue

            max_pos = int(np.argmax(seg_dist))

            if max_pos == 0 or max_pos == n - 1:
                # No clear turnaround — fall back to heading
                seg_align = df.loc[seg_indices, "head_aligns_outbound"]
                if seg_align.notna().any():
                    direction = "outbound" if seg_align.dropna().mean() > 0.5 else "inbound"
                    df.loc[seg_indices, "trip_direction"] = direction
            else:
                out_idx = seg_indices[: max_pos + 1]
                in_idx  = seg_indices[max_pos :]
                df.loc[out_idx, "trip_direction"] = "outbound"
                df.loc[in_idx,  "trip_direction"] = "inbound"

    df["is_loaded"] = df["trip_direction"] == "inbound"

    dir_counts = df["trip_direction"].value_counts().to_dict()
    print(f"  Trip direction breakdown: {dir_counts}")
    known = df["trip_direction"].isin(["outbound", "inbound"]).mean()
    print(f"  Direction coverage: {known:.1%} of GPS rows classified")

    return df


# ══════════════════════════════════════════════════════════════════════════════
# STEP 3 — GPS+VCU ← BMS ASOF JOIN
# ══════════════════════════════════════════════════════════════════════════════

def join_bms_onto_gps(gps_vcu: pd.DataFrame, bms: pd.DataFrame) -> pd.DataFrame:
    """Per-vehicle asof join: nearest BMS record within BMS_GAP_MAX_SEC seconds. Drops rows with no BMS voltage match."""
    print("Joining BMS telemetry onto GPS+VCU (asof per vehicle) ...")
    tolerance_ms = BMS_GAP_MAX_SEC * 1000

    bms_data_cols = [c for c in bms.columns if c not in ("registration_number", "gps_time")]
    parts = []

    for reg, gps_v in tqdm(gps_vcu.groupby("registration_number"), desc="BMS join"):
        bms_v = bms[bms["registration_number"] == reg].sort_values("gps_time")
        if bms_v.empty:
            gps_v = gps_v.copy()
            for c in bms_data_cols:
                if c not in gps_v.columns:
                    gps_v[c] = np.nan
            parts.append(gps_v)
            continue

        bms_right_cols = ["gps_time"] + [c for c in bms_data_cols if c in bms_v.columns]
        merged = pd.merge_asof(
            gps_v.sort_values("gps_time"),
            bms_v[bms_right_cols],
            on="gps_time",
            tolerance=tolerance_ms,
            direction="nearest",
            suffixes=("", "_bms"),
        )
        parts.append(merged)

    result = (pd.concat(parts)
                .sort_values(["registration_number", "gps_time"])
                .reset_index(drop=True))

    bms_match = result["voltage"].notna().mean() if "voltage" in result.columns else 0.0
    print(f"  Pre-filter rows: {len(result):,}  |  BMS match rate: {bms_match:.1%}")

    # Drop rows with no BMS voltage (no electrical data)
    n_before = len(result)
    result   = result[result["voltage"].notna()].reset_index(drop=True)
    print(f"  Dropped {n_before - len(result):,} GPS rows with no BMS voltage → {len(result):,} rows kept")

    return result


# ══════════════════════════════════════════════════════════════════════════════
# STEP 4 — DERIVED COLUMNS
# ══════════════════════════════════════════════════════════════════════════════

def add_derived_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add:
      cell_spread          = max_cell_voltage - min_cell_voltage
      temp_highest         = alias for temperature_highest
      cell_undervoltage    = min_cell_voltage < CELL_V_WARN_LO  (3.0 V)
      cell_overvoltage     = max_cell_voltage > CELL_V_WARN_HI  (3.5 V)
      cell_spread_warn     = cell_spread > CELL_SPREAD_WARN     (0.02 V)
    """
    df = df.copy()

    if "max_cell_voltage" in df.columns and "min_cell_voltage" in df.columns:
        df["cell_spread"]     = (df["max_cell_voltage"] - df["min_cell_voltage"]).astype("float32")
        df["cell_undervoltage"] = df["min_cell_voltage"] < CELL_V_WARN_LO
        df["cell_overvoltage"]  = df["max_cell_voltage"] > CELL_V_WARN_HI
        df["cell_spread_warn"]  = df["cell_spread"]      > CELL_SPREAD_WARN

    if "temperature_highest" in df.columns:
        df["temp_highest"] = df["temperature_highest"]

    return df


# ══════════════════════════════════════════════════════════════════════════════
# STEP 5 — ANOMALY REPORT
# ══════════════════════════════════════════════════════════════════════════════

def report_anomalies(df: pd.DataFrame, label: str = "") -> None:
    """
    Structured anomaly report printed to stdout. Checks:
      1. BMS voltage coverage per vehicle
      2. VCU odometer coverage + backward jumps
      3. GPS teleportation (implied speed > 300 kph)
      4. Impossible GPS speed (> 120 kph for trucks)
      5. Cell voltage outside normal range (3.0–3.5 V)
      6. Cell spread above warning threshold (> 0.02 V)
      7. Duplicate gps_time per vehicle
      8. SoC outside 0–100
      9. Voltage during charging vs discharging sanity check
     10. Timestamp ordering within vehicle
    """
    tag = f"  Anomaly Report{': ' + label if label else ''}"
    print(f"\n{'═' * 64}")
    print(tag)
    print(f"  Rows: {len(df):,}  |  Vehicles: {df['registration_number'].nunique()}")
    print(f"{'─' * 64}")

    # 1. BMS coverage
    if "voltage" in df.columns:
        cov = df.groupby("registration_number")["voltage"].apply(lambda s: s.notna().mean())
        low = cov[cov < 0.50]
        if not low.empty:
            print(f"\n  [1] BMS voltage coverage < 50%  ({len(low)} vehicles):")
            for r, v in low.items():
                print(f"      {r}: {v:.0%}")
        else:
            print(f"\n  [1] BMS voltage coverage: all vehicles ≥ 50% ✓")

    # 2. VCU odometer coverage + backward jumps
    if "vcu_odometer" in df.columns:
        odo_cov = df.groupby("registration_number")["vcu_odometer"].apply(
            lambda s: s.notna().mean())
        low_odo = odo_cov[odo_cov < 0.50]
        if not low_odo.empty:
            print(f"\n  [2] VCU odometer coverage < 50% ({len(low_odo)} vehicles):")
            for r, v in low_odo.items():
                print(f"      {r}: {v:.0%}")
        back_issues = []
        for reg, grp in df.groupby("registration_number"):
            odo = grp.sort_values("gps_time")["vcu_odometer"].dropna()
            n_back = (odo.diff() < -1.0).sum()
            if n_back:
                back_issues.append(f"      {reg}: {n_back} drops > 1 km")
        if back_issues:
            print(f"\n  [2b] Odometer backward drops ({len(back_issues)} vehicles):")
            for s in back_issues:
                print(s)

    # 3. GPS teleportation
    tp_issues = []
    for reg, grp in df.groupby("registration_number"):
        grp  = grp.sort_values("gps_time")
        dt_h = grp["gps_time"].diff() / 3_600_000
        dk   = _haversine_km(
            grp["latitude"].shift().fillna(grp["latitude"].iloc[0]).values,
            grp["longitude"].shift().fillna(grp["longitude"].iloc[0]).values,
            grp["latitude"].values, grp["longitude"].values,
        )
        n_tp = (dk / dt_h.replace(0, np.nan) > 300).sum()
        if n_tp:
            tp_issues.append(f"      {reg}: {n_tp} teleportation events (>300 kph implied)")
    if tp_issues:
        print(f"\n  [3] GPS teleportation anomalies ({len(tp_issues)} vehicles):")
        for s in tp_issues:
            print(s)
    else:
        print(f"\n  [3] GPS teleportation: none detected ✓")

    # 4. Speed anomalies
    if "speed" in df.columns:
        n_fast = (df["speed"] > 120).sum()
        if n_fast:
            print(f"\n  [4] GPS speed > 120 kph (trucks): {n_fast:,} rows")

    # 5 + 6. Cell voltage health
    if "cell_undervoltage" in df.columns:
        n_u = df["cell_undervoltage"].sum()
        print(f"\n  [5] Cell undervoltage (< {CELL_V_WARN_LO} V): {n_u:,} rows "
              f"({n_u / len(df):.1%})")
    if "cell_overvoltage" in df.columns:
        n_o = df["cell_overvoltage"].sum()
        print(f"  [5] Cell overvoltage  (> {CELL_V_WARN_HI} V): {n_o:,} rows "
              f"({n_o / len(df):.1%})")
    if "cell_spread_warn" in df.columns:
        n_s = df["cell_spread_warn"].sum()
        print(f"  [6] Cell spread > {CELL_SPREAD_WARN} V: {n_s:,} rows "
              f"({n_s / len(df):.1%})")

    # 7. Duplicate timestamps
    n_dup = df.groupby("registration_number")["gps_time"].apply(
        lambda s: s.duplicated().sum()).sum()
    if n_dup:
        print(f"\n  [7] Duplicate gps_time entries: {n_dup:,} (may inflate Coulomb counts)")
    else:
        print(f"\n  [7] Duplicate timestamps: none ✓")

    # 8. SoC range
    if "soc" in df.columns:
        n_soc = ((df["soc"] < 0) | (df["soc"] > 100)).sum()
        if n_soc:
            print(f"\n  [8] SoC outside 0–100: {n_soc:,} rows")

    # 9. Voltage sanity by session
    if "voltage" in df.columns and "current" in df.columns:
        vlo, vhi = VOLTAGE_RANGE
        chg_rows  = df["current"] < CHARGE_A
        disc_rows = df["current"] > DISCHARGE_A
        n_chg_low  = (chg_rows  & (df["voltage"] < vlo * 0.9)).sum()
        n_disc_hi  = (disc_rows & (df["voltage"] > vhi * 1.05)).sum()
        if n_chg_low or n_disc_hi:
            print(f"\n  [9] Voltage outside plausible range:")
            if n_chg_low:
                print(f"      Charging rows below {vlo * 0.9:.0f} V: {n_chg_low:,}")
            if n_disc_hi:
                print(f"      Discharge rows above {vhi * 1.05:.0f} V: {n_disc_hi:,}")

    # 10. Timestamp ordering
    n_ts_back = df.groupby("registration_number")["gps_time"].apply(
        lambda s: (s.diff() < 0).sum()).sum()
    if n_ts_back:
        print(f"\n  [10] Out-of-order timestamps: {n_ts_back:,} rows")
    else:
        print(f"\n  [10] Timestamp ordering: all in order ✓")

    print(f"{'═' * 64}\n")


# ══════════════════════════════════════════════════════════════════════════════
# STEP 6 — SESSION LABELLING (6 types + strict charging guard)
# ══════════════════════════════════════════════════════════════════════════════

def label_sessions(df: pd.DataFrame) -> pd.DataFrame:
    """
    Assign detailed_type (6-way) and session_type (3-way for session_id boundaries).

    Session types:
      moving_discharging   speed > REGEN_SPEED_KPH, current_sm > DISCHARGE_A
      moving_charging      speed > REGEN_SPEED_KPH, current_sm < CHARGE_A  (regen)
      charging             stationary or relay ON, current_sm < CHARGE_A   (plugin)
      discharging_stopped  speed ≤ REGEN_SPEED_KPH, current_sm > DISCHARGE_A
      stop                 speed ≤ REGEN_SPEED_KPH, current in dead zone
      idle                 GPS stale, current in dead zone

    Charging guard (two layers):
      Layer 1 — relay state: status_charge_relay_on == 1 → force to 'charging'
      Layer 2 — SoC monotonicity: discharge sessions where SoC rose ≥ 5% →
                reclassify as 'charging' and rebuild session_id boundaries.

    Regen handling:
      moving_charging → session_type = 'discharge' (keeps drive session continuous)
      capacity_ah_charge is credited separately in extract_cycles().
    """
    df = df.copy()

    # Smooth current per vehicle
    df["current_sm"] = (
        df.groupby("registration_number")["current"]
          .transform(lambda s: s.rolling(SMOOTH_WINDOW, min_periods=1, center=True).mean())
    )

    # Motion state (use GPS speed column produced by load_gps)
    if "speed" in df.columns:
        moving    = df["speed"].notna() & (df["speed"]  > REGEN_SPEED_KPH)
        stopped   = df["speed"].notna() & (df["speed"] <= REGEN_SPEED_KPH)
        gps_stale = df["speed"].isna()
    else:
        moving    = pd.Series(False, index=df.index)
        stopped   = pd.Series(False, index=df.index)
        gps_stale = pd.Series(True,  index=df.index)

    disc_rows = df["current_sm"] > DISCHARGE_A
    chg_rows  = df["current_sm"] < CHARGE_A
    dead_zone = ~disc_rows & ~chg_rows

    # Charging relay guard (layer 1)
    relay_chg = pd.Series(False, index=df.index)
    if "status_charge_relay_on" in df.columns:
        relay_chg = df["status_charge_relay_on"].fillna(0).astype(bool)

    # ── 6-type detailed_type ──────────────────────────────────────────────────
    df["detailed_type"] = "idle"
    df.loc[disc_rows & moving,                        "detailed_type"] = "moving_discharging"
    df.loc[disc_rows & stopped,                       "detailed_type"] = "discharging_stopped"
    df.loc[chg_rows   & moving & ~relay_chg,          "detailed_type"] = "moving_discharging"  # regen: same session, tracked via current sign
    df.loc[chg_rows  & (stopped | gps_stale),         "detailed_type"] = "charging"
    df.loc[relay_chg & chg_rows,                      "detailed_type"] = "charging"  # relay guard
    df.loc[dead_zone & stopped,                       "detailed_type"] = "stop"
    # Remaining: disc_rows but GPS stale → stays "idle" (motion unknown)

    # ── 3-type session_type (for session_id boundaries) ───────────────────────
    _MAP = {
        "moving_discharging":  "discharge",
        "charging":            "charging",
        "discharging_stopped": "discharge",
        "stop":                "idle",
        "idle":                "idle",
    }
    df["session_type"] = df["detailed_type"].map(_MAP)

    # Relay guard override
    if relay_chg.any():
        n_forced = (relay_chg & (df["session_type"] != "charging")).sum()
        if n_forced:
            print(f"  Charging relay guard: {n_forced:,} rows forced to 'charging' (relay ON)")
        df.loc[relay_chg, "session_type"] = "charging"
        df.loc[relay_chg, "detailed_type"] = "charging"

    # ── Session ID (run-length encoding) ──────────────────────────────────────
    df["session_id"] = (
        df.groupby("registration_number")["session_type"]
          .transform(lambda s: (s != s.shift()).cumsum())
    )

    # ── SoC monotonicity guard (layer 2) ─────────────────────────────────────
    if "soc" in df.columns:
        to_reclassify = []
        for (reg, sid), grp in df[df["session_type"] == "discharge"].groupby(
                ["registration_number", "session_id"]):
            soc = grp["soc"].dropna()
            if len(soc) < 5:
                continue
            # SoC trend > 5% in a labelled discharge session = mislabelled charging
            if soc.iloc[-1] - soc.iloc[0] > 5.0:
                to_reclassify.append((reg, sid))

        if to_reclassify:
            print(f"  SoC monotonicity guard: reclassifying {len(to_reclassify)} sessions "
                  f"(discharge→charging, SoC rose >5%)")
            for reg, sid in to_reclassify:
                mask = (df["registration_number"] == reg) & (df["session_id"] == sid)
                df.loc[mask, "session_type"]  = "charging"
                df.loc[mask, "detailed_type"] = "charging"
            # Rebuild session_id after corrections
            df["session_id"] = (
                df.groupby("registration_number")["session_type"]
                  .transform(lambda s: (s != s.shift()).cumsum())
            )

    df.drop(columns="current_sm", inplace=True)

    # Bridge short idle/stop gaps that fragment continuous drive sessions
    df = _merge_discharge_gaps(df)

    # Session type summary
    st = df["detailed_type"].value_counts().to_dict()
    print(f"  Session row counts: {st}")

    return df


def _merge_discharge_gaps(df: pd.DataFrame, max_gap_sec: float = MERGE_GAP_SEC) -> pd.DataFrame:
    """
    Bridge short idle/stop gaps between consecutive discharge sessions on the same vehicle.

    During coasting or light-load cruise, current falls into the dead zone (−50 A to +20 A),
    creating 'idle' rows that split a continuous drive into fragments. If the gap between two
    discharge sessions contains no charging rows and spans ≤ max_gap_sec, the gap rows are
    relabelled as discharge and session_id is rebuilt.

    Raw current values are never modified — capacity_ah and regen_ah are unaffected.
    """
    df = df.copy()

    # Build session-level boundary table
    sess = (
        df.groupby(["registration_number", "session_id"], sort=False)
          .agg(stype   = ("session_type", "first"),
               t_start = ("gps_time",     "min"),
               t_end   = ("gps_time",     "max"))
          .reset_index()
          .sort_values(["registration_number", "t_start"])
          .reset_index(drop=True)
    )

    gap_records = []   # list of (registration_number, session_id) tuples to relabel

    for reg, vsess in sess.groupby("registration_number", sort=False):
        vsess = vsess.reset_index(drop=True)
        n     = len(vsess)
        i     = 0
        while i < n:
            if vsess.loc[i, "stype"] != "discharge":
                i += 1
                continue
            # Scan forward past non-discharge, non-charging rows
            j = i + 1
            has_charging = False
            while j < n and vsess.loc[j, "stype"] != "discharge":
                if vsess.loc[j, "stype"] == "charging":
                    has_charging = True
                    break
                j += 1

            if not has_charging and j < n and vsess.loc[j, "stype"] == "discharge":
                gap_ms = vsess.loc[j, "t_start"] - vsess.loc[i, "t_end"]
                if 0 < gap_ms <= max_gap_sec * 1000:
                    for k in range(i + 1, j):
                        gap_records.append((reg, vsess.loc[k, "session_id"]))
            i = j

    if gap_records:
        merge_df = (pd.DataFrame(gap_records, columns=["registration_number", "session_id"])
                      .drop_duplicates()
                      .assign(_merge=True))
        df = df.merge(merge_df, on=["registration_number", "session_id"], how="left")
        mask = df["_merge"].fillna(False)
        df.loc[mask, "session_type"] = "discharge"
        df = df.drop(columns=["_merge"])
        print(f"  Gap merge: {mask.sum():,} rows across "
              f"{len(gap_records)} gap-session(s) bridged (max_gap={max_gap_sec}s)")

    # Rebuild session_id after merging
    df["session_id"] = (
        df.groupby("registration_number")["session_type"]
          .transform(lambda s: (s != s.shift()).cumsum())
    )
    return df


# ══════════════════════════════════════════════════════════════════════════════
# STEP 7 — VOLTAGE SAG (combined score + per-row flags)
# ══════════════════════════════════════════════════════════════════════════════

def compute_voltage_sag(df: pd.DataFrame, thresholds: dict = None) -> pd.DataFrame:
    """
    Fleet-consistent voltage sag detection using pre-computed thresholds.
    Adds per-row bool flag:
      _vsag  — True when voltage < vsag_mild (p25 discharge) while high-current
               AND previous row was above this vehicle's median discharge voltage.

    Using the per-vehicle median discharge voltage as the reference (not rest/idle
    voltage) means the guard fires throughout continuous discharge, not just on
    transitions from idle. With GPS at 30 s resolution trucks are always under load
    so rest_v would never be seen — the vehicle median is the correct operating-point
    reference.
    Session-level count n_vsag is computed in extract_cycles().
    """
    df  = df.copy()
    SK  = ["registration_number", "session_id"]

    if thresholds:
        vsag_mild     = thresholds["vsag_mild"]
        high_curr_thr = thresholds["high_curr_thr"]
    else:
        disc = df[df["session_type"] == "discharge"]
        if len(disc) < 100:
            df["_vsag"] = False
            return df
        vsag_mild     = float(disc["voltage"].quantile(0.25))
        high_curr_thr = float(disc["current"].quantile(0.75))

    # Per-vehicle median discharge voltage — the typical operating point.
    # A sag fires when voltage drops from above the vehicle's own median to below
    # vsag_mild in one step under high current. Fleet median used as fallback for
    # vehicles with too few discharge rows.
    disc_mask    = df["session_type"] == "discharge"
    fleet_med_v  = float(df.loc[disc_mask, "voltage"].median()) if disc_mask.any() else vsag_mild
    veh_med_map  = df[disc_mask].groupby("registration_number")["voltage"].median()
    _ref_v       = df["registration_number"].map(veh_med_map).fillna(fleet_med_v)

    print(f"  Vsag ref: per-vehicle median discharge V "
          f"(fleet median {fleet_med_v:.1f} V, sag threshold <{vsag_mild:.1f} V)")

    _hc            = df["current"] > high_curr_thr
    _prev_vol      = df.groupby(SK)["voltage"].shift(1)
    _from_above_med = (_prev_vol >= _ref_v) & _prev_vol.notna()

    df["_vsag"] = ((df["voltage"] < vsag_mild) & _hc & _from_above_med).fillna(False)

    return df


# ══════════════════════════════════════════════════════════════════════════════
# STEP 8 — INTERNAL RESISTANCE (per-row ir_ohm + high_ir flag)
# ══════════════════════════════════════════════════════════════════════════════

def compute_ir_metrics(df: pd.DataFrame, thresholds: dict = None) -> pd.DataFrame:
    """
    Per-row IR estimate: ir_ohm = |ΔV / ΔI| (valid when |ΔI| ≥ 2A).
    high_ir_flag = True when ir_ohm > IR_THRESHOLD_MOHM/1000 at high current.

    Session-level ir_ohm_mean, n_high_ir and rate-of-change (d_ir_per_cycle)
    are computed in extract_cycles().
    """
    df  = df.copy()
    SK  = ["registration_number", "session_id"]

    if thresholds and "high_curr_thr" in thresholds:
        high_curr_thr = thresholds["high_curr_thr"]
        ir_thr_ohm    = thresholds.get("ir_thr_ohm", IR_THRESHOLD_MOHM / 1000)
    else:
        disc = df[df["session_type"] == "discharge"]
        if len(disc) < 100:
            df["ir_ohm"]       = np.nan
            df["high_ir_flag"] = False
            return df
        high_curr_thr = float(disc["current"].quantile(0.75))
        ir_thr_ohm    = IR_THRESHOLD_MOHM / 1000

    _dv       = df.groupby(SK)["voltage"].diff()
    _di       = df.groupby(SK)["current"].diff()
    _valid_di = _di.abs() >= 2.0

    df["ir_ohm"]       = np.where(_valid_di, _dv.abs() / _di.abs(), np.nan)
    _hc                = df["current"] > high_curr_thr
    df["high_ir_flag"] = (
        (df["ir_ohm"] > ir_thr_ohm) & _hc & _valid_di
    ).fillna(False)

    return df


# ══════════════════════════════════════════════════════════════════════════════
# STEP 8b — ASSIGN SESSION ID TO RAW BMS ROWS (for high-resolution vsag/IR)
# ══════════════════════════════════════════════════════════════════════════════

def assign_session_to_bms(bms_v: pd.DataFrame, cycles_v: pd.DataFrame) -> pd.DataFrame:
    """
    Assign session_id and session_type to each raw BMS row using the session
    time boundaries from cycles_v (which were derived from GPS-resolution data).

    BMS rows that fall inside a known session window [start_time, end_time] are
    labelled; rows outside all session windows are dropped.

    Uses np.searchsorted for O(n log n) assignment — no per-row Python loop.
    """
    if bms_v.empty or cycles_v.empty:
        return bms_v.iloc[:0].copy()

    sessions = (cycles_v[["session_id", "session_type", "start_time", "end_time"]]
                .sort_values("start_time")
                .reset_index(drop=True))

    starts = sessions["start_time"].values
    ends   = sessions["end_time"].values
    sids   = sessions["session_id"].values
    stypes = sessions["session_type"].values

    t   = bms_v["gps_time"].values
    idx = np.searchsorted(starts, t, side="right") - 1

    # Row is valid only if it falls within the matched session's time window
    in_bounds = idx >= 0
    clipped   = np.clip(idx, 0, len(ends) - 1)
    valid     = in_bounds & (t <= ends[clipped])

    out = bms_v[valid].copy()
    out["session_id"]   = sids[clipped[valid]].astype(int)
    out["session_type"] = stypes[clipped[valid]]
    return out


# ══════════════════════════════════════════════════════════════════════════════
# STEP 9 — SESSION-LEVEL CYCLE FEATURE EXTRACTION
# ══════════════════════════════════════════════════════════════════════════════

def extract_cycles(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate row-level data into one row per session.

    Capacity buckets (three separate accumulators):
      capacity_ah_discharge — Coulombs where current > 0 (true discharge)
      capacity_ah_charge    — Coulombs where detailed_type == moving_charging (regen)
      capacity_ah_plugin    — Coulombs where detailed_type == charging (stationary)
      capacity_ah           — abs(total) Coulombs (legacy, used by add_capacity_soh)

    Rate-of-change columns (added after aggregation, per-vehicle session chain):
      d_vsag_per_cycle    — session-to-session delta in n_vsag
      d_n_high_ir         — session-to-session delta in n_high_ir count
      d_ir_ohm_per_cycle  — session-to-session delta in ir_ohm_mean
    """
    df  = df.copy()
    SK  = ["registration_number", "session_id", "session_type"]

    # Coulomb counting
    df["_dt_hr"]        = df.groupby(SK[:2])["gps_time"].diff().fillna(0) / 3_600_000
    df["_dt_hr"]        = df["_dt_hr"].clip(0, MAX_DT_MIN / 60)
    # Tighter dt for regen: braking is momentary, not sustained for the full GPS interval
    df["_dt_hr_regen"]  = df["_dt_hr"].clip(0, REGEN_DT_MAX_SEC / 3_600)
    df["_delta_q"]      = df["current"] * df["_dt_hr"]
    df["_delta_q_regen"]= df["current"] * df["_dt_hr_regen"]

    # Split Ah into 3 buckets
    _dt = df.get("detailed_type", pd.Series("", index=df.index))
    moving_mask     = _dt == "moving_discharging"
    plugin_chg_mask = _dt == "charging"
    disc_curr_mask  = df["current"] > 0
    # Regen: moving row with negative current (braking within drive session)
    moving_chg_mask = moving_mask & (df["current"] < CHARGE_A)

    df["_dq_discharge"] = np.where(disc_curr_mask,  df["_delta_q"],                    0.0)
    df["_dq_regen"]     = np.where(moving_chg_mask, df["_delta_q_regen"].abs(),        0.0)
    df["_dq_plugin"]    = np.where(plugin_chg_mask,  df["_delta_q"].abs(),             0.0)

    # Helper columns for per-type current means and row counts
    df["_curr_disc"]        = np.where(disc_curr_mask,  df["current"], np.nan)
    df["_curr_regen"]       = np.where(moving_chg_mask, df["current"], np.nan)
    df["_moving_disc_flag"] = (moving_mask & disc_curr_mask).astype(int)
    df["_moving_chg_flag"]  = moving_chg_mask.astype(int)

    # ── Aggregation spec ──────────────────────────────────────────────────────
    agg_spec = dict(
        n_rows           = ("gps_time",    "count"),
        start_time       = ("gps_time",    "min"),
        end_time         = ("gps_time",    "max"),
        soh              = ("soh",         "median"),
        soc_start        = ("soc",         "first"),
        soc_end          = ("soc",         "last"),
        voltage_mean     = ("voltage",     "mean"),
        voltage_min      = ("voltage",     "min"),
        current_mean           = ("current", lambda x: x[x > 0].mean() if (x > 0).sum() > (x < 0).sum() else x[x < 0].mean()),
        current_max            = ("current",     "max"),
        current_mean_discharge = ("_curr_disc",  "mean"),
        current_mean_charge    = ("_curr_regen", "mean"),
        n_low_soc        = ("soc",         lambda x: (x < LOW_SOC_PCT).sum()),
        # Capacity — three consistent buckets.
        # capacity_ah is derived POST-aggregation as (discharge - charge - plugin)
        # so all three terms use the same dt convention (5s cap for regen, full dt for discharge).
        # Using raw sum(_delta_q) here was wrong: regen rows used full GPS dt (~27s) in
        # _delta_q but only the 5s-capped dt in _dq_regen, producing a sign contradiction
        # when cap_ah_discharge > cap_ah_charge yet sum(_delta_q) was negative.
        capacity_ah_discharge = ("_dq_discharge", "sum"),
        capacity_ah_charge    = ("_dq_regen",     "sum"),
        capacity_ah_plugin    = ("_dq_plugin",    "sum"),
    )

    # Optional columns — add only if present
    def _maybe(key, col, func):
        if col in df.columns:
            agg_spec[key] = (col, func)

    _maybe("cell_spread_mean",    "cell_spread",           "mean")
    _maybe("cell_spread_max",     "cell_spread",           "max")
    _maybe("temp_start",          "temp_highest",          "first")
    _maybe("temp_max",            "temp_highest",          "max")
    _maybe("temp_mean",           "temp_highest",          "mean")
    _maybe("temp_lowest_mean",    "temperature_lowest",    "mean")
    _maybe("insulation_mean",     "insulation_resistance", "mean")
    _maybe("max_disc_lim",        "max_discharge_current_limit", "mean")
    _maybe("max_chg_lim",         "max_charge_current_limit",    "mean")
    _maybe("max_disc_pwr_lim",    "max_discharge_power_limit",   "mean")
    _maybe("max_chg_pwr_lim",     "max_charge_power_limit",      "mean")
    # Voltage sag (count of sag events — prev row at rest voltage → drops below vsag_mild)
    _maybe("n_vsag", "_vsag", "sum")
    # IR
    _maybe("ir_ohm_mean",   "ir_ohm",       "mean")
    _maybe("n_high_ir",     "high_ir_flag", "sum")
    # Cell health
    _maybe("n_cell_undervoltage", "cell_undervoltage", "sum")
    _maybe("n_cell_overvoltage",  "cell_overvoltage",  "sum")
    _maybe("n_cell_spread_warn",  "cell_spread_warn",  "sum")
    # Speed
    _maybe("speed_mean", "speed", "mean")
    _maybe("speed_max",  "speed", "max")
    # Location
    _maybe("lat_start", "latitude",  "first")
    _maybe("lon_start", "longitude", "first")
    # Odometer
    _maybe("odometer_start", "vcu_odometer", "first")
    _maybe("odometer_end",   "vcu_odometer", "last")
    # Direction / load
    _maybe("altitude_mean",  "altitude",  "mean")
    _maybe("altitude_range", "altitude",  lambda x: x.max() - x.min())

    if "detailed_type" in df.columns:
        agg_spec["detailed_type"] = (
            "detailed_type",
            lambda x: x.mode().iloc[0] if len(x) > 0 else "idle"
        )
    _maybe("n_discharge_rows", "_moving_disc_flag", "sum")
    _maybe("n_charge_rows",    "_moving_chg_flag",  "sum")
    if "trip_direction" in df.columns:
        agg_spec["load_direction"] = (
            "trip_direction",
            lambda x: x.mode().iloc[0] if len(x) > 0 else "unknown"
        )
    if "is_loaded" in df.columns:
        agg_spec["is_loaded"] = (
            "is_loaded",
            lambda x: int(x.mode().iloc[0]) if len(x) > 0 else 0
        )

    agg = df.groupby(SK).agg(**agg_spec).reset_index()

    # Signed net Ah: discharge bucket uses full GPS dt; regen bucket uses 5s cap.
    # Both are now applied consistently here so the sign matches the SoC direction.
    agg["capacity_ah"] = (
        agg["capacity_ah_discharge"] - agg["capacity_ah_charge"] - agg["capacity_ah_plugin"]
    )

    # ── Derived columns ───────────────────────────────────────────────────────
    agg["duration_hr"]   = (agg["end_time"] - agg["start_time"]) / 3_600_000
    agg["energy_kwh"]    = agg["capacity_ah"] * agg["voltage_mean"] / 1000

    # capacity_ah_charge_total: regen + plugin (useful for charging-side SOH)
    agg["capacity_ah_charge_total"] = agg["capacity_ah_charge"] + agg["capacity_ah_plugin"]

    disc_mask = agg["session_type"] == "discharge"
    chg_mask  = agg["session_type"] == "charging"

    agg["soc_diff"]  = agg["soc_end"] - agg["soc_start"]
    agg["soc_range"] = np.where(
        disc_mask, (agg["soc_start"] - agg["soc_end"]).abs(),
        np.where(chg_mask, (agg["soc_end"] - agg["soc_start"]).abs(), np.nan)
    )

    if "temp_start" in agg.columns and "temp_max" in agg.columns:
        agg["temp_rise_rate"] = (
            (agg["temp_max"] - agg["temp_start"]) /
            agg["duration_hr"].replace(0, np.nan)
        )

    if "odometer_start" in agg.columns:
        agg["odometer_km"] = (agg["odometer_end"] - agg["odometer_start"]).clip(lower=0)
        valid_km = disc_mask & (agg["odometer_km"] > 0.5)
        agg["energy_per_km"] = np.where(
            valid_km,
            (agg["soc_range"] / 100.0 / agg["odometer_km"]) * BATTERY_CAPACITY_KWH,
            np.nan
        )

    agg["charging_rate_kw"] = np.where(
        chg_mask & (agg["duration_hr"] > 0),
        agg["energy_kwh"] / agg["duration_hr"],
        np.nan
    )

    # IST datetime
    def to_ist(ms_col):
        return (pd.to_datetime(agg[ms_col], unit="ms", utc=True)
                  .dt.tz_convert("Asia/Kolkata")
                  .dt.tz_localize(None)
                  .dt.strftime("%Y-%m-%d %H:%M:%S"))

    agg.insert(agg.columns.get_loc("start_time") + 1, "start_time_ist", to_ist("start_time"))
    agg.insert(agg.columns.get_loc("end_time")   + 1, "end_time_ist",   to_ist("end_time"))

    # Drop temp row-level columns
    drop_raw = ["_dt_hr", "_dt_hr_regen", "_delta_q", "_delta_q_regen",
                "_dq_discharge", "_dq_regen", "_dq_plugin",
                "_curr_disc", "_curr_regen", "_moving_disc_flag", "_moving_chg_flag",
                "_vsag"]
    df.drop(columns=[c for c in drop_raw if c in df.columns], inplace=True)

    # Filter short / sparse sessions
    agg = agg[
        (agg["duration_hr"] >= MIN_SESSION_MIN / 60) &
        (agg["n_rows"]      >= MIN_BMS_ROWS)
    ].reset_index(drop=True)

    # Remove anomalous charging sessions where SoC dropped
    # Recompute chg_mask on the filtered agg (index reset above)
    anomalous_chg = (agg["session_type"] == "charging") & (agg["soc_diff"] < 0)
    if anomalous_chg.sum():
        print(f"  Removing {anomalous_chg.sum()} anomalous charging sessions (SoC dropped)")
    agg = agg[~anomalous_chg].reset_index(drop=True)

    agg = agg.sort_values(["registration_number", "start_time"]).reset_index(drop=True)

    # time_delta_hr: gap between end of previous session and start of this one
    agg["time_delta_hr"] = (
        agg["start_time"] - agg.groupby("registration_number")["end_time"].shift(1)
    ) / 3_600_000

    agg["cycle_number"] = agg.groupby("registration_number").cumcount() + 1

    # ── Rate-of-change columns (per vehicle, per session chain) ───────────────
    roc_pairs = [
        ("n_vsag",        "d_vsag_per_cycle"),
        ("n_high_ir",     "d_n_high_ir"),
        ("ir_ohm_mean",   "d_ir_ohm_per_cycle"),
    ]
    for src_col, roc_col in roc_pairs:
        if src_col not in agg.columns:
            continue
        # Diff only across discharge sessions (discharge cycle N vs discharge cycle N-1).
        # Mixing session types (diff over idle/charging rows too) produces meaningless
        # negatives when an idle session with 0 events follows a discharge with N events.
        disc_idx  = agg["session_type"] == "discharge"
        disc_sub  = agg[disc_idx].sort_values(["registration_number", "start_time"])
        roc_vals  = disc_sub.groupby("registration_number")[src_col].transform(lambda s: s.diff())
        agg[roc_col] = np.nan
        agg.loc[disc_idx, roc_col] = roc_vals.values

    # EPK sanity cap
    if "energy_per_km" in agg.columns:
        n_capped = (agg["energy_per_km"] > EPK_MAX_KWH_KM).sum()
        if n_capped:
            print(f"  EPK sanity cap: {n_capped} sessions > {EPK_MAX_KWH_KM} kWh/km → NaN")
        agg.loc[agg["energy_per_km"] > EPK_MAX_KWH_KM, "energy_per_km"] = np.nan

    # Block linkage — discharge/charge blocks bounded by charging/discharge events
    agg = compute_block_linkage(agg)

    # Block-level EPK (replaces session-level and trip-level)
    if "block_odometer_km" in agg.columns and "block_soc_diff" in agg.columns:
        disc_mask = agg["session_type"] == "discharge"
        valid_km  = disc_mask & (agg["block_odometer_km"] > 0.5)
        agg["energy_per_km"] = np.where(
            valid_km,
            (agg["block_soc_diff"].abs() / 100.0 / agg["block_odometer_km"]) * BATTERY_CAPACITY_KWH,
            np.nan,
        )
    if "energy_per_km" in agg.columns:
        agg.loc[agg["energy_per_km"] > EPK_MAX_KWH_KM, "energy_per_km"] = np.nan

    # Loaded vs unloaded EPK sanity check
    if "is_loaded" in agg.columns and "energy_per_km" in agg.columns:
        disc_mask = agg["session_type"] == "discharge"
        loaded_epk   = agg.loc[disc_mask &  agg["is_loaded"], "energy_per_km"].dropna()
        unloaded_epk = agg.loc[disc_mask & ~agg["is_loaded"], "energy_per_km"].dropna()
        if len(loaded_epk) > 5 and len(unloaded_epk) > 5:
            print(f"\n  EPK by load state (expect loaded > unloaded for trucks):")
            print(f"    Loaded   (inbound) : {loaded_epk.median():.3f} kWh/km  "
                  f"(n={len(loaded_epk)})")
            print(f"    Unloaded (outbound): {unloaded_epk.median():.3f} kWh/km  "
                  f"(n={len(unloaded_epk)})")
            if loaded_epk.median() < unloaded_epk.median() * 0.95:
                print(f"  ⚠ Loaded EPK < unloaded — check inbound/outbound classification "
                      f"or depot location")

    return agg


# ══════════════════════════════════════════════════════════════════════════════
# STEP 9b — TRIP LINKAGE + STOP/START INTENSITY
# ══════════════════════════════════════════════════════════════════════════════

def compute_block_linkage(agg: pd.DataFrame) -> pd.DataFrame:
    """
    Group sessions into charge-bounded blocks — no time-gap threshold needed.

    Discharge block: all discharge sessions between two charging events
                     (or start / end of the vehicle's history).
    Charge block:    all charging sessions between two discharge events.
    Idle / regen / stop sessions are absorbed into the preceding block via
    forward-fill, so their Ah and SoC are correctly attributed.

    Replaces compute_trip_linkage (time-gap approach). Using a charging event as
    the natural block boundary eliminates TRIP_GAP_MIN tuning and captures the
    full discharge Ah for each drive-to-charge cycle regardless of how many
    dead-zone fragments were created inside it.

    Adds per-session columns:
      block_id           — per-vehicle integer
      block_type         — "discharge" | "charging"
      block_soc_diff     — SoC swing across block (negative = discharge)
      block_capacity_ah  — sum discharge Ah (disc blocks) or charge Ah (chg blocks)
      block_n_sessions   — active session count in block
      block_odometer_km  — total km in block (discharge blocks, if GPS available)
    """
    agg = agg.copy()
    agg = agg.sort_values(["registration_number", "start_time"]).reset_index(drop=True)

    # ── Step 1: identify block boundaries from discharge + charging sessions ──
    active_mask = agg["session_type"].isin(["discharge", "charging"])
    active = agg.loc[active_mask, ["registration_number", "session_id",
                                   "session_type", "start_time",
                                   "soc_start", "soc_end",
                                   "capacity_ah_discharge"]].copy()
    if "capacity_ah_charge_total" in agg.columns:
        active["_chg_ah"] = agg.loc[active_mask, "capacity_ah_charge_total"].values
    else:
        active["_chg_ah"] = 0.0
    if "odometer_km" in agg.columns:
        active["_odo"] = agg.loc[active_mask, "odometer_km"].values
    else:
        active["_odo"] = np.nan

    active = active.sort_values(["registration_number", "start_time"]).reset_index(drop=True)

    # New block when vehicle changes or session type switches (D→C or C→D)
    active["_new_block"] = (
        (active["registration_number"] != active["registration_number"].shift()) |
        (active["session_type"]        != active["session_type"].shift())
    )
    active["_block_id"] = active.groupby("registration_number")["_new_block"].cumsum()

    # ── Step 2: aggregate block-level stats ───────────────────────────────────
    block_agg = (
        active.groupby(["registration_number", "_block_id"], sort=False)
        .agg(
            block_type        = ("session_type",          "first"),
            block_soc_start   = ("soc_start",             "first"),
            block_soc_end     = ("soc_end",               "last"),
            block_capacity_ah = ("capacity_ah_discharge", "sum"),
            _block_chg_ah     = ("_chg_ah",               "sum"),
            block_n_sessions  = ("session_id",            "count"),
            _block_odo        = ("_odo",                  "sum"),
        )
        .reset_index()
    )
    block_agg["block_soc_diff"] = block_agg["block_soc_end"] - block_agg["block_soc_start"]
    # Charge blocks: use charge Ah (discharge blocks already have discharge Ah)
    chg_blk = block_agg["block_type"] == "charging"
    block_agg.loc[chg_blk, "block_capacity_ah"] = block_agg.loc[chg_blk, "_block_chg_ah"]
    block_agg = block_agg.drop(columns=["_block_chg_ah", "block_soc_start", "block_soc_end"])
    if "odometer_km" in agg.columns:
        block_agg = block_agg.rename(columns={"_block_odo": "block_odometer_km"})
    else:
        block_agg = block_agg.drop(columns=["_block_odo"])

    n_disc = (block_agg["block_type"] == "discharge").sum()
    n_chg  = (block_agg["block_type"] == "charging").sum()
    print(f"  Block linkage: {n_disc} discharge blocks, {n_chg} charge blocks "
          f"across {block_agg['registration_number'].nunique()} vehicles")

    # ── Step 3: broadcast block_id to all sessions (forward-fill idle/regen) ──
    agg = agg.merge(
        active[["registration_number", "session_id", "_block_id"]]
              .rename(columns={"_block_id": "block_id"}),
        on=["registration_number", "session_id"], how="left"
    )
    agg["block_id"] = agg.groupby("registration_number")["block_id"].transform("ffill")

    # Join block-level stats
    keep_cols = ["registration_number", "_block_id", "block_type",
                 "block_soc_diff", "block_capacity_ah", "block_n_sessions"]
    if "block_odometer_km" in block_agg.columns:
        keep_cols.append("block_odometer_km")
    agg = agg.merge(
        block_agg[keep_cols].rename(columns={"_block_id": "block_id"}),
        on=["registration_number", "block_id"], how="left"
    )
    return agg


# ══════════════════════════════════════════════════════════════════════════════
# STEP 10 — FLEET FLAGS
# ══════════════════════════════════════════════════════════════════════════════

def add_fleet_flags(cycles: pd.DataFrame) -> pd.DataFrame:
    """
    Fleet-percentile flags:
      bms_coverage          — fraction of session time with BMS samples
      ecu_fault_suspected   — bms_coverage < 20%
      rapid_heating         — temp_rise_rate > p75 of non-zero events
      high_energy_per_km    — EPK > p75 (discharge sessions)
      slow_charging         — charging_rate_kw < p25 (charging sessions)
      fast_charging         — charging_rate_kw > p75 (charging sessions)
      cell_health_poor      — n_cell_undervoltage + n_cell_overvoltage > 0
                              OR n_cell_spread_warn / n_rows > 10%
    """
    cycles = cycles.copy()

    # BMS coverage
    cycles["bms_coverage"] = (
        (cycles["n_rows"] * (10.0 / 3600.0)) /
        cycles["duration_hr"].replace(0, np.nan)
    ).clip(0, 1.0)
    cycles["ecu_fault_suspected"] = cycles["bms_coverage"] < 0.20

    # Rapid heating
    if "temp_rise_rate" in cycles.columns:
        nz = cycles.loc[cycles["temp_rise_rate"] > 0, "temp_rise_rate"]
        p75 = float(nz.quantile(0.75)) if len(nz) else 0.0
        cycles["rapid_heating"] = cycles["temp_rise_rate"] > p75

    # High EPK
    if "energy_per_km" in cycles.columns:
        disc = cycles[cycles["session_type"] == "discharge"]
        epk_p75 = disc["energy_per_km"].quantile(0.75)
        cycles["high_energy_per_km"] = (
            (cycles["session_type"] == "discharge") &
            (cycles["energy_per_km"] > epk_p75)
        )

    # Charging rate
    if "charging_rate_kw" in cycles.columns:
        chg_kw = cycles[cycles["session_type"] == "charging"]["charging_rate_kw"].dropna()
        if len(chg_kw) > 5:
            p25, p75 = chg_kw.quantile(0.25), chg_kw.quantile(0.75)
            cycles["slow_charging"] = (
                (cycles["session_type"] == "charging") & (cycles["charging_rate_kw"] < p25))
            cycles["fast_charging"]  = (
                (cycles["session_type"] == "charging") & (cycles["charging_rate_kw"] > p75))

    # Cell health summary flag
    health_cols = [c for c in ("n_cell_undervoltage", "n_cell_overvoltage") if c in cycles.columns]
    if health_cols:
        cycles["cell_health_poor"] = cycles[health_cols].sum(axis=1) > 0
    if "n_cell_spread_warn" in cycles.columns:
        spread_frac = cycles["n_cell_spread_warn"] / cycles["n_rows"].replace(0, np.nan)
        if "cell_health_poor" in cycles.columns:
            cycles["cell_health_poor"] |= (spread_frac > 0.10)
        else:
            cycles["cell_health_poor"] = spread_frac > 0.10

    return cycles


# ══════════════════════════════════════════════════════════════════════════════
# STEP 11b — ENGINEERED FEATURES FOR ML DEGRADATION MODELS
# ══════════════════════════════════════════════════════════════════════════════

def add_engineered_features(cycles: pd.DataFrame) -> pd.DataFrame:
    """
    Adds engineered features required by the ML degradation models:

    1. EFC & aging clock  — cum_efc, days_since_first, aging_index
    2. Normalised rate features  — vsag_rate_per_hr, ir_event_rate
       (removes session-length confound from raw counts)
    3. Rolling 20-session OLS slopes per vehicle — *_trend_slope
       (actual degradation rate; replaces noisy single-step diffs)
    4. Rolling EWM (span=10) smoothed signals  — *_ewm10
    5. Operating stress features — c_rate_chg, dod_stress, thermal_stress,
       energy_per_loaded_session
    6. load_direction encoding  — load_direction_enc
    """
    from config import EFC_MAX, BATTERY_CAPACITY_KWH

    cycles = cycles.copy()
    cycles = cycles.sort_values(["registration_number", "start_time"])

    # ── 1. EFC and calendar aging ──────────────────────────────────────────
    # Use |soc_range|/100 as per-session EFC contribution (all session types)
    cycles["_efc_session"] = cycles["soc_range"].abs() / 100.0
    cycles["cum_efc"] = cycles.groupby("registration_number")["_efc_session"].transform(
        "cumsum"
    )

    # Days since first record per vehicle (start_time is Unix ms)
    cycles["days_since_first"] = (
        cycles.groupby("registration_number")["start_time"]
        .transform(lambda x: (x - x.min()) / 86_400_000.0)
    )

    # Aging index: EFC 70% + calendar 30%, clipped to [0, 1]
    cycles["aging_index"] = (
        0.7 * (cycles["cum_efc"] / EFC_MAX) +
        0.3 * (cycles["days_since_first"] / 3650.0)
    ).clip(0.0, 1.0)

    cycles.drop(columns=["_efc_session"], inplace=True)

    # ── 2. Normalised rate features ────────────────────────────────────────
    if "n_vsag" in cycles.columns and "duration_hr" in cycles.columns:
        cycles["vsag_rate_per_hr"] = (
            cycles["n_vsag"] / cycles["duration_hr"].clip(lower=0.1)
        )

    if "n_high_ir" in cycles.columns and "n_rows" in cycles.columns:
        cycles["ir_event_rate"] = (
            cycles["n_high_ir"] / cycles["n_rows"].clip(lower=1)
        )

    # ── 3. Rolling 20-session OLS slopes per vehicle ───────────────────────
    def _ols_slope(y: np.ndarray) -> float:
        """OLS slope for a window array, skipping NaN. Returns NaN if <5 valid."""
        mask = ~np.isnan(y)
        if mask.sum() < 5:
            return np.nan
        x = np.where(mask)[0].astype(float)
        return float(np.polyfit(x, y[mask], 1)[0])

    slope_specs = [
        ("vsag_rate_per_hr",  "vsag_trend_slope"),
        ("ir_event_rate",     "ir_event_trend_slope"),
        ("ir_ohm_mean",       "ir_ohm_trend_slope"),
        ("cell_spread_mean",  "spread_trend_slope"),
        ("capacity_soh",      "soh_trend_slope"),   # meaningful on charging sessions
    ]
    for src_col, label in slope_specs:
        if src_col not in cycles.columns:
            continue
        cycles[label] = cycles.groupby("registration_number")[src_col].transform(
            lambda x: x.rolling(20, min_periods=5).apply(_ols_slope, raw=True)
        )

    # ── 4. Rolling EWM (span=10) smoothed signals ─────────────────────────
    ewm_specs = [
        ("ir_ohm_mean",      "ir_ohm_mean_ewm10"),
        ("cell_spread_mean", "cell_spread_mean_ewm10"),
        ("temp_rise_rate",   "temp_rise_rate_ewm10"),
    ]
    if "vsag_rate_per_hr" in cycles.columns:
        ewm_specs.append(("vsag_rate_per_hr", "vsag_rate_per_hr_ewm10"))

    for src_col, label in ewm_specs:
        if src_col not in cycles.columns:
            continue
        cycles[label] = cycles.groupby("registration_number")[src_col].transform(
            lambda x: x.ewm(span=10, min_periods=3).mean()
        )

    # ── 5. Operating stress features ──────────────────────────────────────
    if "charging_rate_kw" in cycles.columns:
        cycles["c_rate_chg"] = cycles["charging_rate_kw"] / BATTERY_CAPACITY_KWH

    # DoD severity: nonlinear (LFP/NMC aging accelerates at high DoD)
    cycles["dod_stress"] = (cycles["soc_range"].abs() / 100.0) ** 1.5

    if "temp_max" in cycles.columns:
        cycles["thermal_stress"] = (cycles["temp_max"] - 45.0).clip(lower=0.0)

    if "energy_kwh" in cycles.columns and "is_loaded" in cycles.columns:
        cycles["energy_per_loaded_session"] = (
            cycles["energy_kwh"] / (cycles["is_loaded"].fillna(0) + 1)
        )

    # ── 6. load_direction encoding ─────────────────────────────────────────
    if "load_direction" in cycles.columns:
        direction_map = {"outbound": 0, "inbound": 1, "unknown": np.nan}
        cycles["load_direction_enc"] = cycles["load_direction"].map(direction_map)

    new_cols = [
        "cum_efc", "days_since_first", "aging_index",
        "vsag_rate_per_hr", "ir_event_rate",
        "ir_ohm_mean_ewm10", "cell_spread_mean_ewm10",
        "temp_rise_rate_ewm10", "vsag_rate_per_hr_ewm10",
        "vsag_trend_slope", "ir_event_trend_slope", "ir_ohm_trend_slope",
        "spread_trend_slope", "soh_trend_slope",
        "c_rate_chg", "dod_stress", "thermal_stress",
        "energy_per_loaded_session", "load_direction_enc",
    ]
    present = [c for c in new_cols if c in cycles.columns]
    print(f"\n  Engineered features added ({len(present)}): {', '.join(present)}")
    return cycles


# ══════════════════════════════════════════════════════════════════════════════
# STEP 11 — CAPACITY-BASED SoH PROXY (discharge + charging side)
# ══════════════════════════════════════════════════════════════════════════════

def add_capacity_soh(cycles: pd.DataFrame) -> pd.DataFrame:
    """
    Compute capacity_soh from both discharge and charging sessions for maximum
    SOH estimation coverage.

    Discharge side (primary):
      norm_cap = block_capacity_ah / (|block_soc_diff| / 100)
      Uses block-level Coulombs — one block per drive-to-charge cycle.

    Charging side (secondary, fills gaps):
      norm_cap_charge = capacity_ah_charge_total / (|soc_range| / 100)
      Uses session-level charge Ah (regen + plugin).
      Note: charging Ah ≥ actual capacity (coulombic inefficiency), so a
      correction factor of 0.97 is applied to align with discharge estimates.

    Combined:
      capacity_soh = (norm_cap from whichever side is available) /
                      ref_capacity_ah * 100
      capacity_soh_source: 'discharge' | 'charge' | NaN

    Quality gates (same as data_prep.py):
      |soc_diff| >= MIN_SOC_RANGE_PCT (10%)
      data_density >= MIN_DATA_DENSITY (72 rows/hr)
      current_mean > 0 (true discharge, not regen)
    """
    CHARGE_COULOMBIC_EFF = 0.97   # ~97% coulombic efficiency for LiNMC

    cycles = cycles.copy()
    n_total = cycles["registration_number"].nunique()

    # ── Discharge side — block-level ──────────────────────────────────────────
    # Each discharge block (all discharge sessions between two charging events)
    # contributes exactly ONE capacity estimate: block_capacity_ah / (block DoD).
    # block_capacity_ah and block_soc_diff are pre-aggregated by compute_block_linkage,
    # so we just deduplicate to one row per block.
    disc = cycles[
        (cycles["session_type"] == "discharge") &
        (cycles["current_mean"] > 0)
    ].copy()

    has_blocks = "block_id" in disc.columns and disc["block_id"].notna().any()

    if has_blocks:
        # One row per discharge block (block stats already aggregated)
        block_disc = (
            disc[disc["block_id"].notna()]
            .sort_values(["registration_number", "start_time"])
            .drop_duplicates(subset=["registration_number", "block_id"])
            [["registration_number", "block_id",
              "block_capacity_ah", "block_soc_diff"]]
            .copy()
        )
        # block_soc_diff negative for discharge; abs = depth-of-discharge
        block_disc["dod"] = block_disc["block_soc_diff"].abs()
        block_disc["norm_cap"] = (
            block_disc["block_capacity_ah"] / (block_disc["dod"] / 100.0)
        ).replace([np.inf, -np.inf], np.nan)

        quality_blocks = block_disc[
            (block_disc["dod"]              >= MIN_SOC_RANGE_DISC_PCT) &
            (block_disc["block_capacity_ah"] >  0) &
            block_disc["norm_cap"].notna()
        ].copy()

        n_q_disc = quality_blocks["registration_number"].nunique()
        print(f"  capacity_soh discharge: {len(quality_blocks):,} quality blocks "
              f"across {n_q_disc}/{n_total} vehicles")
        print(f"    Nominal: {NOMINAL_CAPACITY_AH} Ah  "
              f"({BATTERY_CAPACITY_KWH} kWh / {NOMINAL_VOLTAGE_V} V)")

        # Per-vehicle reference: p90 of block norm_cap, capped at nominal
        ref_disc = quality_blocks.groupby("registration_number")["norm_cap"].agg(
            lambda s: min(s.quantile(0.90), NOMINAL_CAPACITY_AH)
                      if len(s) >= MIN_REF_SESSIONS else np.nan
        )

        # Block-level capacity_soh
        block_disc = block_disc.join(ref_disc.rename("ref_cap"), on="registration_number")
        block_disc["ref_cap"] = block_disc["ref_cap"].fillna(NOMINAL_CAPACITY_AH)
        block_disc["capacity_soh_block"] = (
            (block_disc["norm_cap"] / block_disc["ref_cap"] * 100).clip(0, 100)
        )

        # Index for fast lookup: (registration_number, block_id) → capacity_soh
        block_soh_idx = block_disc.set_index(
            ["registration_number", "block_id"]
        )["capacity_soh_block"]

    else:
        # Fallback: no block_id — session-level (original behaviour)
        disc["dod"] = disc["soc_diff"].abs()
        disc["norm_cap"] = (
            disc["capacity_ah_discharge"] / (disc["dod"] / 100.0)
        ).replace([np.inf, -np.inf], np.nan)
        quality_disc = disc[
            (disc["dod"]          >= MIN_SOC_RANGE_DISC_PCT) &
            (disc["current_mean"] >  0) &
            disc["norm_cap"].notna()
        ]
        ref_disc = quality_disc.groupby("registration_number")["norm_cap"].agg(
            lambda s: min(s.quantile(0.90), NOMINAL_CAPACITY_AH)
                      if len(s) >= MIN_REF_SESSIONS else np.nan
        )
        print(f"  capacity_soh discharge (session-level fallback): "
              f"{len(quality_disc):,} sessions across "
              f"{quality_disc['registration_number'].nunique()}/{n_total} vehicles")

    # ── Charging side ─────────────────────────────────────────────────────────
    chg = cycles[cycles["session_type"] == "charging"].copy()
    if "capacity_ah_charge_total" in chg.columns:
        chg_soc = chg["soc_range"].abs()
        chg["norm_cap_charge"] = (
            (chg["capacity_ah_charge_total"] * CHARGE_COULOMBIC_EFF) /
            (chg_soc / 100.0)
        ).replace([np.inf, -np.inf], np.nan)

        chg["data_density"] = chg["n_rows"] / chg["duration_hr"].replace(0, np.nan)
        quality_mask_chg = (
            (chg_soc                  >= MIN_SOC_RANGE_PCT) &
            (chg["data_density"]      >= MIN_DATA_DENSITY)  &
            (chg["soc_diff"]          >  0)
        )
        quality_chg = chg[quality_mask_chg]
    else:
        quality_chg = pd.DataFrame()

    n_q_chg = quality_chg["registration_number"].nunique() if not quality_chg.empty else 0
    print(f"  capacity_soh charging : {len(quality_chg):,} sessions "
          f"across {n_q_chg}/{n_total} vehicles")

    # ── Per-vehicle reference: discharge preferred, charge fills gaps ─────────
    ref_chg = pd.Series(dtype=float)
    if not quality_chg.empty and "norm_cap_charge" in quality_chg.columns:
        ref_chg = quality_chg.groupby("registration_number")["norm_cap_charge"].agg(
            lambda s: min(s.quantile(0.90), NOMINAL_CAPACITY_AH)
                      if len(s) >= MIN_REF_SESSIONS else np.nan
        )

    ref_vehicle = ref_disc.combine_first(ref_chg)
    n_fallback = ref_vehicle.isna().sum()
    if n_fallback:
        print(f"    {n_fallback} vehicle(s) use nominal fallback ({NOMINAL_CAPACITY_AH} Ah)")
    ref_vehicle = ref_vehicle.fillna(NOMINAL_CAPACITY_AH)

    cycles = cycles.join(ref_vehicle.rename("ref_capacity_ah"), on="registration_number")
    cycles["ref_capacity_ah"] = cycles["ref_capacity_ah"].fillna(NOMINAL_CAPACITY_AH)

    # ── Assign capacity_soh ───────────────────────────────────────────────────
    cycles["capacity_soh"]        = np.nan
    cycles["capacity_soh_source"] = pd.array([pd.NA] * len(cycles), dtype="object")

    # Discharge: broadcast block-level SoH to every session in the block
    if has_blocks:
        disc_rows = cycles[
            (cycles["session_type"] == "discharge") &
            (cycles["current_mean"] > 0) &
            cycles["block_id"].notna() &
            (cycles["block_soc_diff"] < 0)
        ]
        keys = list(zip(disc_rows["registration_number"], disc_rows["block_id"]))
        soh_vals = [block_soh_idx.get(k, np.nan) for k in keys]
        cycles.loc[disc_rows.index, "capacity_soh"] = soh_vals
        cycles.loc[disc_rows.index, "capacity_soh_source"] = "discharge"
    else:
        # Session-level fallback
        dod_all  = cycles["soc_diff"].abs()
        norm_all = (
            cycles["capacity_ah_discharge"] / (dod_all / 100.0)
        ).replace([np.inf, -np.inf], np.nan)
        true_disc = (
            (cycles["session_type"] == "discharge") &
            (cycles["current_mean"] > 0) &
            (cycles["soc_diff"] < 0)
        )
        cycles.loc[true_disc, "capacity_soh"] = (
            (norm_all[true_disc] / cycles.loc[true_disc, "ref_capacity_ah"] * 100).clip(0, 100)
        )
        cycles.loc[true_disc, "capacity_soh_source"] = "discharge"

    # Charging: fill sessions where discharge SoH is unavailable
    if "capacity_ah_charge_total" in cycles.columns:
        chg_mask    = cycles["session_type"] == "charging"
        chg_soc_all = cycles["soc_range"].abs()
        norm_chg_all = (
            (cycles["capacity_ah_charge_total"] * CHARGE_COULOMBIC_EFF) /
            (chg_soc_all / 100.0)
        ).replace([np.inf, -np.inf], np.nan)
        true_chg = (
            chg_mask &
            (cycles["soc_diff"] > 0) &
            (chg_soc_all >= MIN_SOC_RANGE_PCT)
        )
        cycles.loc[true_chg, "capacity_soh"] = (
            (norm_chg_all[true_chg] / cycles.loc[true_chg, "ref_capacity_ah"] * 100).clip(0, 100)
        )
        cycles.loc[true_chg, "capacity_soh_source"] = "charge"

    return cycles


# ══════════════════════════════════════════════════════════════════════════════
# STEP 12 — DISCHARGE SEQUENCE EXTRACTION (unchanged from data_prep.py)
# ══════════════════════════════════════════════════════════════════════════════

def extract_sequences(df: pd.DataFrame, cycles: pd.DataFrame):
    valid = cycles[
        (cycles["session_type"]  == "discharge") &
        (cycles["current_mean"]  >  0) &
        (~cycles.get("ecu_fault_suspected", pd.Series(False, index=cycles.index)))
    ][["registration_number", "session_id"]].assign(_keep=True)

    disc_df = (
        df[df["session_type"] == "discharge"]
        .merge(valid, on=["registration_number", "session_id"], how="inner")
        .drop(columns="_keep")
    )

    sequences, seq_meta = [], []
    for (reg, sid), grp in tqdm(
        disc_df.groupby(["registration_number", "session_id"], sort=False),
        desc="Binning sequences",
    ):
        grp = grp.sort_values("gps_time").reset_index(drop=True)
        if len(grp) < NUM_BINS:
            continue

        idx_chunks = np.array_split(np.arange(len(grp)), NUM_BINS)
        rows = []
        for idx in idx_chunks:
            b   = grp.iloc[idx]
            row = [float(b[f].mean()) if f in b.columns else np.nan for f in SEQ_FEATURES]
            rows.append(row)

        arr = np.array(rows, dtype=np.float32)
        if np.isnan(arr).all(axis=0).any():
            continue

        col_means = np.nanmean(arr, axis=0)
        for j in range(arr.shape[1]):
            arr[np.isnan(arr[:, j]), j] = col_means[j]

        meta_row = cycles[
            (cycles["registration_number"] == reg) & (cycles["session_id"] == sid)
        ].iloc[0]

        sequences.append(arr)
        entry = {
            "seq_index":           len(sequences) - 1,
            "registration_number": reg,
            "session_id":          sid,
            "cycle_number":        meta_row["cycle_number"],
            "soh":                 meta_row["soh"],
            "capacity_soh":        meta_row.get("capacity_soh", np.nan),
            "capacity_soh_source": meta_row.get("capacity_soh_source", np.nan),
        }
        for f in SCALAR_FEATURES:
            entry[f] = meta_row.get(f, np.nan)
        seq_meta.append(entry)

    return sequences, seq_meta


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    bms_by_veh, bms_val_cols, fleet_thr = load_bms(BMS_FILE)
    gps_by_veh = load_gps(GPS_FILE)
    print("\nDetecting depot locations ...")
    depots = detect_depot(gps_by_veh)
    vcu_by_veh = load_vcu(VCU_FILE)

    vehicles = sorted(set(gps_by_veh.keys()) & set(bms_by_veh.keys()))

    print("\nProcessing vehicles ...")
    tol_odo_ms  = ODO_GAP_MAX_SEC * 1000
    tol_bms_ms  = BMS_GAP_MAX_SEC * 1000

    all_cycles     = []
    all_disc_rows  = []   # discharge rows (small column set) for sequence extraction
    anomaly_frames = []   # per-vehicle joined frames for the fleet anomaly report
    vehicle_bms_rates = []
    vehicle_vcu_rates = []

    for reg in tqdm(vehicles, desc="Vehicles"):
        gps_v = gps_by_veh.get(reg, pd.DataFrame())
        vcu_v = vcu_by_veh.get(reg, pd.DataFrame())
        bms_v = bms_by_veh.get(reg, pd.DataFrame())

        if gps_v.empty or bms_v.empty:
            continue

        # Step 1: GPS ← VCU
        if not vcu_v.empty:
            gps_v = pd.merge_asof(
                gps_v,
                vcu_v[["gps_time", "vcu_odometer"]],
                on="gps_time", tolerance=tol_odo_ms, direction="nearest",
            )
        else:
            gps_v["vcu_odometer"] = np.nan

        _vcu_rate_v = gps_v["vcu_odometer"].notna().mean()
        vehicle_vcu_rates.append(_vcu_rate_v)

        # Step 2: Trip direction (inline per-vehicle, avoids full-fleet label_trip_direction)
        if reg in depots:
            depot_lat, depot_lon, _ = depots[reg]
            dist = _haversine_km(gps_v["latitude"].values, gps_v["longitude"].values,
                                 depot_lat, depot_lon)
            gps_v["dist_from_depot_km"] = dist

            if gps_v["head"].notna().any():
                brg   = _bearing_deg(np.full(len(gps_v), depot_lat),
                                     np.full(len(gps_v), depot_lon),
                                     gps_v["latitude"].values, gps_v["longitude"].values)
                adiff = _angle_diff(gps_v["head"].values, brg)
                head_out = np.where(np.isnan(adiff), np.nan, (adiff < 90.0).astype(float))
            else:
                head_out = np.full(len(gps_v), np.nan)
            gps_v["head_aligns_outbound"] = head_out

            dt_ms  = gps_v["gps_time"].diff().fillna(0).values
            seg_id = np.cumsum(dt_ms > TRIP_GAP_MIN * 60 * 1000)
            trip_dir = np.full(len(gps_v), "unknown", dtype=object)

            for seg in np.unique(seg_id):
                seg_pos  = np.where(seg_id == seg)[0]
                seg_dist = dist[seg_pos]
                n        = len(seg_pos)
                if n < 3:
                    continue
                max_pos = int(np.argmax(seg_dist))
                if max_pos == 0 or max_pos == n - 1:
                    valid_h = head_out[seg_pos]
                    valid_h = valid_h[~np.isnan(valid_h)]
                    if len(valid_h):
                        trip_dir[seg_pos] = "outbound" if valid_h.mean() > 0.5 else "inbound"
                else:
                    trip_dir[seg_pos[: max_pos + 1]] = "outbound"
                    trip_dir[seg_pos[max_pos:]]      = "inbound"

            gps_v["trip_direction"] = trip_dir
            gps_v["is_loaded"]      = (trip_dir == "inbound").astype(int)
        else:
            gps_v["dist_from_depot_km"]   = np.nan
            gps_v["head_aligns_outbound"] = np.nan
            gps_v["trip_direction"]       = "unknown"
            gps_v["is_loaded"]            = 0

        # Step 3: GPS+VCU ← BMS
        bms_right = ["gps_time"] + [c for c in bms_val_cols if c in bms_v.columns]
        df_v = pd.merge_asof(
            gps_v,
            bms_v[bms_right],
            on="gps_time", tolerance=tol_bms_ms, direction="nearest",
        )
        _bms_rate_v = df_v["voltage"].notna().mean() if "voltage" in df_v.columns else 0.0
        vehicle_bms_rates.append(_bms_rate_v)
        df_v = df_v[df_v["voltage"].notna()].reset_index(drop=True)
        if len(df_v) < 10:
            continue

        # Steps 4-6: derived cols, sessions, sag, IR
        df_v = add_derived_columns(df_v)
        df_v = label_sessions(df_v)
        df_v = compute_voltage_sag(df_v, fleet_thr)
        df_v = compute_ir_metrics(df_v, fleet_thr)

        # Step 7: session-level aggregation
        cycles_v = extract_cycles(df_v)
        if cycles_v.empty:
            continue

        # Fix A: Re-run vsag + IR on raw BMS rows at 10s resolution.
        # GPS-resolution counts miss sub-30s sag events; raw BMS rows catch them.
        if not bms_v.empty and len(cycles_v) > 0:
            bms_sess = assign_session_to_bms(bms_v, cycles_v)
            if len(bms_sess) > 10:
                bms_sess = compute_voltage_sag(bms_sess, fleet_thr)
                bms_sess = compute_ir_metrics(bms_sess, fleet_thr)

                bms_vsag = bms_sess.groupby("session_id").agg(
                    n_vsag      = ("_vsag",        "sum"),
                    n_high_ir   = ("high_ir_flag", "sum"),
                    ir_ohm_mean = ("ir_ohm",       "mean"),
                ).reset_index()
                bms_vsag["session_id"] = bms_vsag["session_id"].astype(
                    cycles_v["session_id"].dtype)
                # Overwrite GPS-resolution counts in cycles_v
                drop_cols = [c for c in bms_vsag.columns
                             if c != "session_id" and c in cycles_v.columns]
                cycles_v = cycles_v.drop(columns=drop_cols)
                cycles_v = cycles_v.merge(bms_vsag, on="session_id", how="left")
                # Recompute rate-of-change for vsag/IR from updated values
                # Diff across discharge sessions only — see extract_cycles() for rationale.
                for src_col, roc_col in [("n_vsag",      "d_vsag_per_cycle"),
                                         ("n_high_ir",    "d_n_high_ir"),
                                         ("ir_ohm_mean",  "d_ir_ohm_per_cycle")]:
                    if src_col not in cycles_v.columns:
                        continue
                    disc_idx = cycles_v["session_type"] == "discharge"
                    disc_sub = cycles_v[disc_idx].sort_values("start_time")
                    roc_vals = disc_sub[src_col].diff()
                    cycles_v[roc_col] = np.nan
                    cycles_v.loc[disc_idx, roc_col] = roc_vals.values

        all_cycles.append(cycles_v)

        # Keep minimal discharge rows for sequence extraction
        seq_cols = list({*SEQ_FEATURES,
                         "registration_number", "session_id", "session_type", "gps_time"})
        seq_cols = [c for c in seq_cols if c in df_v.columns]
        all_disc_rows.append(df_v[df_v["session_type"] == "discharge"][seq_cols].copy())

        # Keep small sample for anomaly report (first 5k rows per vehicle)
        anomaly_frames.append(df_v.head(5000)[[
            c for c in ("registration_number", "gps_time", "latitude", "longitude",
                        "speed", "voltage", "current", "soc", "vcu_odometer",
                        "cell_undervoltage", "cell_overvoltage", "cell_spread_warn")
            if c in df_v.columns
        ]])

        del df_v  # free per-vehicle RAM immediately

    # ── Fleet BMS match rate summary (Fix E) ──────────────────────────────────
    if vehicle_bms_rates:
        rates = np.array(vehicle_bms_rates)
        print(f"\n  BMS match rates: fleet median {np.median(rates):.1%}  "
              f"(min {rates.min():.1%}  max {rates.max():.1%}  "
              f">=80%: {(rates >= 0.8).sum()}/{len(rates)} vehicles)")

    if vehicle_vcu_rates:
        vcu_rates = np.array(vehicle_vcu_rates)
        print(f"\n  VCU odometer coverage: fleet median {np.median(vcu_rates):.1%}  "
              f"(min {vcu_rates.min():.1%}  max {vcu_rates.max():.1%}  "
              f">=90%: {(vcu_rates >= 0.9).sum()}/{len(vcu_rates)} vehicles)")

    # ── Combine ───────────────────────────────────────────────────────────────
    print("\nCombining vehicle results ...")
    cycles = pd.concat(all_cycles).reset_index(drop=True)
    df_disc = pd.concat(all_disc_rows).reset_index(drop=True) if all_disc_rows else pd.DataFrame()

    # Fleet anomaly report on sampled data
    if anomaly_frames:
        report_anomalies(pd.concat(anomaly_frames).reset_index(drop=True),
                         label="post-join sample (5k rows/vehicle)")

    # Fleet-level post-processing
    cycles = add_capacity_soh(cycles)
    cycles = add_fleet_flags(cycles)

    # ── SOH reliability flag ──────────────────────────────────────────────────
    # Marks sessions where the SOC swing is too small for a trustworthy capacity_soh.
    # Discharge: soc_range < MIN_SOC_RANGE_DISC_PCT (15%) → ≤7% quantisation error
    # Charging : soc_range < MIN_SOC_RANGE_PCT       (10%)
    _disc_low = (cycles["session_type"] == "discharge") & (cycles["soc_range"].fillna(0) < MIN_SOC_RANGE_DISC_PCT)
    _chg_low  = (cycles["session_type"] == "charging")  & (cycles["soc_range"].fillna(0) < MIN_SOC_RANGE_PCT)
    cycles["soh_low_confidence"] = (_disc_low | _chg_low)
    n_low = cycles["soh_low_confidence"].sum()
    print(f"\n  SOH low-confidence sessions: {n_low:,} "
          f"({n_low / len(cycles):.1%})  "
          f"[disc soc_range<{MIN_SOC_RANGE_DISC_PCT}% or chg soc_range<{MIN_SOC_RANGE_PCT}%]")

    # ── Engineered features for ML degradation models ─────────────────────────
    cycles = add_engineered_features(cycles)

    # ── Final sanity report ───────────────────────────────────────────────────
    n_disc = (cycles["session_type"] == "discharge").sum()
    n_chg  = (cycles["session_type"] == "charging").sum()
    n_idle = (cycles["session_type"] == "idle").sum()
    print(f"\n  Total sessions : {len(cycles):,}")
    print(f"  Discharge      : {n_disc:,}")
    print(f"  Charging       : {n_chg:,}")
    print(f"  Idle           : {n_idle:,}")

    if "detailed_type" in cycles.columns:
        print(f"\n  Detailed type breakdown:\n"
              f"{cycles['detailed_type'].value_counts().to_string()}")

    if "n_discharge_rows" in cycles.columns and "n_charge_rows" in cycles.columns:
        disc_c   = cycles[cycles["session_type"] == "discharge"]
        total_dr = disc_c["n_discharge_rows"].fillna(0).sum()
        total_cr = disc_c["n_charge_rows"].fillna(0).sum()
        total    = total_dr + total_cr
        if total > 0:
            print(f"\n  Within-session GPS row counts (discharge sessions only):")
            print(f"    moving_discharging rows : {total_dr:,.0f}  ({total_dr/total:.1%})")
            print(f"    moving_charging rows    : {total_cr:,.0f}  ({total_cr/total:.1%})")
            print(f"    regen sparsity          : {total_cr/total:.1%} of drive rows  (expect <20% for trucks)")
        if "capacity_ah_discharge" in cycles.columns and "capacity_ah_charge" in cycles.columns:
            tot_disc_ah  = disc_c["capacity_ah_discharge"].fillna(0).sum()
            tot_regen_ah = disc_c["capacity_ah_charge"].fillna(0).sum()
            print(f"    capacity_ah_discharge   : {tot_disc_ah:,.1f} Ah")
            print(f"    capacity_ah_charge      : {tot_regen_ah:,.1f} Ah  ({tot_regen_ah/(tot_disc_ah+1e-9):.1%} of discharge Ah)")

    wrong_disc = ((cycles["session_type"] == "discharge") & (cycles["soc_diff"] > 5)).sum()
    wrong_chg  = ((cycles["session_type"] == "charging")  & (cycles["soc_diff"] < -5)).sum()
    print(f"\n  Pool integrity:")
    print(f"    Discharge with SoC gain >5%: {wrong_disc}  "
          f"({'⚠' if wrong_disc else '✓'})")
    print(f"    Charging  with SoC drop >5%: {wrong_chg}  "
          f"({'⚠' if wrong_chg else '✓'})")

    disc_rows = cycles[cycles["session_type"] == "discharge"]
    chg_rows  = cycles[cycles["session_type"] == "charging"]
    print(f"\n  capacity_ah coverage:")
    for col, lbl in [("capacity_ah_discharge",  "Discharge Ah"),
                     ("capacity_ah_charge",      "Regen Ah    "),
                     ("capacity_ah_charge_total","Charge Ah   ")]:
        tgt = disc_rows if "discharge" in col or col == "capacity_ah_charge" else chg_rows
        if col in tgt.columns:
            print(f"    {lbl} — mean: {tgt[col].mean():.1f}  median: {tgt[col].median():.1f}")

    print(f"\n  BMS SoH distribution:\n{cycles['soh'].value_counts().sort_index().to_string()}")
    if "capacity_soh" in cycles.columns and cycles["capacity_soh"].notna().any():
        by_src = cycles.groupby("capacity_soh_source")["capacity_soh"].agg(["count","mean","std"])
        print(f"\n  Capacity-SoH by source:\n{by_src.to_string()}")

    for col, lbl in [("n_cell_undervoltage", f"Cell UV (<{CELL_V_WARN_LO}V) sessions"),
                     ("n_cell_overvoltage",  f"Cell OV (>{CELL_V_WARN_HI}V) sessions"),
                     ("n_cell_spread_warn",  f"Cell spread >{CELL_SPREAD_WARN}V sessions")]:
        if col in cycles.columns:
            print(f"  {lbl}: {(cycles[col] > 0).sum():,}")

    # Loaded vs unloaded EPK
    if "is_loaded" in cycles.columns and "energy_per_km" in cycles.columns:
        dm = cycles["session_type"] == "discharge"
        for grp_name, mask in [("Unloaded (outbound)", dm & (cycles["is_loaded"] == 0)),
                                ("Loaded   (inbound)",  dm & (cycles["is_loaded"] == 1))]:
            epk = cycles.loc[mask, "energy_per_km"].dropna()
            if len(epk):
                print(f"  EPK {grp_name}: n={len(epk)}  "
                      f"median={epk.median():.3f}  mean={epk.mean():.3f}  "
                      f"p25={epk.quantile(0.25):.3f}  p75={epk.quantile(0.75):.3f} kWh/km")

    # ── Column reorder ────────────────────────────────────────────────────────
    _col_order = [
        # Identity
        "registration_number", "session_id", "cycle_number",
        "session_type", "detailed_type",
        # Time
        "start_time_ist", "end_time_ist", "start_time", "end_time",
        "duration_hr", "time_delta_hr",
        # SOC
        "soc_start", "soc_end", "soc_diff", "soc_range", "n_low_soc",
        # SOH — BMS reported
        "soh",
        # Capacity SOH
        "capacity_soh", "capacity_soh_source", "ref_capacity_ah", "soh_low_confidence",
        # Electrical
        "voltage_mean", "voltage_min",
        "current_mean", "current_max", "current_mean_discharge", "current_mean_charge",
        # Capacity / energy
        "capacity_ah_discharge", "capacity_ah_charge", "capacity_ah_plugin",
        "capacity_ah", "capacity_ah_charge_total",
        "energy_kwh", "energy_per_km", "charging_rate_kw",
        # Voltage sag
        "n_vsag", "d_vsag_per_cycle",
        # Internal resistance
        "ir_ohm_mean", "n_high_ir", "d_n_high_ir", "d_ir_ohm_per_cycle",
        # Cell health
        "cell_spread_mean", "cell_spread_max",
        "n_cell_undervoltage", "n_cell_overvoltage", "n_cell_spread_warn", "cell_health_poor",
        # Temperature
        "temp_start", "temp_max", "temp_mean", "temp_lowest_mean",
        "temp_rise_rate", "rapid_heating",
        # Session quality
        "n_rows", "n_discharge_rows", "n_charge_rows",
        "bms_coverage", "ecu_fault_suspected",
        # Block linkage
        "block_id", "block_type", "block_soc_diff", "block_capacity_ah",
        "block_n_sessions", "block_odometer_km",
        # Location / movement
        "lat_start", "lon_start", "altitude_mean", "altitude_range",
        "speed_mean", "speed_max",
        "odometer_start", "odometer_end", "odometer_km",
        "load_direction", "is_loaded",
        # Fleet flags
        "high_energy_per_km", "slow_charging", "fast_charging",
        # Power limits
        "max_disc_lim", "max_chg_lim", "max_disc_pwr_lim", "max_chg_pwr_lim",
        # Insulation
        "insulation_mean",
        # ── Engineered features for ML degradation models ──────────────────
        # EFC & aging clock
        "cum_efc", "days_since_first", "aging_index",
        # Normalised rate features (remove session-length confound)
        "vsag_rate_per_hr", "ir_event_rate",
        # EWM-smoothed signals (span=10)
        "ir_ohm_mean_ewm10", "cell_spread_mean_ewm10",
        "temp_rise_rate_ewm10", "vsag_rate_per_hr_ewm10",
        # Rolling OLS trend slopes (20-session window)
        "vsag_trend_slope", "ir_event_trend_slope", "ir_ohm_trend_slope",
        "spread_trend_slope", "soh_trend_slope",
        # Operating stress
        "c_rate_chg", "dod_stress", "thermal_stress", "energy_per_loaded_session",
        # Load direction
        "load_direction_enc",
    ]
    _existing = [c for c in _col_order if c in cycles.columns]
    _remaining = [c for c in cycles.columns if c not in set(_existing)]
    cycles = cycles[_existing + _remaining]

    # Save to a temp name first, then rename — avoids PermissionError if cycles.csv
    # is open in Excel or another program.
    import tempfile, shutil
    _tmp = CYCLES_CSV + ".tmp"
    cycles.to_csv(_tmp, index=False)
    try:
        if os.path.exists(CYCLES_CSV):
            os.replace(CYCLES_CSV, CYCLES_CSV + ".bak")
        shutil.move(_tmp, CYCLES_CSV)
        print(f"\nSaved cycles → {CYCLES_CSV}  ({len(cycles):,} sessions)")
    except PermissionError:
        alt = CYCLES_CSV.replace("cycles.csv", "cycles_new.csv")
        shutil.move(_tmp, alt)
        print(f"\nPermissionError on {CYCLES_CSV} (open in Excel?).")
        print(f"Saved cycles → {alt}  ({len(cycles):,} sessions)")

    # ── Sequence extraction ───────────────────────────────────────────────────
    print("\nExtracting discharge sequences ...")
    sequences, seq_meta = extract_sequences(df_disc, cycles)

    if sequences:
        arr = np.stack(sequences).astype(np.float32)
        np.save(SEQ_NPY, arr)
        pd.DataFrame(seq_meta).to_csv(SEQ_META, index=False)
        print(f"Saved sequences → {SEQ_NPY}  shape={arr.shape}")
        print(f"Saved meta      → {SEQ_META}")
    else:
        print("No sequences extracted.")
