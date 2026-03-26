import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) if "__file__" in dir() else os.getcwd())

import numpy as np
import pandas as pd
from tqdm import tqdm
from config import (
    BMS_FILE, GPS_FILE, VCU_FILE, CYCLES_CSV, SEQ_NPY, SEQ_META,
    VOLTAGE_RANGE, CURRENT_RANGE, CELL_V_RANGE, TEMP_RANGE, SOH_MIN,
    DISCHARGE_A, CHARGE_A, MIN_SESSION_MIN, MIN_BMS_ROWS, MAX_DT_MIN, TRIP_GAP_MIN,
    NUM_BINS, SEQ_FEATURES, SCALAR_FEATURES,
    IR_THRESHOLD_MOHM, LOW_SOC_PCT, BATTERY_CAPACITY_KWH, NOMINAL_CAPACITY_AH, NOMINAL_VOLTAGE_V,
    REGEN_SPEED_KPH,          # moved from module-level constant
    GPS_GAP_MAX_SEC,           # new: GPS staleness window
    ODO_GAP_MAX_SEC,           # new: VCU odometer staleness window
    EPK_MAX_KWH_KM,            # new: physical EPK upper bound
)

CHUNK_SIZE    = 200_000
SMOOTH_WINDOW = 5          # rows for current smoothing (~50 s at 10 s BMS rate)

# ── capacity_soh quality gates ─────────────────────────────────────────────────
# Fix 1+2: only sessions with sufficient SoC swing yield reliable normalized capacity
MIN_SOC_RANGE_PCT  = 10.0   # % — sessions with shallower swing excluded from reference
# Fix 3: vehicles with too few qualifying sessions fall back to fleet reference
MIN_REF_SESSIONS   = 5      # minimum deep+dense sessions to build own reference
# Fix 5: at 10 s BMS rate expected ~360 rows/hr; below 20% = too many timestamp gaps
BMS_RATE_PER_HR    = 360    # rows/hr at nominal 10 s polling
MIN_DATA_DENSITY   = BMS_RATE_PER_HR * 0.20   # 72 rows/hr minimum coverage

# All BMS columns to load from the source CSV
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

GPS_LOAD_COLS = ["registration_number", "gps_time", "latitude", "longitude", "speed"]
VCU_LOAD_COLS = ["registration_number", "gps_time", "vcu_odometer"]


# ── 0a. Load GPS ───────────────────────────────────────────────────────────────

def load_gps(path: str) -> pd.DataFrame:
    print(f"Loading GPS from {path} ...")
    gps = pd.read_csv(path, usecols=GPS_LOAD_COLS,
                      dtype={"speed": "float32", "latitude": "float64", "longitude": "float64"},
                      low_memory=False)
    gps["gps_time"] = pd.to_numeric(gps["gps_time"], errors="coerce")
    gps = gps.dropna(subset=["gps_time", "registration_number"]).sort_values(
        ["registration_number", "gps_time"]).reset_index(drop=True)
    print(f"  GPS rows: {len(gps):,}  |  Vehicles: {gps['registration_number'].nunique()}")
    return gps


# ── 0b. Load VCU (odometer) ────────────────────────────────────────────────────

def load_vcu(path: str) -> pd.DataFrame:
    print(f"Loading VCU (odometer) from {path} ...")
    vcu = pd.read_csv(path, usecols=VCU_LOAD_COLS,
                      dtype={"vcu_odometer": "float64"}, low_memory=False)
    vcu["gps_time"] = pd.to_numeric(vcu["gps_time"], errors="coerce")
    vcu = vcu.dropna(subset=["gps_time", "registration_number", "vcu_odometer"]).sort_values(
        ["registration_number", "gps_time"]).reset_index(drop=True)
    print(f"  VCU rows: {len(vcu):,}  |  Vehicles: {vcu['registration_number'].nunique()}")
    return vcu


# ── Helper: gap-aware interpolation ──────────────────────────────────────────

def _interp_with_gap(t_bms: np.ndarray, t_src: np.ndarray,
                     v_src: np.ndarray, gap_max_ms: float) -> np.ndarray:
    """
    Linear interpolation of source signal onto BMS timestamps, with a validity
    window: any BMS row whose nearest source record is farther than gap_max_ms
    milliseconds away receives NaN instead of the extrapolated/interpolated value.

    This prevents np.interp edge-clamping from propagating stale GPS speed or
    odometer values across long gaps (e.g. GPS silence during depot charging).
    """
    result = np.interp(t_bms, t_src, v_src)           # linear interp + edge clamp
    lo = np.searchsorted(t_src, t_bms, side="right") - 1
    hi = (lo + 1).clip(0, len(t_src) - 1)
    lo = lo.clip(0, len(t_src) - 1)
    dist = np.minimum(np.abs(t_bms - t_src[lo]),
                      np.abs(t_bms - t_src[hi]))
    result = np.where(dist > gap_max_ms, np.nan, result)
    return result


# ── 1. Load & clean BMS, interpolate GPS + VCU onto BMS timestamps ────────────

def load_and_clean(bms_path: str, gps_df: pd.DataFrame, vcu_df: pd.DataFrame) -> pd.DataFrame:
    """
    1. Load all BMS columns in chunks, apply range filters.
    2. Per vehicle, linearly interpolate GPS (lat/lon/speed) and VCU (odometer)
       onto BMS timestamps — no BMS row is ever dropped due to missing GPS/VCU.
       np.interp clamps to edge values outside the source time range.
    3. Add derived columns.
    """
    print(f"Loading BMS from {bms_path} ...")
    vlo, vhi = VOLTAGE_RANGE
    ilo, ihi = CURRENT_RANGE
    clo, chi = CELL_V_RANGE
    tlo, thi = TEMP_RANGE

    reader_test = pd.read_csv(bms_path, nrows=0)
    existing_cols = set(reader_test.columns)
    load_cols  = [c for c in BMS_LOAD_COLS  if c in existing_cols]
    float_cols = [c for c in BMS_FLOAT_COLS if c in existing_cols]
    missing = set(BMS_LOAD_COLS) - existing_cols
    if missing:
        print(f"  Note: {len(missing)} BMS columns not in file: {sorted(missing)}")

    vehicle_frames: dict[str, list] = {}
    raw_total, clean_total = 0, 0

    reader = pd.read_csv(bms_path, usecols=load_cols,
                         dtype={c: "float32" for c in float_cols},
                         chunksize=CHUNK_SIZE, low_memory=False)
    for chunk in reader:
        raw_total += len(chunk)
        chunk["gps_time"] = pd.to_numeric(chunk["gps_time"], errors="coerce")
        chunk = chunk.dropna(subset=["gps_time", "registration_number", "voltage"])
        chunk = chunk[
            chunk["voltage"].between(vlo, vhi) &
            chunk["current"].between(ilo, ihi) &
            chunk["soh"].ge(SOH_MIN) &
            chunk["min_cell_voltage"].between(clo, chi) &
            chunk["max_cell_voltage"].between(clo, chi) &
            chunk["temperature_lowest"].between(tlo, thi) &
            chunk["temperature_highest"].between(tlo, thi)
        ]
        clean_total += len(chunk)
        for reg, grp in chunk.groupby("registration_number", sort=False):
            vehicle_frames.setdefault(reg, []).append(grp)

    print(f"  Raw: {raw_total:,}  |  Clean: {clean_total:,}  |  Vehicles: {len(vehicle_frames)}")

    parts = []
    for reg, frames in vehicle_frames.items():
        bms_v = pd.concat(frames).sort_values("gps_time").reset_index(drop=True)
        t_bms = bms_v["gps_time"].values.astype(np.float64)

        # ── GPS interpolation ─────────────────────────────────────────────────
        gps_v = gps_df[gps_df["registration_number"] == reg].sort_values("gps_time")
        if gps_v.empty:
            bms_v["latitude"]  = np.nan
            bms_v["longitude"] = np.nan
            bms_v["gps_speed"] = np.nan          # NaN prevents false regen tagging
        else:
            t_g   = gps_v["gps_time"].values.astype(np.float64)
            gap_g = GPS_GAP_MAX_SEC * 1000        # ms
            bms_v["latitude"]  = _interp_with_gap(t_bms, t_g, gps_v["latitude"].values,  gap_g)
            bms_v["longitude"] = _interp_with_gap(t_bms, t_g, gps_v["longitude"].values, gap_g)
            bms_v["gps_speed"] = _interp_with_gap(t_bms, t_g, gps_v["speed"].values,     gap_g)

        # ── VCU odometer interpolation ────────────────────────────────────────
        vcu_v = vcu_df[vcu_df["registration_number"] == reg].sort_values("gps_time")
        if vcu_v.empty:
            bms_v["vcu_odometer"] = np.nan
        else:
            t_v   = vcu_v["gps_time"].values.astype(np.float64)
            gap_v = ODO_GAP_MAX_SEC * 1000        # ms
            bms_v["vcu_odometer"] = _interp_with_gap(t_bms, t_v,
                                                     vcu_v["vcu_odometer"].values, gap_v)

        parts.append(bms_v)

    df = pd.concat(parts).reset_index(drop=True)
    print(f"  BMS rows after join: {len(df):,}  |  Vehicles: {df['registration_number'].nunique()}")

    df["cell_spread"]  = (df["max_cell_voltage"] - df["min_cell_voltage"]).astype("float32")
    df["temp_highest"] = df["temperature_highest"]
    return df


# ── 2. Detect sessions ─────────────────────────────────────────────────────────

def label_sessions(df: pd.DataFrame) -> pd.DataFrame:
    """
    1. Smooth current per vehicle.
    2. Assign row-level session_type (3-way: discharge / charging / idle).
       Regen correction: charging current while GPS speed is *reliably* above
       REGEN_SPEED_KPH → reclassify to discharge (for session_id continuity).
       GPS speed of NaN (stale/missing) is treated as stationary — no regen correction.
    3. Assign row-level detailed_type (5-way):
         moving_discharging   — driving with net discharge
         discharging_stopped  — discharge current but vehicle stationary
         moving_charging      — regen braking (negative current while moving)
         charging             — plug-in / depot charging (stationary or GPS stale)
         stop                 — vehicle stationary, current in dead zone
         idle                 — current in dead zone, GPS stale or vehicle coasting
    4. Assign session_id via run-length encoding on session_type.
    """
    df = df.copy()
    df["current_sm"] = (
        df.groupby("registration_number")["current"]
        .transform(lambda s: s.rolling(SMOOTH_WINDOW, min_periods=1, center=True).mean())
    )

    # ── 3-type session_type (for session_id boundary detection) ──────────────
    df["session_type"] = "idle"
    df.loc[df["current_sm"] > DISCHARGE_A, "session_type"] = "discharge"
    df.loc[df["current_sm"] < CHARGE_A,    "session_type"] = "charging"

    # ── GPS-aware regen correction ────────────────────────────────────────────
    # Only apply when gps_speed is not NaN (i.e. a GPS record exists within
    # GPS_GAP_MAX_SEC). NaN speed = GPS stale → assume stationary → no regen.
    has_gps = "gps_speed" in df.columns
    if has_gps:
        moving     = df["gps_speed"].notna() & (df["gps_speed"] > REGEN_SPEED_KPH)
        stopped    = df["gps_speed"].notna() & (df["gps_speed"] <= REGEN_SPEED_KPH)
        gps_stale  = df["gps_speed"].isna()
        regen_mask = (df["session_type"] == "charging") & moving
        df.loc[regen_mask, "session_type"] = "discharge"
    else:
        moving     = pd.Series(False, index=df.index)
        stopped    = pd.Series(False, index=df.index)
        gps_stale  = pd.Series(True,  index=df.index)
        regen_mask = pd.Series(False, index=df.index)

    # ── 5-type detailed_type ──────────────────────────────────────────────────
    disc_rows = df["current_sm"] > DISCHARGE_A
    chg_rows  = df["current_sm"] < CHARGE_A
    dead_zone = ~disc_rows & ~chg_rows

    df["detailed_type"] = "idle"                                              # fallback
    df.loc[disc_rows & moving,              "detailed_type"] = "moving_discharging"
    df.loc[disc_rows & stopped,             "detailed_type"] = "discharging_stopped"
    df.loc[regen_mask,                      "detailed_type"] = "moving_charging"
    df.loc[chg_rows & (stopped | gps_stale), "detailed_type"] = "charging"
    df.loc[dead_zone & stopped,             "detailed_type"] = "stop"
    # Rows with disc_rows but no GPS info remain "idle" (can't determine motion)

    # ── session_id: run-length encoding on session_type ───────────────────────
    df["session_id"] = (
        df.groupby("registration_number")["session_type"]
        .transform(lambda s: (s != s.shift()).cumsum())
    )
    df.drop(columns="current_sm", inplace=True)
    return df


# ── 3. Vectorised cycle feature extraction ────────────────────────────────────

def extract_cycles(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute session-level aggregates including:
      - Coulomb counting (capacity_ah)
      - Voltage sag counts (mild / moderate / severe) at high current
      - Internal resistance flag count (|ΔV/ΔI| > IR_THRESHOLD_MOHM at high current)
      - Low-SoC row count (soc < LOW_SOC_PCT)
      - Odometer distance per session (from VCU)
      - Temperature at session start (for heating-rate computation)
    """
    SK = ["registration_number", "session_id", "session_type"]

    # ── Coulomb counting ──────────────────────────────────────────────────────
    df["_dt_hr"]   = df.groupby(SK[:2])["gps_time"].diff().fillna(0) / 3_600_000
    df["_dt_hr"]   = df["_dt_hr"].clip(0, MAX_DT_MIN / 60)
    df["_delta_q"] = df["current"] * df["_dt_hr"]

    # ── Data-driven thresholds (computed from actual discharge rows) ─────────────
    # Fixed thresholds (e.g. 610V, 150A) don't apply to this fleet's voltage/current
    # range, so we derive them from the data distribution instead.
    _disc_rows = df[df["session_type"] == "discharge"]
    vsag_severe   = float(_disc_rows["voltage"].quantile(0.05))   # p5  — severe sag
    vsag_moderate = float(_disc_rows["voltage"].quantile(0.10))   # p10 — moderate sag
    vsag_mild     = float(_disc_rows["voltage"].quantile(0.25))   # p25 — mild sag
    high_curr_thr = float(_disc_rows["current"].quantile(0.75))   # p75 — high discharge current

    # rest_v: mean voltage when current is near-zero (battery at rest), same as bda_4
    _rest_rows = df[df["current"].abs() < 5.0]
    rest_v = float(_rest_rows["voltage"].mean()) if len(_rest_rows) > 0 else vsag_mild
    print(f"  Rest voltage (idle mean): {rest_v:.1f}V")
    print(f"  Voltage sag thresholds  : mild<{vsag_mild:.1f}V  "
          f"moderate<{vsag_moderate:.1f}V  severe<{vsag_severe:.1f}V")
    print(f"  High current threshold  : >{high_curr_thr:.1f}A  (p75 of discharge current)")

    # ── Voltage sag flags (transition-based, same logic as bda_4) ─────────────
    # A sag is counted only when the previous row was at/above rest voltage and the
    # current row dropped below the sag threshold — i.e. a genuine sudden drop,
    # not sustained low voltage that would be counted on every consecutive row.
    _hc       = df["current"] > high_curr_thr
    _prev_vol = df.groupby(SK[:2])["voltage"].shift(1)
    _from_rest = _prev_vol >= rest_v                              # previous row was at rest level
    df["_vsag_severe"]   = (df["voltage"] < vsag_severe)                                       & _hc & _from_rest
    df["_vsag_moderate"] = ((df["voltage"] < vsag_moderate) & (df["voltage"] >= vsag_severe)  ) & _hc & _from_rest
    df["_vsag_mild"]     = ((df["voltage"] < vsag_mild)     & (df["voltage"] >= vsag_moderate)) & _hc & _from_rest

    # ── Internal resistance: |ΔV / ΔI| > IR_THRESHOLD (0.03 Ω) at high current ─
    # ΔI minimum of 2A filters out noise-dominated ratios from near-constant current
    _dv = df.groupby(SK[:2])["voltage"].diff()
    _di = df.groupby(SK[:2])["current"].diff()
    _valid_di = _di.abs() >= 2.0                              # require meaningful current step
    df["_ir_ohm"]  = np.where(_valid_di, _dv.abs() / _di.abs(), np.nan)
    df["_high_ir"] = (df["_ir_ohm"] > (IR_THRESHOLD_MOHM / 1000)) & _hc & _valid_di

    # ── Low-SoC flag ──────────────────────────────────────────────────────────
    df["_low_soc"] = df["soc"] < LOW_SOC_PCT

    # ── Aggregation ───────────────────────────────────────────────────────────
    agg_spec = dict(
        n_rows               = ("gps_time",       "count"),
        start_time           = ("gps_time",        "min"),
        end_time             = ("gps_time",        "max"),
        soh                  = ("soh",             "median"),
        soc_start            = ("soc",             "first"),
        soc_end              = ("soc",             "last"),
        temp_start           = ("temp_highest",    "first"),
        voltage_mean         = ("voltage",         "mean"),
        voltage_min          = ("voltage",         "min"),
        current_mean         = ("current",         "mean"),
        current_max          = ("current",         "max"),
        cell_spread_mean     = ("cell_spread",     "mean"),
        cell_spread_max      = ("cell_spread",     "max"),
        temp_max             = ("temp_highest",    "max"),
        temp_mean            = ("temp_highest",    "mean"),
        temp_lowest_mean     = ("temperature_lowest", "mean"),
        capacity_ah          = ("_delta_q",        lambda x: abs(x.sum())),
        insulation_mean      = ("insulation_resistance",        "mean"),
        max_disc_lim         = ("max_discharge_current_limit",  "mean"),
        max_chg_lim          = ("max_charge_current_limit",     "mean"),
        max_disc_pwr_lim     = ("max_discharge_power_limit",    "mean"),
        max_chg_pwr_lim      = ("max_charge_power_limit",       "mean"),
        # Health flags
        n_vsag_severe        = ("_vsag_severe",    "sum"),
        n_vsag_moderate      = ("_vsag_moderate",  "sum"),
        n_vsag_mild          = ("_vsag_mild",       "sum"),
        n_high_ir            = ("_high_ir",         "sum"),
        n_low_soc            = ("_low_soc",         "sum"),
    )

    if "detailed_type" in df.columns:
        agg_spec["detailed_type"] = ("detailed_type",
                                     lambda x: x.mode().iloc[0] if len(x) > 0 else "idle")

    if "gps_speed" in df.columns:
        agg_spec["speed_mean"] = ("gps_speed", "mean")
        agg_spec["speed_max"]  = ("gps_speed", "max")
    if "latitude" in df.columns:
        agg_spec["lat_start"] = ("latitude",  "first")
        agg_spec["lon_start"] = ("longitude", "first")
    if "vcu_odometer" in df.columns:
        agg_spec["odometer_start"] = ("vcu_odometer", "first")
        agg_spec["odometer_end"]   = ("vcu_odometer", "last")

    agg = df.groupby(SK).agg(**agg_spec).reset_index()

    # ── Derived session-level columns ─────────────────────────────────────────
    agg["duration_hr"]   = (agg["end_time"] - agg["start_time"]) / 3_600_000
    agg["energy_kwh"]    = agg["capacity_ah"] * agg["voltage_mean"] / 1000

    disc_mask = agg["session_type"] == "discharge"
    chg_mask  = agg["session_type"] == "charging"

    # soc_diff: raw signed change (soc_end - soc_start)
    #   discharge: negative (SoC falls)  |  charging: positive (SoC rises)
    agg["soc_diff"] = agg["soc_end"] - agg["soc_start"]

    # soc_range: absolute magnitude of SoC change — always >= 0
    #   discharge: how much SoC was consumed  |  charging: how much SoC was gained
    agg["soc_range"] = np.where(
        disc_mask, (agg["soc_start"] - agg["soc_end"]).abs(),
        np.where(chg_mask, (agg["soc_end"] - agg["soc_start"]).abs(), np.nan)
    )

    # Temperature rise rate (°C/hr) — proxy for abnormal heating
    agg["temp_rise_rate"] = (agg["temp_max"] - agg["temp_start"]) / agg["duration_hr"].replace(0, np.nan)

    # Odometer distance and energy-per-km (discharge sessions with odometer)
    if "odometer_start" in agg.columns:
        agg["odometer_km"] = (agg["odometer_end"] - agg["odometer_start"]).clip(lower=0)
        valid_km = disc_mask & (agg["odometer_km"] > 0.5)   # at least 0.5 km to avoid division noise
        agg["energy_per_km"] = np.where(
            valid_km,
            (agg["soc_range"] / agg["odometer_km"]) * BATTERY_CAPACITY_KWH,
            np.nan
        )

    # Charging rate (kW) — for charging sessions
    agg["charging_rate_kw"] = np.where(
        chg_mask & (agg["duration_hr"] > 0),
        agg["energy_kwh"] / agg["duration_hr"],
        np.nan
    )

    # IST datetime columns placed immediately after their unix timestamps
    def to_ist(ms_col):
        return (pd.to_datetime(agg[ms_col], unit="ms", utc=True)
                  .dt.tz_convert("Asia/Kolkata")
                  .dt.tz_localize(None)
                  .dt.strftime("%Y-%m-%d %H:%M:%S"))

    agg.insert(agg.columns.get_loc("start_time") + 1, "start_time_ist", to_ist("start_time"))
    agg.insert(agg.columns.get_loc("end_time")   + 1, "end_time_ist",   to_ist("end_time"))

    # Drop temp columns from raw df
    drop_cols = ["_dt_hr", "_delta_q", "_vsag_severe", "_vsag_moderate", "_vsag_mild",
                 "_high_ir", "_ir_ohm", "_low_soc"]
    df.drop(columns=[c for c in drop_cols if c in df.columns], inplace=True)

    # Filter short / sparse sessions
    agg = agg[
        (agg["duration_hr"] >= MIN_SESSION_MIN / 60) &
        (agg["n_rows"] >= MIN_BMS_ROWS)
    ].reset_index(drop=True)

    # Remove anomalous charging sessions where SoC dropped (soc_diff < 0).
    # These indicate a mislabelled or corrupted session — not genuine charging.
    anomalous_chg = (agg["session_type"] == "charging") & (agg["soc_diff"] < 0)
    if anomalous_chg.sum():
        print(f"  Removing {anomalous_chg.sum()} anomalous charging sessions (soc_diff < 0)")
    agg = agg[~anomalous_chg].reset_index(drop=True)

    # Sort by vehicle + time so time_delta is computed correctly across all session types
    agg = agg.sort_values(["registration_number", "start_time"]).reset_index(drop=True)

    # time_delta_hr: gap (hours) between end of previous session and start of this one
    # per vehicle. NaN for each vehicle's first session.
    agg["time_delta_hr"] = (
        agg["start_time"] - agg.groupby("registration_number")["end_time"].shift(1)
    ) / 3_600_000

    agg["cycle_number"] = agg.groupby("registration_number").cumcount() + 1

    # Trip-level grouping: merges trip_soc_diff, trip_capacity_ah, etc. onto agg
    agg = compute_trip_linkage(agg)

    # Trip-level energy_per_km (replaces session-level version for discharge sessions)
    if "trip_odometer_km" in agg.columns and "trip_soc_diff" in agg.columns:
        disc_mask = agg["session_type"] == "discharge"
        valid_km  = disc_mask & (agg["trip_odometer_km"] > 0.5)
        agg["energy_per_km"] = np.where(
            valid_km,
            (agg["trip_soc_diff"].abs() / agg["trip_odometer_km"]) * BATTERY_CAPACITY_KWH,
            np.nan,
        )

    # Sanity cap: physically impossible EPK values indicate GPS/odometer artifacts.
    # 282 kWh / 5 kWh/km = 56 km range at absolute worst — anything above EPK_MAX_KWH_KM
    # is a data quality issue; set to NaN rather than letting it contaminate CUSUM EPK.
    if "energy_per_km" in agg.columns:
        n_capped = (agg["energy_per_km"] > EPK_MAX_KWH_KM).sum()
        if n_capped:
            print(f"  EPK sanity cap: {n_capped} sessions exceed {EPK_MAX_KWH_KM} kWh/km → NaN")
        agg.loc[agg["energy_per_km"] > EPK_MAX_KWH_KM, "energy_per_km"] = np.nan

    return agg


# ── 3b. Trip-level session grouping ──────────────────────────────────────────

def compute_trip_linkage(agg: pd.DataFrame) -> pd.DataFrame:
    """
    Group consecutive discharge sessions into "trips" using TRIP_GAP_MIN.
    A new trip starts when: it's the vehicle's first discharge session, the gap
    to the previous session > TRIP_GAP_MIN minutes, or time_delta_hr is NaN.

    Adds per-session columns:
      trip_soc_start, trip_soc_end, trip_soc_diff   — full trip SoC swing
      trip_capacity_ah                               — summed Coulombs over trip
      trip_n_sessions                                — number of micro-sessions
      trip_odometer_km                               — total trip distance (if available)
    """
    # Only group real discharge sessions (net positive current).
    # Regen sessions (current_mean < 0) are classified as "discharge" by the regen
    # correction but actually charge the battery — excluding them keeps trip_soc_diff clean.
    disc = agg[
        (agg["session_type"] == "discharge") & (agg["current_mean"] > 0)
    ].copy()
    disc = disc.sort_values(["registration_number", "start_time"])

    # Mark trip boundaries
    disc["_new_trip"] = (
        (disc["registration_number"] != disc["registration_number"].shift()) |
        disc["time_delta_hr"].isna() |
        (disc["time_delta_hr"] > TRIP_GAP_MIN / 60)
    )
    disc["trip_id"] = disc.groupby("registration_number")["_new_trip"].cumsum()

    # Aggregate per trip
    trip_cols = dict(
        trip_soc_start   = ("soc_start",   "first"),
        trip_soc_end     = ("soc_end",     "last"),
        trip_capacity_ah = ("capacity_ah", "sum"),
        trip_n_sessions  = ("session_id",  "count"),
    )
    if "odometer_km" in disc.columns:
        trip_cols["trip_odometer_km"] = ("odometer_km", "sum")

    trip_agg = disc.groupby(["registration_number", "trip_id"]).agg(**trip_cols).reset_index()
    trip_agg["trip_soc_diff"] = trip_agg["trip_soc_end"] - trip_agg["trip_soc_start"]

    # Count stop + idle sessions within each trip's time window.
    # High trip_n_stops indicates a stop/start route pattern.
    if "detailed_type" in agg.columns:
        stop_sessions = agg[agg["detailed_type"].isin(["stop", "idle"])][
            ["registration_number", "start_time", "end_time"]
        ].copy()

        if not stop_sessions.empty:
            # Get trip time bounds from the constituent discharge sessions
            trip_bounds = disc.groupby(["registration_number", "trip_id"]).agg(
                trip_start=("start_time", "min"),
                trip_end=("end_time",   "max"),
            ).reset_index()

            n_stops_rows = []
            for reg, reg_stops in stop_sessions.groupby("registration_number"):
                reg_trips = trip_bounds[trip_bounds["registration_number"] == reg]
                for _, t in reg_trips.iterrows():
                    n = ((reg_stops["start_time"] >= t["trip_start"]) &
                         (reg_stops["start_time"] <= t["trip_end"])).sum()
                    n_stops_rows.append({
                        "registration_number": reg,
                        "trip_id": t["trip_id"],
                        "trip_n_stops": int(n),
                    })

            if n_stops_rows:
                trip_stops_df = pd.DataFrame(n_stops_rows)
                trip_agg = trip_agg.merge(trip_stops_df,
                                          on=["registration_number", "trip_id"], how="left")
                trip_agg["trip_n_stops"] = trip_agg["trip_n_stops"].fillna(0).astype(int)
            else:
                trip_agg["trip_n_stops"] = 0
        else:
            trip_agg["trip_n_stops"] = 0

    # Join trip_id onto disc, then merge trip-level columns back via trip_id
    disc = disc[["registration_number", "session_id", "trip_id"]].merge(
        trip_agg.drop(columns=["trip_soc_start", "trip_soc_end"]),
        on=["registration_number", "trip_id"], how="left"
    )

    # Join back onto full agg (non-discharge rows get NaN)
    return agg.merge(
        disc.drop(columns="trip_id"),
        on=["registration_number", "session_id"], how="left"
    )


# ── 4. Fleet-level percentile flags ───────────────────────────────────────────

def add_fleet_flags(cycles: pd.DataFrame) -> pd.DataFrame:
    """
    Add boolean fleet-percentile flags after all sessions are aggregated:
      - rapid_heating    : temp_rise_rate > 75th pct of non-zero fleet heating events
      - frequent_low_soc : n_low_soc > 0 in more than 25% of a vehicle's sessions
      - high_energy_per_km: energy_per_km > 75th pct of fleet (discharge sessions)
      - slow_charging    : charging_rate_kw < 25th pct of fleet charging sessions
      - fast_charging    : charging_rate_kw > 75th pct of fleet charging sessions
    """
    cycles = cycles.copy()

    # BMS coverage: fraction of session time actually sampled
    # At 10 s polling, n_rows * (10/3600) = expected hours of data
    cycles["bms_coverage"] = (
        (cycles["n_rows"] * (10.0 / 3600.0)) /
        cycles["duration_hr"].replace(0, np.nan)
    ).clip(0, 1.0)

    # Sessions where BMS was silent >80% of the time (broken ECU indicator)
    cycles["ecu_fault_suspected"] = cycles["bms_coverage"] < 0.20

    # Rapid heating: session-level flag — temp_rise_rate > p75 of non-zero values across fleet
    # Using per-vehicle medians results in all-False because ~65% of sessions have 0 rise
    # (temp_start == temp_max). Comparing sessions directly against the non-zero distribution
    # gives meaningful flags.
    nonzero_heat = cycles.loc[cycles["temp_rise_rate"] > 0, "temp_rise_rate"]
    heat_p75 = float(nonzero_heat.quantile(0.75)) if len(nonzero_heat) else 0.0
    cycles["rapid_heating"] = cycles["temp_rise_rate"] > heat_p75

    # High energy consumption (discharge only)
    if "energy_per_km" in cycles.columns:
        disc = cycles[cycles["session_type"] == "discharge"]
        epk_p75 = disc["energy_per_km"].quantile(0.75)
        cycles["high_energy_per_km"] = (
            (cycles["session_type"] == "discharge") &
            (cycles["energy_per_km"] > epk_p75)
        )

    # Slow / fast charging
    if "charging_rate_kw" in cycles.columns:
        chg = cycles[cycles["session_type"] == "charging"]["charging_rate_kw"].dropna()
        cr_p25, cr_p75 = chg.quantile(0.25), chg.quantile(0.75)
        cycles["slow_charging"] = (cycles["session_type"] == "charging") & (cycles["charging_rate_kw"] < cr_p25)
        cycles["fast_charging"] = (cycles["session_type"] == "charging") & (cycles["charging_rate_kw"] > cr_p75)

    return cycles


# ── 5. Capacity-based continuous SoH proxy ────────────────────────────────────

def add_capacity_soh(cycles: pd.DataFrame) -> pd.DataFrame:
    """
    Compute a normalised, depth-of-discharge-corrected SoH proxy.

    Five accuracy fixes vs. naive capacity_ah / p90:
      1. Normalise by abs(soc_diff): capacity_ah / (abs(soc_diff)/100) estimates
         full-cycle capacity regardless of how deeply the battery was discharged.
         soc_diff is the direct signed SoC change; abs() gives the DoD fraction.
      2. Per-vehicle reference built from normalised capacity, not raw capacity_ah,
         capped at NOMINAL_CAPACITY_AH so SoH is an absolute % of spec capacity.
      3. Fleet fallback = NOMINAL_CAPACITY_AH for vehicles with too few qualifying
         sessions — anchors all vehicles to the same physical spec baseline.
      4. Sessions with abs(soc_diff) < MIN_SOC_RANGE_PCT excluded from reference —
         shallow/fragmented sessions amplify noise in the normalisation.
      5. Sessions with sparse BMS coverage (rows/hr < MIN_DATA_DENSITY) excluded
         from reference — timestamp gaps cause Coulomb counting to undercount.
    """
    cycles = cycles.copy()
    disc   = cycles[cycles["session_type"] == "discharge"].copy()

    # ── Fix 1: normalise by abs(trip_soc_diff) — full-trip DoD fraction ──────
    dod_col = "trip_soc_diff" if "trip_soc_diff" in disc.columns else "soc_diff"
    cap_col = "trip_capacity_ah" if "trip_capacity_ah" in disc.columns else "capacity_ah"
    dod = disc[dod_col].abs()
    disc["norm_capacity_ah"] = (
        disc[cap_col] / (dod / 100.0)
    ).replace([np.inf, -np.inf], np.nan)

    # ── Fix 4 + 5 + 6: quality gate — only reliable sessions feed the reference ──
    disc["data_density"] = disc["n_rows"] / disc["duration_hr"].replace(0, np.nan)
    quality_mask = (
        (dod                  >= MIN_SOC_RANGE_PCT) &   # fix 4: deep enough (trip-level)
        (disc["data_density"] >= MIN_DATA_DENSITY)    & # fix 5: dense enough
        (disc["current_mean"] >  0)                     # fix 6: net positive current (not regen)
    )
    quality_disc = disc[quality_mask]

    n_total   = disc["registration_number"].nunique()
    n_quality = quality_disc["registration_number"].nunique()
    print(f"  capacity_soh reference: {len(quality_disc):,} qualifying sessions "
          f"across {n_quality}/{n_total} vehicles "
          f"(|soc_diff|>={MIN_SOC_RANGE_PCT}%, density>={MIN_DATA_DENSITY:.0f} rows/hr)")
    print(f"  Nominal reference      : {NOMINAL_CAPACITY_AH} Ah  "
          f"({BATTERY_CAPACITY_KWH} kWh / {NOMINAL_VOLTAGE_V} V)")

    # ── Fix 2: per-vehicle p90 capped at nominal — can't exceed spec capacity ─
    ref_vehicle = (
        quality_disc.groupby("registration_number")["norm_capacity_ah"]
        .agg(lambda s: min(s.quantile(0.90), NOMINAL_CAPACITY_AH)
             if len(s) >= MIN_REF_SESSIONS else np.nan)
    )

    # ── Fix 3: fallback = NOMINAL_CAPACITY_AH (not fleet p90) ────────────────
    n_fallback = ref_vehicle.isna().sum()
    if n_fallback:
        print(f"  {n_fallback} vehicle(s) use nominal fallback ({NOMINAL_CAPACITY_AH} Ah) "
              f"— fewer than {MIN_REF_SESSIONS} qualifying sessions")
    ref_vehicle = ref_vehicle.fillna(NOMINAL_CAPACITY_AH)

    cycles = cycles.join(ref_vehicle.rename("ref_capacity_ah"), on="registration_number")
    cycles["ref_capacity_ah"] = cycles["ref_capacity_ah"].fillna(NOMINAL_CAPACITY_AH)

    # ── Compute capacity_soh for discharge sessions only ─────────────────────
    disc_mask = cycles["session_type"] == "discharge"
    dod_all   = cycles[dod_col].abs()
    cap_all   = cycles[cap_col]
    norm_cap  = (
        cap_all / (dod_all / 100.0)
    ).replace([np.inf, -np.inf], np.nan)

    # Only assign capacity_soh where current is net positive (true discharge, not regen)
    true_disc = disc_mask & (cycles["current_mean"] > 0) & (cycles[dod_col] < 0)
    cycles["capacity_soh"] = np.where(
        true_disc,
        (norm_cap / cycles["ref_capacity_ah"] * 100).clip(0, 100),
        np.nan,
    )
    return cycles


# ── 6. Extract binned discharge sequences ─────────────────────────────────────

def extract_sequences(df: pd.DataFrame, cycles: pd.DataFrame):
    # Filter to true discharge sessions only:
    #   - current_mean > 0: exclude regen sessions (58% of "discharge" rows have
    #     current_mean < 0 due to regen correction mislabelling)
    #   - ecu_fault_suspected == False: exclude sparse ECU-fault sessions
    valid = cycles[
        (cycles["session_type"] == "discharge") &
        (cycles["current_mean"] > 0) &
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
            (cycles["registration_number"] == reg) &
            (cycles["session_id"] == sid)
        ].iloc[0]

        sequences.append(arr)
        entry = {
            "seq_index":           len(sequences) - 1,
            "registration_number": reg,
            "session_id":          sid,
            "cycle_number":        meta_row["cycle_number"],
            "soh":                 meta_row["soh"],
            "capacity_soh":        meta_row.get("capacity_soh", np.nan),
        }
        # Persist session-level scalar health features into seq_meta
        for f in SCALAR_FEATURES:
            entry[f] = meta_row.get(f, np.nan)
        seq_meta.append(entry)

    return sequences, seq_meta


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    gps_df = load_gps(GPS_FILE)
    vcu_df = load_vcu(VCU_FILE)
    df     = load_and_clean(BMS_FILE, gps_df, vcu_df)
    df     = label_sessions(df)

    print("Extracting cycle features ...")
    cycles = extract_cycles(df)
    cycles = add_capacity_soh(cycles)
    cycles = add_fleet_flags(cycles)

    n_disc = (cycles["session_type"] == "discharge").sum()
    n_chg  = (cycles["session_type"] == "charging").sum()
    print(f"  Total sessions : {len(cycles):,}")
    print(f"  Discharge      : {n_disc:,}")
    print(f"  Charging       : {n_chg:,}")
    print(f"\nSoH distribution:\n{cycles['soh'].value_counts().sort_index()}")
    print(f"\nCapacity-SoH: mean={cycles['capacity_soh'].mean():.1f}  "
          f"std={cycles['capacity_soh'].std():.2f}")

    # Sanity check: soc_range is now always abs() so should be 0 everywhere.
    # soc_diff reveals the direction — flag sessions where direction is wrong.
    wrong_disc = ((cycles["session_type"] == "discharge") & (cycles["soc_diff"] > 0)).sum()
    wrong_chg  = ((cycles["session_type"] == "charging")  & (cycles["soc_diff"] < 0)).sum()
    print(f"\nAnomaly check — discharge with SoC gain: {wrong_disc}  "
          f"charging with SoC drop: {wrong_chg}  (non-zero = possible mis-classification)")

    cycles.to_csv(CYCLES_CSV, index=False)
    print(f"\nSaved cycles: {CYCLES_CSV}")

    print("\nExtracting discharge sequences ...")
    sequences, seq_meta = extract_sequences(df, cycles)

    if sequences:
        arr = np.stack(sequences).astype(np.float32)
        np.save(SEQ_NPY, arr)
        pd.DataFrame(seq_meta).to_csv(SEQ_META, index=False)
        print(f"Saved sequences: {SEQ_NPY}  shape={arr.shape}")
        print(f"Saved meta     : {SEQ_META}")
    else:
        print("No sequences extracted.")
