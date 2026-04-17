import functools
import json
import math
import os
import sys

import numpy as np
import pandas as pd
from django.http import JsonResponse, HttpResponse
from django.shortcuts import render
from django.views.decorators.http import require_GET

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "code"))
from config import ARTIFACTS_DIR

RUL_FILE      = os.path.join(ARTIFACTS_DIR, "rul_estimates.csv")
EKF_FILE      = os.path.join(ARTIFACTS_DIR, "ekf_soh.csv")
ANOM_FILE     = os.path.join(ARTIFACTS_DIR, "anomaly_scores.csv")
COEF_FILE     = os.path.join(ARTIFACTS_DIR, "bayes_coefficients.csv")
TELEMETRY_DB  = os.path.join(ARTIFACTS_DIR, "telemetry.db")

ANOM_COLS = [
    "registration_number", "session_id", "start_time_ist", "end_time_ist", "start_time", "session_type",
    "soc_start", "soc_end",
    "soh", "ekf_soh", "if_score", "if_anomaly", "if_reason",
    "cusum_ekf_soh_alarm", "cusum_soh_alarm", "cusum_epk_alarm",
    "cusum_heat_alarm", "cusum_spread_alarm", "cusum_spread_slope_alarm",
    "cusum_cycle_soh_alarm", "cusum_ir_slope_alarm",
    "anomaly_reason", "composite_degradation_score",
    # Core per-session metrics
    "duration_hr", "energy_per_km", "energy_kwh", "energy_per_loaded_session",
    "n_vsag", "n_high_ir", "n_low_soc",
    "ir_ohm_mean", "cell_spread_mean", "cell_spread_max", "temp_rise_rate",
    # Capacity / voltage / current
    "cycle_soh", "ref_capacity_ah",
    "voltage_mean_new", "current_mean_new",
    "capacity_ah_discharge_new", "capacity_ah_charge_new", "capacity_ah_plugin_new",
    "capacity_ah_charge_total_new",   # regen + plugin combined (feeds block SoH for charge blocks)
    # Block-level aggregates
    "block_capacity_ah", "block_odometer_km", "charging_rate_kw",
    # Subsystem health
    "weak_subsystem_consistency", "hot_subsystem_consistency", "subsystem_voltage_std",
    "bms_coverage", "total_alerts", "cell_health_poor", "n_cell_undervoltage", "n_cell_overvoltage",
    # Usage / aging
    "speed_mean", "is_loaded", "odometer_km", "days_since_first",
    "cum_efc", "aging_index", "dod_stress", "thermal_stress", "c_rate_chg",
    "block_soc_diff",
    # EWM-smoothed signals
    "ir_ohm_mean_ewm10", "cell_spread_mean_ewm10", "temp_rise_rate_ewm10",
    "vsag_rate_per_hr", "vsag_rate_per_hr_ewm10", "ir_event_rate",
    # Trend slopes
    "vsag_trend_slope", "ir_event_trend_slope", "ir_ohm_trend_slope",
    "spread_trend_slope", "soh_trend_slope",
    # Binary usage flags
    "rapid_heating", "high_energy_per_km", "slow_charging", "fast_charging",
    # Anomaly flag
    "anomaly",
]

COEF_HIDE = ["efc_x_days", "ir_ohm_mean", "cell_spread_mean",
             "temp_rise_rate", "vsag_rate_per_hr"]

TIER1 = ["MH18BZ3028", "MH18BZ3392", "MH18BZ3341"]
TIER2 = ["MH18BZ2648", "MH18BZ3198", "MH18BZ2689", "MH18BZ2958"]
TIER3 = ["MH18BZ2649", "MH18BZ3163", "MH18BZ3345", "MH18BZ2690", "MH18BZ2647"]

TIER1_SIGNALS = {
    "MH18BZ3028": "Lowest battery health in fleet (94.97%); highest degradation risk score (0.658); 12 flagged sessions (13.2% of total)",
    "MH18BZ3392": "Fastest declining battery in fleet (−0.062%/day); projected replacement in under 1 year; 12 flagged sessions (13.6%)",
    "MH18BZ3341": "3rd highest degradation risk in fleet (0.514); 87 flagged sessions (14.2%); rate of decline −0.029%/day",
}
TIER2_NOTES = {
    "MH18BZ2648": "Highest flagged session count in fleet (124, 17.9%); degradation risk score 0.479",
    "MH18BZ3198": "Fastest declining in tier 2 (−0.044%/day); projected replacement in ~1 year; 43 flagged sessions (8.4%)",
    "MH18BZ2689": "61 flagged sessions (8.1%); degradation risk score 0.473",
    "MH18BZ2958": "36 flagged sessions (6.7%); degradation risk score 0.464",
}
TIER3_NOTES = {
    "MH18BZ2649": "141 flagged sessions (15.8%) but battery health nearly stable (−0.002%/day); long projected lifespan",
    "MH18BZ3163": "52 flagged sessions (8.1%); degradation risk score 0.440; elevated internal resistance",
    "MH18BZ3345": "99 flagged sessions (13.3%); near-flat health trajectory (−0.018%/day)",
    "MH18BZ2690": "84 flagged sessions (11.2%); near-flat health trajectory (−0.012%/day)",
    "MH18BZ2647": "70 flagged sessions (10.6%); stable long-term trajectory",
}


@functools.lru_cache(maxsize=1)
def _load_rul():
    return pd.read_csv(RUL_FILE)


@functools.lru_cache(maxsize=1)
def _load_ekf():
    df = pd.read_csv(EKF_FILE)
    df["date"] = pd.to_datetime(df["start_time"], unit="ms").dt.strftime("%Y-%m-%d")
    return df


_anom_cache = {"mtime": None, "df": None}

def _load_anom():
    mtime = os.path.getmtime(ANOM_FILE)
    if _anom_cache["mtime"] != mtime:
        avail = pd.read_csv(ANOM_FILE, nrows=0).columns.tolist()
        cols = [c for c in ANOM_COLS if c in avail]
        _anom_cache["df"]    = pd.read_csv(ANOM_FILE, usecols=cols, encoding="utf-8")
        _anom_cache["mtime"] = mtime
    return _anom_cache["df"]


@functools.lru_cache(maxsize=1)
def _load_coef():
    if not os.path.exists(COEF_FILE):
        return pd.DataFrame()
    return pd.read_csv(COEF_FILE)


def _df_to_records(df):
    """Convert a DataFrame to a JSON-safe list of dicts (NaN → null)."""
    return json.loads(df.to_json(orient="records"))


def _live_anom_counts():
    """Return {registration_number: n_anomalous} computed live from anomaly_scores.csv."""
    anom = _load_anom()
    if "anomaly" in anom.columns:
        flag = anom["anomaly"].fillna(False).astype(bool)
    else:
        flag = (
            anom.get("if_anomaly",
                     pd.Series(False, index=anom.index)).fillna(False).astype(bool) |
            anom.get("cusum_ekf_soh_alarm",
                     pd.Series(False, index=anom.index)).fillna(False).astype(bool)
        )
    return (
        anom.assign(_flag=flag)
        .groupby("registration_number")["_flag"]
        .sum()
        .astype(int)
        .to_dict()
    )


def _safe_json(payload):
    """Return an HttpResponse with JSON that handles NaN → null."""
    return HttpResponse(
        json.dumps(payload, allow_nan=False,
                   default=lambda v: None if (isinstance(v, float) and math.isnan(v)) else v),
        content_type="application/json",
    )


def dashboard_page(request):
    return render(request, "fleet/dashboard.html")


def executive_summary_page(request):
    return render(request, "fleet/executive_summary.html")


@require_GET
def api_overview(request):
    rul = _load_rul()
    ekf = _load_ekf()
    anom = _load_anom()

    last_ekf = ekf.groupby("registration_number")["ekf_soh"].last()
    last_rul_days = ekf.groupby("registration_number")["ekf_rul_days"].last() \
        if "ekf_rul_days" in ekf.columns else pd.Series(dtype=float)

    # Fleet-wide SoH trend: mean of first vs last EKF SoH per vehicle
    ekf_sorted = ekf.sort_values("start_time")
    first_soh = ekf_sorted.groupby("registration_number")["ekf_soh"].first().mean()
    last_soh  = ekf_sorted.groupby("registration_number")["ekf_soh"].last().mean()
    soh_trend_pct = round(float(last_soh - first_soh), 2)

    # Session counts
    total_sessions    = len(anom)
    charging_sessions  = int((anom["session_type"] == "charging").sum())  if "session_type" in anom.columns else None
    discharge_sessions = int((anom["session_type"] == "discharge").sum()) if "session_type" in anom.columns else None

    # EKF RUL interquartile range across vehicles
    ekf_rul_p25 = ekf_rul_p75 = None
    if last_rul_days.notna().any():
        rul_vals = last_rul_days.dropna()
        ekf_rul_p25 = round(float(rul_vals.quantile(0.25)), 0)
        ekf_rul_p75 = round(float(rul_vals.quantile(0.75)), 0)

    # Remaining EFC: estimate per vehicle = (daily EFC rate) × EKF RUL days
    remaining_efc_per_veh = []
    if all(c in anom.columns for c in ["cum_efc", "days_since_first", "registration_number"]) \
            and "start_time" in anom.columns:
        latest_sess = (anom.sort_values("start_time")
                       .groupby("registration_number")[["cum_efc", "days_since_first"]]
                       .last())
        for reg, row in latest_sess.iterrows():
            rul_d = last_rul_days.get(reg)
            if rul_d is not None and not math.isnan(float(rul_d)) and float(row["days_since_first"]) > 0:
                daily_rate = float(row["cum_efc"]) / float(row["days_since_first"])
                remaining_efc_per_veh.append(round(daily_rate * float(rul_d), 1))
    fleet_median_remaining_efc = (round(float(np.median(remaining_efc_per_veh)), 1)
                                  if remaining_efc_per_veh else None)

    return JsonResponse({
        "n_vehicles":      int(rul["registration_number"].nunique()),
        "first_date":      str(pd.to_datetime(rul["first_date"].min()).date()),
        "last_date":       str(pd.to_datetime(rul["last_date"].max()).date()),
        "span_days":       int((pd.to_datetime(rul["last_date"].max()) -
                                pd.to_datetime(rul["first_date"].min())).days),
        "fleet_mean_soh":  round(float(last_ekf.mean()), 3),
        "fleet_std_soh":   round(float(last_ekf.std()), 3),
        "median_rul_days": round(float(rul["rul_days"].median()), 0)
                           if rul["rul_days"].notna().any() else None,
        "median_ekf_rul":  round(float(last_rul_days.median()), 0)
                           if last_rul_days.notna().any() else None,
        "ekf_rul_p25":     ekf_rul_p25,
        "ekf_rul_p75":     ekf_rul_p75,
        "soh_trend_pct":   soh_trend_pct,
        "first_soh":       round(float(first_soh), 3),
        "last_soh":        round(float(last_soh), 3),
        "cycle_soh_obs_n":   1514,
        "cycle_soh_obs_pct": 19.5,
        "cycle_soh_total":   7745,
        "eol_threshold":     80.0,
        "total_sessions":    total_sessions,
        "charging_sessions": charging_sessions,
        "discharge_sessions": discharge_sessions,
        "fleet_median_remaining_efc": fleet_median_remaining_efc,
        "remaining_efc_per_veh":      remaining_efc_per_veh,
    })


@require_GET
def api_fleet_trend(request):
    ekf = _load_ekf()
    daily = (
        ekf.groupby("date")["ekf_soh"]
        .median()
        .reset_index()
        .rename(columns={"ekf_soh": "median_soh"})
    )
    total_vehicles = ekf["registration_number"].nunique()
    coverage = (
        ekf.groupby("date")["registration_number"]
        .nunique()
        .reset_index()
        .rename(columns={"registration_number": "vehicle_count"})
    )
    coverage["pct"] = (coverage["vehicle_count"] / total_vehicles * 100).round(1)
    daily = daily.merge(coverage, on="date", how="left")
    return JsonResponse({"trend": daily.to_dict(orient="records"), "total_vehicles": total_vehicles})


@require_GET
def api_quintiles(request):
    ekf = _load_ekf()
    ekf = ekf.copy()
    ekf["q"], bins = pd.qcut(
        ekf["days_since_first_session"], q=5,
        labels=False, retbins=True, duplicates="drop"
    )
    labels = [f"Q{i+1} ({bins[i]:.0f}-{bins[i+1]:.0f}d)"
              for i in range(len(bins) - 1)]
    ekf["quintile"] = ekf["q"].map(dict(enumerate(labels)))
    qt = (
        ekf.groupby("quintile", sort=False)["ekf_soh"]
        .median()
        .reset_index()
        .rename(columns={"ekf_soh": "median_soh"})
    )
    qt["_q"] = qt["quintile"].str.extract(r"Q(\d)").astype(int)
    qt = qt.sort_values("_q").drop(columns=["_q"])
    return JsonResponse({"quintiles": qt.to_dict(orient="records")})


@require_GET
def api_vehicles(request):
    rul = _load_rul()
    COLS = [
        "registration_number", "current_soh", "rul_days", "rul_reliability",
        "dual_dominant_path", "bayes_soh_pred", "bayes_soh_std",
        "composite_degradation_score", "n_combined_anom", "if_score_mean",
        "soh_slope_%per_day", "first_date", "last_date",
    ]
    avail = [c for c in COLS if c in rul.columns]
    df = rul[avail].copy()
    # Override stale rul.csv count with live count from anomaly_scores.csv
    live = _live_anom_counts()
    df["n_combined_anom"] = df["registration_number"].map(live).fillna(0).astype(int)
    df = df.sort_values("composite_degradation_score", ascending=False)

    # Attach EKF RUL (last value per vehicle from ekf_soh.csv) so the
    # histogram on the executive summary uses the same model as the KPI card
    ekf = _load_ekf()
    if "ekf_rul_days" in ekf.columns:
        last_ekf_rul = (ekf.groupby("registration_number")["ekf_rul_days"]
                        .last()
                        .rename("ekf_rul_days"))
        df = df.merge(last_ekf_rul, on="registration_number", how="left")

    return _safe_json({"vehicles": _df_to_records(df)})


@require_GET
def api_bayes_coef(request, reg=None):
    coef = _load_coef()
    if coef.empty:
        return JsonResponse(
            {"error": "bayes_coefficients.csv not found. Run soh_rul.py first."},
            status=404,
        )
    feat_cols = [c for c in coef.columns
                 if c != "registration_number" and c not in COEF_HIDE]
    global_coef = coef[feat_cols].median().dropna()

    veh_coef = {}
    if reg:
        row = coef[coef["registration_number"] == reg]
        if not row.empty:
            veh_coef = row[feat_cols].iloc[0].dropna().to_dict()

    return JsonResponse({
        "global": {k: round(float(v), 6) for k, v in global_coef.items()},
        "vehicle": {k: round(float(v), 6) for k, v in veh_coef.items()},
        "registration_number": reg,
    })


@require_GET
def api_soh_scatter(request):
    """Per-session BMS-reported SoH vs EKF SoH for the scatter plot."""
    anom = _load_anom()
    cols = ["registration_number", "soh", "ekf_soh", "start_time_ist"]
    avail = [c for c in cols if c in anom.columns]
    df = anom[avail].dropna(subset=["soh", "ekf_soh"])
    if len(df) > 3000:
        df = df.sample(3000, random_state=42)
    return _safe_json({"points": _df_to_records(df)})


@require_GET
def api_soh_delta_trend(request):
    """Daily (EKF SoH − BMS SoH) delta per vehicle for the delta time-series chart."""
    anom = _load_anom()
    cols = ["registration_number", "soh", "ekf_soh", "start_time_ist"]
    avail = [c for c in cols if c in anom.columns]
    df = anom[avail].dropna(subset=["soh", "ekf_soh"]).copy()
    df["date"] = pd.to_datetime(df["start_time_ist"], errors="coerce").dt.strftime("%Y-%m-%d")
    df["delta"] = (df["ekf_soh"] - df["soh"]).round(3)
    # Fleet daily median delta
    fleet = (
        df.groupby("date")["delta"]
        .median()
        .reset_index()
        .rename(columns={"delta": "fleet_median_delta"})
        .sort_values("date")
    )
    # Per-vehicle records (date + reg + delta) for slider mode
    veh_records = _df_to_records(
        df[["registration_number", "date", "delta"]].sort_values(["registration_number", "date"])
    )
    return _safe_json({
        "fleet_trend": fleet.to_dict(orient="records"),
        "vehicle_points": veh_records,
    })


@require_GET
def api_anomaly_tiers(request):
    rul = _load_rul()
    ekf = _load_ekf()
    anom = _load_anom()
    cusum = anom.groupby("registration_number").agg(
        cusum_ekf=("cusum_ekf_soh_alarm", "sum"),
        cusum_bms=("cusum_soh_alarm", "sum"),
    ).reset_index()
    live = _live_anom_counts()

    # Latest EKF RUL per vehicle
    ekf_rul_map = {}
    if "ekf_rul_days" in ekf.columns:
        ekf_rul_map = (
            ekf.groupby("registration_number")["ekf_rul_days"]
            .last()
            .dropna()
            .to_dict()
        )

    def veh_row(reg):
        r = rul[rul["registration_number"] == reg]
        if r.empty:
            return {"registration_number": reg}
        r = r.iloc[0]
        ekf_rul = ekf_rul_map.get(reg)
        return {
            "registration_number": reg,
            "current_soh":     round(float(r.get("current_soh", 0) or 0), 2),
            "soh_slope":       round(float(r.get("soh_slope_%per_day", 0) or 0), 5),
            "composite":       round(float(r.get("composite_degradation_score", 0) or 0), 4),
            "n_combined_anom": live.get(reg, 0),
            "rul_days":        round(float(ekf_rul), 0) if ekf_rul is not None and not math.isnan(float(ekf_rul)) else None,
        }

    tier1 = []
    for v in TIER1:
        row = veh_row(v)
        row["primary_signal"] = TIER1_SIGNALS.get(v, "")
        tier1.append(row)

    tier2 = []
    for v in TIER2:
        row = veh_row(v)
        row["note"] = TIER2_NOTES.get(v, "")
        tier2.append(row)

    tier3 = []
    for v in TIER3:
        row = veh_row(v)
        row["note"] = TIER3_NOTES.get(v, "")
        tier3.append(row)

    return JsonResponse({"tier1": tier1, "tier2": tier2, "tier3": tier3})


@require_GET
def api_sessions(request, reg):
    anom = _load_anom()
    veh = anom[anom["registration_number"] == reg].copy()
    if "start_time" in veh.columns:
        veh = veh.sort_values("start_time", ascending=False)
    if "anomaly" in veh.columns:
        veh["is_anomalous"] = veh["anomaly"].fillna(False).astype(bool)
    else:
        veh["is_anomalous"] = (
            veh.get("if_anomaly",
                    pd.Series(False, index=veh.index)).fillna(False).astype(bool) |
            veh.get("cusum_ekf_soh_alarm",
                    pd.Series(False, index=veh.index)).fillna(False).astype(bool)
        )
    # Composite CUSUM flag: True if ANY cusum alarm fired
    _cusum_cols = ["cusum_ekf_soh_alarm", "cusum_soh_alarm", "cusum_cycle_soh_alarm",
                   "cusum_heat_alarm", "cusum_spread_alarm", "cusum_spread_slope_alarm",
                   "cusum_epk_alarm", "cusum_ir_slope_alarm"]
    _avail_cusum = [c for c in _cusum_cols if c in veh.columns]
    if _avail_cusum:
        veh["cusum_anomaly"] = veh[_avail_cusum].fillna(False).astype(bool).any(axis=1)
    else:
        veh["cusum_anomaly"] = False
    # Display fields (shown in table) + filter fields (used for client-side filtering)
    SHOW = [
        # Identity / time
        "session_id", "start_time_ist", "end_time_ist", "registration_number", "session_type",
        "soc_start", "soc_end",
        # EKF & anomaly flags
        "soh", "ekf_soh", "duration_hr", "if_score", "if_anomaly", "if_reason",
        "cusum_anomaly", "cusum_soh_alarm", "cusum_cycle_soh_alarm",
        "cusum_heat_alarm", "cusum_spread_alarm", "cusum_spread_slope_alarm",
        "cusum_epk_alarm", "cusum_ir_slope_alarm",
        "composite_degradation_score", "anomaly_reason", "is_anomalous",
        # Core metrics
        "n_vsag", "ir_ohm_mean", "cell_spread_mean", "energy_per_km",
        "energy_kwh", "n_low_soc",
        "ref_capacity_ah", "voltage_mean_new", "current_mean_new",
        "capacity_ah_discharge_new", "capacity_ah_charge_new", "capacity_ah_plugin_new",
        "capacity_ah_charge_total_new",
        "cycle_soh", "block_capacity_ah", "odometer_km", "block_odometer_km", "charging_rate_kw",
        "cell_spread_max", "weak_subsystem_consistency", "hot_subsystem_consistency",
        "subsystem_voltage_std", "temp_rise_rate", "bms_coverage",
        "speed_mean", "is_loaded", "cum_efc", "days_since_first", "aging_index",
        "block_soc_diff",
        # EWM & trends
        "vsag_rate_per_hr", "ir_event_rate",
        "ir_ohm_mean_ewm10", "cell_spread_mean_ewm10", "temp_rise_rate_ewm10",
        "vsag_rate_per_hr_ewm10",
        "vsag_trend_slope", "ir_event_trend_slope", "ir_ohm_trend_slope",
        "spread_trend_slope", "soh_trend_slope",
        "c_rate_chg", "dod_stress", "thermal_stress",
        "energy_per_loaded_session", "total_alerts", "cell_health_poor",
        "n_cell_undervoltage", "n_cell_overvoltage",
        "rapid_heating", "high_energy_per_km", "slow_charging", "fast_charging",
        # Filter-only fields (not displayed but needed client-side)
        "n_high_ir",
    ]
    # Compute soc_diff = soc_start − soc_end (positive = battery discharged)
    if "soc_start" in veh.columns and "soc_end" in veh.columns:
        veh["soc_diff"] = (veh["soc_start"] - veh["soc_end"]).round(1)

    total = len(veh)
    # Keep only anomalous sessions for display (no cap — show all)
    anom_veh = veh[veh["is_anomalous"]]
    show_cols = SHOW + ["soc_diff"]
    show = [c for c in show_cols if c in anom_veh.columns]
    return _safe_json({
        "registration_number": reg,
        "total_sessions": total,
        "total_anomalous": int(veh["is_anomalous"].sum()),
        "sessions": _df_to_records(anom_veh[show]),
    })


@require_GET
def api_anomaly_breakdown(request, reg=None):
    anom = _load_anom()

    # Filter to a single vehicle if requested
    if reg:
        anom = anom[anom["registration_number"] == reg]

    # Only rows that are anomalous
    if "anomaly" in anom.columns:
        anom_only = anom[anom["anomaly"].fillna(False).astype(bool)]
    else:
        anom_only = anom

    # Optional: filter by session type ("charging" or "discharge")
    session_type = request.GET.get("session_type", None)
    if session_type and "session_type" in anom_only.columns:
        anom_only = anom_only[anom_only["session_type"] == session_type]

    # Optional: further filter by detector ("if" or "cusum")
    detector = request.GET.get("detector", None)
    cusum_cols = [
        "cusum_ekf_soh_alarm", "cusum_soh_alarm", "cusum_cycle_soh_alarm",
        "cusum_heat_alarm", "cusum_spread_alarm", "cusum_spread_slope_alarm",
        "cusum_epk_alarm", "cusum_ir_slope_alarm",
    ]
    if detector == "if" and "if_anomaly" in anom_only.columns:
        anom_only = anom_only[anom_only["if_anomaly"].fillna(False).astype(bool)]
    elif detector == "cusum":
        avail = [c for c in cusum_cols if c in anom_only.columns]
        if avail:
            anom_only = anom_only[
                anom_only[avail].fillna(False).astype(bool).any(axis=1)
            ]

    def cnt(col):
        """Count anomalous rows where column is truthy."""
        if col not in anom_only.columns:
            return 0
        return int(anom_only[col].fillna(0).astype(bool).sum())

    def cnt_any(*cols):
        """Count rows where ANY of the given columns is truthy (OR logic)."""
        masks = [anom_only[c].fillna(0).astype(bool)
                 for c in cols if c in anom_only.columns]
        if not masks:
            return 0
        combined = masks[0]
        for m in masks[1:]:
            combined = combined | m
        return int(combined.sum())

    # By detector: IF vs any CUSUM
    avail_cusum = [c for c in cusum_cols if c in anom_only.columns]
    cusum_any = int(
        anom_only[avail_cusum].fillna(False).astype(bool).any(axis=1).sum()
    ) if avail_cusum else 0

    by_detector = {
        "Isolation Forest": cnt("if_anomaly"),
        "CUSUM": cusum_any,
    }

    # By physical signal — IF uses if_reason text; CUSUM uses alarm columns.
    if detector == "if" and "if_reason" in anom_only.columns:
        reasons = anom_only["if_reason"].fillna("")
        IF_MAP = [
            ("IR Degradation",          ["n_high_ir", "ir_ohm_mean", "d_n_high_ir", "ir_event_rate", "d_ir_ohm"]),
            ("Voltage Sag",             ["n_vsag", "d_vsag_per_cycle"]),
            ("Cell Spread / Imbalance", ["cell_spread", "n_cell_spread_warn", "subsystem_voltage_std"]),
            ("Thermal Stress",          ["temp_lowest_mean", "temp_max", "temp_rise_rate", "thermal_stress"]),
            ("Efficiency / Capacity",   ["energy_per_loaded_session", "capacity_ah_discharge"]),
            ("High DoD",                ["dod_stress"]),
            ("Low SoC / Undervoltage",  ["n_low_soc", "voltage_min"]),
            ("SoH Decline",             ["capacity_soh_disc_new", "soh_smooth", "ekf_soh_delta", "cycle_soh"]),
            ("Usage Pattern",           ["odometer_km", "duration_hr"]),
        ]
        by_signal = {}
        for label, keywords in IF_MAP:
            pattern = "|".join(keywords)
            n = int(reasons.str.contains(pattern, case=False, na=False).sum())
            if n > 0:
                by_signal[label] = n
    else:
        # CUSUM-based breakdown — each category uses OR logic, counts per session
        by_signal = {k: v for k, v in {
            "EKF SoH Decline": cnt("cusum_ekf_soh_alarm"),
            "BMS SoH Decline": cnt("cusum_soh_alarm"),
            "Cycle SoH Drop":  cnt("cusum_cycle_soh_alarm"),
            "IR Degradation":  cnt_any("cusum_ir_slope_alarm", "n_high_ir"),
            "Cell Spread":     cnt_any("cusum_spread_alarm", "cusum_spread_slope_alarm"),
            "Thermal Stress":  cnt("cusum_heat_alarm"),
            "Efficiency Loss": cnt("cusum_epk_alarm"),
            "Voltage Sag":     cnt("n_vsag"),
        }.items() if v > 0}

    return JsonResponse({"by_detector": by_detector, "by_signal": by_signal})


@require_GET
def api_soh_bands(request, reg):
    ekf = _load_ekf()
    df = ekf[ekf["registration_number"] == reg].sort_values("start_time").copy()
    if df.empty:
        return JsonResponse({"error": f"No EKF data for {reg}"}, status=404)

    df["upper"] = (df["ekf_soh"] + 2 * df["ekf_soh_std"]).round(3)
    df["lower"] = (df["ekf_soh"] - 2 * df["ekf_soh_std"]).round(3)

    cols = ["date", "ekf_soh", "upper", "lower"]
    if "bms_soh_obs" in df.columns:
        cols.append("bms_soh_obs")

    # Discharge sessions with forward-filled EKF for full timeline view
    anom = _load_anom()
    disc_ekf = []
    if "session_type" in anom.columns and "start_time_ist" in anom.columns and "ekf_soh" in anom.columns:
        disc = (
            anom[(anom["registration_number"] == reg) & (anom["session_type"] == "discharge")]
            [["start_time_ist", "ekf_soh"]]
            .rename(columns={"start_time_ist": "date"})
        )
        disc_ekf = _df_to_records(disc)

    return _safe_json({"reg": reg, "bands": _df_to_records(df[cols]), "discharge_ekf": disc_ekf})


@require_GET
def api_telemetry(request, reg, session_id):
    """Return unaggregated row-level telemetry for one session from SQLite."""
    import sqlite3 as _sqlite3
    if not os.path.exists(TELEMETRY_DB):
        return JsonResponse({"error": "Telemetry DB not found — run data_prep_1.py first."}, status=404)
    try:
        con = _sqlite3.connect(TELEMETRY_DB)
        query = (
            "SELECT * FROM telemetry "
            "WHERE registration_number = ? AND session_id = ? "
            "ORDER BY gps_time"
        )
        df = pd.read_sql_query(query, con, params=(reg, session_id))
        con.close()
    except Exception as exc:
        return JsonResponse({"error": str(exc)}, status=500)

    if df.empty:
        return JsonResponse({"rows": [], "session_id": session_id, "reg": reg})

    # Convert gps_time (ms epoch) to ISO strings for the frontend
    if "gps_time" in df.columns:
        df["ts"] = pd.to_datetime(df["gps_time"], unit="ms").dt.strftime("%Y-%m-%dT%H:%M:%S")

    return _safe_json({
        "reg": reg,
        "session_id": session_id,
        "session_type": str(df["session_type"].iloc[0]) if "session_type" in df.columns else None,
        "rows": _df_to_records(df),
    })


# ── RUL Timeline (per-vehicle, charging sessions) ─────────────────────────────
@require_GET
def api_rul_timeline(request, reg):
    ekf = _load_ekf()
    df  = ekf[ekf["registration_number"] == reg].sort_values("start_time").copy()
    if df.empty:
        return JsonResponse({"error": f"No EKF data for {reg}"}, status=404)

    cols = ["date", "ekf_rul_days"]
    if "ekf_rul_days_lo" in df.columns: cols.append("ekf_rul_days_lo")
    if "ekf_rul_days_hi" in df.columns: cols.append("ekf_rul_days_hi")
    if "ekf_soh"         in df.columns: cols.append("ekf_soh")

    rows = _df_to_records(df[cols])
    return _safe_json({"reg": reg, "points": rows})


# ── Breakdown Timeline ─────────────────────────────────────────────────────────
@require_GET
def api_breakdown_timeline(request):
    rul  = _load_rul()
    ekf  = _load_ekf()
    EOL  = 80.0  # end-of-life SoH threshold

    # Reference date = last session date in EKF data
    ref_ts = pd.to_datetime(ekf["start_time"], unit="ms").max()
    ref_date_str = ref_ts.strftime("%Y-%m-%d")

    # Latest EKF SoH per vehicle (for display)
    last_ekf = (
        ekf.sort_values("start_time")
           .groupby("registration_number")[["ekf_soh", "ekf_soh_std"]]
           .last()
    )

    # Data span per vehicle: first → last session date
    ekf_ts = pd.to_datetime(ekf["start_time"], unit="ms")
    span_df = ekf.copy()
    span_df["_ts"] = ekf_ts
    span_grp = span_df.groupby("registration_number")["_ts"].agg(["min", "max"])
    span_grp["span_days"] = (span_grp["max"] - span_grp["min"]).dt.days
    span_grp["first_date"] = span_grp["min"].dt.strftime("%Y-%m-%d")
    span_grp["last_date"]  = span_grp["max"].dt.strftime("%Y-%m-%d")

    def _add_days(base_ts, days):
        if days is None or not math.isfinite(float(days)):
            return None
        return (base_ts + pd.Timedelta(days=float(days))).strftime("%Y-%m-%d")

    rows = []
    for _, r in rul.iterrows():
        reg        = r["registration_number"]
        curr_soh   = r.get("current_soh")
        slope      = r.get("soh_slope_%per_day")
        composite  = r.get("composite_degradation_score")

        # Slope-based RUL — same formula used in the tier table JS (renderAnomalyTiers)
        rul_days = None
        if (curr_soh is not None and pd.notna(curr_soh) and
                slope is not None and pd.notna(slope) and float(slope) < 0):
            rul_days = max(0.0, (float(curr_soh) - EOL) / abs(float(slope)))

        ekf_row   = last_ekf.loc[reg] if reg in last_ekf.index else None
        ekf_soh   = float(ekf_row["ekf_soh"])   if ekf_row is not None and pd.notna(ekf_row["ekf_soh"])   else None
        ekf_std   = float(ekf_row["ekf_soh_std"]) if ekf_row is not None and pd.notna(ekf_row["ekf_soh_std"]) else None

        tier = 1 if reg in TIER1 else (2 if reg in TIER2 else (3 if reg in TIER3 else 0))

        span_row = span_grp.loc[reg] if reg in span_grp.index else None
        data_span = {
            "first": span_row["first_date"],
            "last":  span_row["last_date"],
            "days":  int(span_row["span_days"]),
        } if span_row is not None else None

        rows.append({
            "registration_number": reg,
            "current_soh": round(float(curr_soh), 2) if curr_soh is not None and pd.notna(curr_soh) else None,
            "soh_slope":   round(float(slope), 5)    if slope    is not None and pd.notna(slope)    else None,
            "ekf_soh":     round(ekf_soh, 2)   if ekf_soh  is not None else None,
            "ekf_soh_std": round(ekf_std, 4)   if ekf_std  is not None else None,
            "composite":   round(float(composite), 4) if composite is not None and pd.notna(composite) else None,
            "rul_days":    round(rul_days, 0)   if rul_days is not None else None,
            "eol_date":    _add_days(ref_ts, rul_days),
            "tier":        tier,
            "ref_date":    ref_date_str,
            "data_span":   data_span,
        })

    # Sort soonest-first (None / no-slope vehicles last)
    rows.sort(key=lambda x: (x["rul_days"] is None, x["rul_days"] or float("inf")))

    return _safe_json({"timeline": rows, "ref_date": ref_date_str})


# ── Data Distributions ─────────────────────────────────────────────────────────
@require_GET
def api_distributions(request):
    anom = _load_anom()

    csoh_vals      = []
    block_soh_vals = []

    if "cycle_soh" in anom.columns:
        # Per-session quality gate: block DoD >= 20%, exclude ceiling artefact
        mask = anom["cycle_soh"].notna() & (anom["cycle_soh"] < 99.5)
        if "block_soc_diff" in anom.columns:
            mask = mask & (anom["block_soc_diff"].abs() >= 20)
        csoh_vals = anom.loc[mask, "cycle_soh"].round(3).tolist()

        # Block SoH: deduplicate to one value per discharge block
        # Use (registration_number, block_capacity_ah) as a proxy block key —
        # different blocks have different total Ah, so this gives ~one value per block.
        dedup_cols = ["registration_number"]
        if "block_capacity_ah" in anom.columns:
            dedup_cols.append("block_capacity_ah")
        block_soh_vals = (
            anom[mask]
            .drop_duplicates(subset=dedup_cols)
            ["cycle_soh"]
            .round(3)
            .tolist()
        )

    return _safe_json({
        "cycle_soh":      csoh_vals,
        "block_soh":      block_soh_vals,
        "cycle_soh_n":    len(csoh_vals),
        "block_soh_n":    len(block_soh_vals),
    })
