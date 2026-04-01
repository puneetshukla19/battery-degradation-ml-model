import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) if "__file__" in dir() else os.getcwd())

"""
soh_rul.py — Per-vehicle SoH trend analysis, RUL estimation, and composite
             degradation scoring.

For each vehicle:
  1. Smooth BMS SoH (rolling median removes integer-step noise).
  2. Fit linear degradation: SoH(t) = a*t + b  (t = days since first record).
  3. Fit linear trends for energy_per_km and temp_rise_rate (secondary signals).
  4. Estimate RUL = days until SoH hits EOL_SOH (default 80%).
  5. Compute a composite degradation rank combining all three signal slopes.

Outputs
-------
soh_trends.csv      Smoothed SoH trajectory per vehicle (one row per cycle)
rul_estimates.csv   One row per vehicle: slopes, RUL, CI, composite rank
"""

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import BayesianRidge
from config import (CYCLES_CSV, EOL_SOH, ARTIFACTS_DIR, EFC_MAX, CAL_AGING_RATE,
                    MIN_SOC_RANGE_FOR_TREND, MIN_UNIQUE_SOH_FOR_OLS, OLS_R2_THRESHOLD,
                    EKF_CSV)
import os

RUL_FILE     = os.path.join(ARTIFACTS_DIR, "rul_estimates.csv")
TREND_FILE   = os.path.join(ARTIFACTS_DIR, "soh_trends.csv")
ANOMALY_FILE = os.path.join(ARTIFACTS_DIR, "anomaly_scores.csv")
NEURAL_FILE  = os.path.join(ARTIFACTS_DIR, "neural_predictions.csv")

MIN_CYCLES_FOR_FIT = 5
SOH_SMOOTH_WINDOW  = 5
BOOTSTRAP_N        = 200

# Weights for composite degradation score (all normalised to [0,1] first)
# Higher weight = signal contributes more to the overall rank
# Primary SoH signal: ekf_soh_norm (EKF-filtered, physics-grounded) is used when
# ekf_soh.csv is available; falls back to soh_slope_norm (OLS) if not.
COMPOSITE_WEIGHTS = {
    "soh_health_norm":        0.25,  # primary: EKF SoH health deficit (or OLS slope fallback)
    "cycle_soh_slope_norm":   0.10,  # per-cycle SoH trend (all session types; more data points)
    "vsag_slope_norm":        0.15,  # rising severe voltage sags = capacity loss
    "ir_slope_norm":          0.15,  # rising high-IR events = internal resistance growth
    "energy_slope_norm":      0.13,  # rising energy-per-km = efficiency loss
    "heat_slope_norm":        0.11,  # rising temp-rise-rate = thermal degradation
    "spread_slope_norm":      0.11,  # rising cell-voltage spread = growing imbalance
}


def fit_degradation(days: np.ndarray, signal: np.ndarray) -> dict:
    """OLS linear fit. Returns slope, intercept, R², p-value."""
    result = stats.linregress(days, signal)
    return {
        "slope":     result.slope,
        "intercept": result.intercept,
        "r2":        result.rvalue ** 2,
        "p_value":   result.pvalue,
    }


def rul_from_fit(current_soh: float, slope: float, eol: float = EOL_SOH) -> float:
    if slope >= 0 or current_soh <= eol:
        return np.inf
    return (current_soh - eol) / abs(slope)


def bootstrap_rul(days: np.ndarray, soh: np.ndarray,
                  current_soh: float, n: int = BOOTSTRAP_N) -> tuple[float, float]:
    ruls, idx = [], np.arange(len(days))
    for _ in range(n):
        sample = np.random.choice(idx, size=len(idx), replace=True)
        if len(np.unique(days[sample])) < 2:
            continue
        ruls.append(rul_from_fit(current_soh, fit_degradation(days[sample], soh[sample])["slope"]))
    ruls = [r for r in ruls if np.isfinite(r)]
    if not ruls:
        return np.inf, np.inf
    return float(np.percentile(ruls, 5)), float(np.percentile(ruls, 95))


def bootstrap_rul_efc(efc_cum: np.ndarray, soh: np.ndarray,
                      current_soh: float, n: int = BOOTSTRAP_N) -> tuple:
    ruls, idx = [], np.arange(len(efc_cum))
    for _ in range(n):
        s = np.sort(np.random.choice(idx, size=len(idx), replace=True))
        if len(np.unique(efc_cum[s])) < 2:
            continue
        ruls.append(rul_from_fit(current_soh, fit_degradation(efc_cum[s], soh[s])["slope"]))
    finite = [r for r in ruls if np.isfinite(r)]
    if not finite:
        return None, None
    return float(np.percentile(finite, 5)), float(np.percentile(finite, 95))


def _finite(v):
    return round(v, 0) if np.isfinite(v) else None


# ── Dual-axis degradation model (EFC + Calendar) ──────────────────────────────

def fit_dual_axis(efc: np.ndarray, days: np.ndarray,
                  soh: np.ndarray) -> dict:
    """
    OLS fit of:  SoH = SoH₀ - α·(efc/EFC_MAX) - β·(days/365)·CAL_AGING_RATE - γ·(efc·days)

    Returns dict with fitted coefficients and current SoH₀ estimate.
    Falls back gracefully if the design matrix is rank-deficient.
    """
    n = len(soh)
    if n < 5:
        return {"soh0": float(soh[0]) if n else np.nan,
                "alpha": np.nan, "beta": np.nan, "gamma": np.nan}

    X = np.column_stack([
        efc / EFC_MAX,                                 # cycle aging term
        (days / 365.0) * CAL_AGING_RATE,               # calendar aging term
        (efc * days) / (EFC_MAX * 365.0),              # interaction term
        np.ones(n),                                    # intercept = SoH₀
    ])
    try:
        coeffs, _, _, _ = np.linalg.lstsq(X, soh, rcond=None)
        # coeffs = [-alpha, -beta, -gamma, SoH0]
        return {
            "soh0":  float(coeffs[3]),
            "alpha": float(-coeffs[0]),   # positive = cycle fade
            "beta":  float(-coeffs[1]),   # positive = calendar fade
            "gamma": float(-coeffs[2]),   # positive = interaction fade
        }
    except Exception:
        return {"soh0": float(soh[0]), "alpha": np.nan, "beta": np.nan, "gamma": np.nan}


def rul_dual_axis(fit: dict, current_soh: float,
                  daily_efc_rate: float, eol: float = EOL_SOH) -> dict:
    """
    Estimate RUL from dual-axis model.
    Returns days until SoH hits EOL via EFC path, calendar path, and minimum.
    """
    result = {"rul_efc_days": np.inf, "rul_cal_days": np.inf, "rul_days": np.inf}
    if current_soh <= eol:
        return {k: 0.0 for k in result}

    remaining_soh = current_soh - eol

    # EFC path: α·(efc/EFC_MAX) drives remaining degradation
    if np.isfinite(fit["alpha"]) and fit["alpha"] > 0 and daily_efc_rate > 0:
        remaining_efc = remaining_soh / (fit["alpha"] / EFC_MAX)
        result["rul_efc_days"] = remaining_efc / daily_efc_rate

    # Calendar path: β·(days/365)·CAL_AGING_RATE
    cal_rate_per_day = fit["beta"] * CAL_AGING_RATE / 365.0
    if np.isfinite(cal_rate_per_day) and cal_rate_per_day > 0:
        result["rul_cal_days"] = remaining_soh / cal_rate_per_day

    result["rul_days"] = min(
        result["rul_efc_days"],
        result["rul_cal_days"],
    )
    return result


# ── Bayesian Ridge per vehicle ─────────────────────────────────────────────────

def bayesian_rul_vehicle(efc: np.ndarray, days: np.ndarray,
                         soh: np.ndarray,
                         eol: float = EOL_SOH,
                         daily_efc_rate: float = np.nan,
                         extra_features: np.ndarray | None = None) -> dict:
    """
    Fit BayesianRidge on [cum_efc, days_since_first, efc*days, ...extra] → capacity_soh.
    extra_features: optional (n, k) array of additional per-session features
                    (e.g. cycle_soh, ir_ohm_mean, cell_spread_mean, temp_mean).
    Returns point-estimate SOH, 1-sigma uncertainty, and RUL estimate.
    """
    n = len(soh)
    empty = {"bayes_soh_pred": np.nan, "bayes_soh_std": np.nan,
             "bayes_rul_days": np.nan}
    if n < 5:
        return empty

    base = np.column_stack([efc, days, efc * days])
    if extra_features is not None and extra_features.shape[0] == n:
        # Replace NaN in extra features with column medians before stacking
        ef = extra_features.copy().astype(float)
        col_medians = np.nanmedian(ef, axis=0)
        for c in range(ef.shape[1]):
            ef[np.isnan(ef[:, c]), c] = col_medians[c]
        X = np.column_stack([base, ef])
    else:
        X = base
    model = BayesianRidge(max_iter=500)
    try:
        model.fit(X, soh)
    except Exception:
        return empty

    # Predict at the latest observation point
    X_last = X[[-1]]
    soh_pred, soh_std = model.predict(X_last, return_std=True)
    soh_pred, soh_std = float(soh_pred[0]), float(soh_std[0])

    # Project forward to EOL using current linear trend
    if soh_pred <= eol:
        bayes_rul = 0.0
    elif np.isfinite(daily_efc_rate) and daily_efc_rate > 0:
        # Approximate gradient: d(SoH)/d(day) via finite differences
        efc_last, day_last = float(efc[-1]), float(days[-1])
        delta = 1.0   # 1 day ahead
        base_next = [efc_last + daily_efc_rate * delta,
                     day_last + delta,
                     (efc_last + daily_efc_rate * delta) * (day_last + delta)]
        if extra_features is not None and extra_features.shape[0] == n:
            # Forward-project extra features by repeating last row
            X_next = np.array([base_next + list(X[-1, 3:])])
        else:
            X_next = np.array([base_next])
        soh_next = float(model.predict(X_next)[0])
        daily_soh_change = soh_next - soh_pred
        if daily_soh_change < 0:
            bayes_rul = (soh_pred - eol) / abs(daily_soh_change)
        else:
            bayes_rul = np.inf
    else:
        bayes_rul = np.inf

    return {
        "bayes_soh_pred": round(soh_pred, 3),
        "bayes_soh_std":  round(soh_std, 3),
        "bayes_rul_days": _finite(bayes_rul),
    }


if __name__ == "__main__":
    cycles = pd.read_csv(CYCLES_CSV)
    disc   = cycles[
        (cycles["session_type"] == "discharge") & (cycles["current_mean"] > 0)
    ].copy()
    disc   = disc.dropna(subset=["soh"]).sort_values(["registration_number", "start_time"])

    # Days since first record per vehicle
    disc["date_days"] = (
        disc.groupby("registration_number")["start_time"]
        .transform(lambda t: (t - t.min()) / (1000 * 3600 * 24))
    )
    disc["date"] = pd.to_datetime(disc["start_time"], unit="ms", utc=True).dt.tz_localize(None)

    # EFC per session: |soc_diff| / 100 (discharge sessions only; soc_diff is negative)
    disc["efc"] = disc["soc_diff"].abs() / 100.0

    # Cumulative EFC per vehicle (time-ordered)
    disc = disc.sort_values(["registration_number", "start_time"])
    disc["efc_cumulative"] = disc.groupby("registration_number")["efc"].cumsum()

    # Smooth BMS SoH
    disc["soh_smooth"] = (
        disc.groupby("registration_number")["soh"]
        .transform(lambda s: s.rolling(SOH_SMOOTH_WINDOW, center=True, min_periods=1).median())
    )

    # Use BMS-reported SoH (smoothed) as the fit signal.
    # capacity_soh was found to be unreliable due to Coulomb counting capturing
    # only 38% of actual discharge energy (auxiliary loads + slow driving invisible
    # to the current threshold). BMS SoH is the manufacturer's on-board estimate
    # and is consistent at 97-98% across this fleet.
    use_col = "soh_smooth"
    print(f"Using '{use_col}' for SoH degradation trend fitting.")
    disc["soh_for_fit"] = disc["soh_smooth"]

    # ── Charging-side capacity_soh (primary dual-axis label) ──────────────────
    # Use high-confidence charging sessions where capacity_soh is reliable
    chg_soh = cycles[
        (cycles["session_type"] == "charging") &
        (cycles.get("soh_low_confidence", pd.Series(False, index=cycles.index)) == False) &
        (cycles["capacity_soh"].notna() if "capacity_soh" in cycles.columns else pd.Series(False, index=cycles.index))
    ].copy() if "capacity_soh" in cycles.columns else pd.DataFrame()

    if not chg_soh.empty:
        chg_soh["date_days"] = (
            chg_soh.groupby("registration_number")["start_time"]
            .transform(lambda t: (t - t.min()) / 86_400_000.0)
        )
        chg_soh["efc_chg"] = chg_soh.get("cum_efc", chg_soh["soc_range"].abs() / 100.0)
        print(f"Charging-side capacity_soh available for dual-axis fit: "
              f"{len(chg_soh):,} sessions across "
              f"{chg_soh['registration_number'].nunique()} vehicles")

    disc.to_csv(TREND_FILE, index=False)
    print(f"Saved SoH trends: {TREND_FILE}")

    # ── Young-fleet sanity notice ─────────────────────────────────────────────
    fleet_max_days = disc["date_days"].max() if len(disc) else 0
    if fleet_max_days < 180:
        print(f"\n{'='*75}")
        print(f"YOUNG FLEET NOTICE  (max data span ~{fleet_max_days:.0f} days)")
        print("="*75)
        print("  With <6 months of data:")
        print("  • OLS trend slopes are highly uncertain (few unique BMS SoH values)")
        print("  • Most vehicles will show rul_reliability = 'insufficient_data'")
        print("  • Composite scores reflect operating PATTERNS more than degradation STATE")
        print("  • EKF SoH near 100% and wide uncertainty bands are CORRECT behaviour")
        print("  • Anomaly detection is most actionable: flags unusual sessions NOW")
        print("  • Re-run after 6+ months for meaningful RUL estimates")

    # ── Per-vehicle trend fitting ──────────────────────────────────────────────
    np.random.seed(42)
    vehicle_results = []
    fleet_soh_slopes, fleet_epk_slopes, fleet_heat_slopes = [], [], []

    has_epk        = "energy_per_km"    in disc.columns
    has_heat       = "temp_rise_rate"   in disc.columns
    has_spread     = "cell_spread_mean" in disc.columns
    has_vsag       = "n_vsag"           in disc.columns  # data_prep_1 consolidates severity into one count
    has_ir         = "n_high_ir"        in disc.columns
    has_cycle_soh  = "cycle_soh"        in cycles.columns  # new per-cycle SoH estimate

    for reg, veh in disc.groupby("registration_number"):
        veh = veh.sort_values("date_days")
        n   = len(veh)
        if n < MIN_CYCLES_FOR_FIT:
            continue

        # ── Filter to quality sessions for OLS trend fitting ──────────────────
        # Only sessions with sufficient SOC swing carry a reliable SoH signal.
        # Falls back to all sessions if filtered set is too small.
        if "soc_range" in veh.columns:
            veh_fit = veh[veh["soc_range"].abs() >= MIN_SOC_RANGE_FOR_TREND]
            if len(veh_fit) < MIN_CYCLES_FOR_FIT:
                veh_fit = veh   # not enough quality sessions — use all
        else:
            veh_fit = veh

        days        = veh_fit["date_days"].values  # filtered — used for SoH OLS only
        days_full   = veh["date_days"].values       # full set — used for secondary signals
        soh_series  = veh_fit["soh_for_fit"].values
        current_soh = float(veh["soh_for_fit"].iloc[-1])   # always use latest
        first_date  = veh["date"].iloc[0].date()
        last_date   = veh["date"].iloc[-1].date()

        # ── Reliability gate: skip OLS when BMS SoH has < MIN_UNIQUE values ──
        # With integer-quantised BMS SoH and only a few weeks of data, OLS
        # fits noise rather than real degradation. Use fleet-average physical
        # rate as a conservative placeholder until enough variance accumulates.
        n_unique_soh = int(np.unique(soh_series).size)
        use_fleet_avg_ols = (n_unique_soh < MIN_UNIQUE_SOH_FOR_OLS)

        if use_fleet_avg_ols:
            # Will be filled from fleet_mean_soh_slope after the loop
            soh_fit      = {"slope": None, "r2": None, "p_value": None}
            rul_point    = None
            rul_lo       = None
            rul_hi       = None
            rul_reliability = "insufficient_data"
        else:
            # SoH trend
            soh_fit = fit_degradation(days, soh_series)
            fleet_soh_slopes.append(soh_fit["slope"])
            rul_point      = rul_from_fit(current_soh, soh_fit["slope"])
            rul_lo, rul_hi = bootstrap_rul(days, soh_series, current_soh)
            # R² reliability gate
            if soh_fit["r2"] < OLS_R2_THRESHOLD:
                rul_reliability = "low_r2"
            else:
                rul_reliability = "reliable"

        # EFC-based RUL
        veh_disc     = disc[disc["registration_number"] == reg].sort_values("start_time")
        efc_cum      = veh_disc["efc_cumulative"].values
        soh_efc_vals = veh_disc["soh_smooth"].values
        efc_total    = float(efc_cum[-1]) if len(efc_cum) else np.nan

        efc_fit        = fit_degradation(efc_cum, soh_efc_vals)
        rul_efc        = rul_from_fit(current_soh, efc_fit["slope"])

        days_span = float(
            (veh_disc["start_time"].max() - veh_disc["start_time"].min()) / 86_400_000
        )
        avg_efc_per_day = (efc_total / days_span) if days_span > 0 else np.nan
        rul_days_efc    = (rul_efc / avg_efc_per_day
                           if (np.isfinite(rul_efc) and np.isfinite(avg_efc_per_day) and avg_efc_per_day > 0)
                           else np.inf)

        rul_efc_lo, rul_efc_hi = bootstrap_rul_efc(efc_cum, soh_efc_vals, current_soh)

        # ── Dual-axis model (EFC + Calendar) on charging-side capacity_soh ────
        dual_row = {}
        bayes_row = {}
        if not chg_soh.empty:
            veh_chg = chg_soh[chg_soh["registration_number"] == reg].sort_values("date_days")
            if len(veh_chg) >= 5:
                chg_efc  = veh_chg["efc_chg"].values
                chg_days = veh_chg["date_days"].values
                chg_cap  = veh_chg["capacity_soh"].values

                dual_fit = fit_dual_axis(chg_efc, chg_days, chg_cap)
                dual_rul = rul_dual_axis(dual_fit, float(chg_cap[-1]),
                                         avg_efc_per_day if np.isfinite(avg_efc_per_day) else 0.0)
                dual_row = {
                    "dual_soh0":           round(dual_fit["soh0"], 3)  if np.isfinite(dual_fit["soh0"])  else None,
                    "dual_alpha":          round(dual_fit["alpha"], 5) if np.isfinite(dual_fit["alpha"]) else None,
                    "dual_beta":           round(dual_fit["beta"], 5)  if np.isfinite(dual_fit["beta"])  else None,
                    "dual_rul_days":       _finite(dual_rul["rul_days"]),
                    "dual_rul_efc_days":   _finite(dual_rul["rul_efc_days"]),
                    "dual_rul_cal_days":   _finite(dual_rul["rul_cal_days"]),
                    "dual_dominant_path":  ("calendar" if dual_rul["rul_cal_days"] <= dual_rul["rul_efc_days"]
                                            else "cycle"),
                }
                # Build extra features for BayesianRidge: cycle_soh, ir_ohm_mean,
                # cell_spread_mean, temp_rise_rate (where available)
                extra_cols = ["cycle_soh", "ir_ohm_mean", "cell_spread_mean", "temp_rise_rate",
                              "ir_ohm_mean_ewm10", "cell_spread_mean_ewm10",
                              "vsag_rate_per_hr", "dod_stress", "c_rate_chg", "thermal_stress",
                              "weak_subsystem_consistency", "subsystem_voltage_std",
                              "current_mean_discharge", "is_loaded"]
                avail_extra = [c for c in extra_cols if c in veh_chg.columns]
                extra_arr = veh_chg[avail_extra].values if avail_extra else None
                bayes_row = bayesian_rul_vehicle(chg_efc, chg_days, chg_cap,
                                                 daily_efc_rate=avg_efc_per_day,
                                                 extra_features=extra_arr)

        row = {
            "registration_number":  reg,
            "n_cycles":             n,
            "n_unique_soh":         n_unique_soh,
            "rul_reliability":      rul_reliability,
            "first_date":           first_date,
            "last_date":            last_date,
            "current_soh":          round(current_soh, 2),
            "soh_slope_%per_day":   round(soh_fit["slope"], 5) if soh_fit["slope"] is not None else None,
            "soh_r2":               round(soh_fit["r2"], 3)    if soh_fit["r2"]    is not None else None,
            "rul_days":             _finite(rul_point) if rul_point is not None else None,
            "rul_lo_days":          _finite(rul_lo)    if rul_lo    is not None else None,
            "rul_hi_days":          _finite(rul_hi)    if rul_hi    is not None else None,
            "eol_threshold":        EOL_SOH,
            "efc_total":            round(efc_total, 3) if np.isfinite(efc_total) else None,
            "avg_efc_per_day":      round(avg_efc_per_day, 4) if np.isfinite(avg_efc_per_day) else None,
            "soh_per_efc_slope":    round(efc_fit["slope"], 6),
            "rul_efc":              _finite(rul_efc),
            "rul_days_efc":         _finite(rul_days_efc),
            "rul_efc_ci_lo":        _finite(rul_efc_lo) if rul_efc_lo is not None else None,
            "rul_efc_ci_hi":        _finite(rul_efc_hi) if rul_efc_hi is not None else None,
            **dual_row,
            **bayes_row,
        }

        # Secondary signal trends — use days_full (all sessions, not filtered)
        # Energy-per-km trend (rising slope = more consumption = worse)
        if has_epk:
            epk = veh["energy_per_km"].dropna()
            if len(epk) >= MIN_CYCLES_FOR_FIT:
                epk_days = days_full[veh["energy_per_km"].notna().values]
                epk_fit  = fit_degradation(epk_days, epk.values)
                fleet_epk_slopes.append(epk_fit["slope"])
                row["epk_slope_per_day"] = round(epk_fit["slope"], 5)
                row["epk_r2"]            = round(epk_fit["r2"], 3)
            else:
                row["epk_slope_per_day"] = np.nan
                row["epk_r2"]            = np.nan

        # Temp rise rate trend (rising slope = faster heating = worse)
        if has_heat:
            thr = veh["temp_rise_rate"].dropna()
            if len(thr) >= MIN_CYCLES_FOR_FIT:
                thr_days = days_full[veh["temp_rise_rate"].notna().values]
                thr_fit  = fit_degradation(thr_days, thr.values)
                fleet_heat_slopes.append(thr_fit["slope"])
                row["heat_slope_per_day"] = round(thr_fit["slope"], 5)
                row["heat_r2"]            = round(thr_fit["r2"], 3)
            else:
                row["heat_slope_per_day"] = np.nan
                row["heat_r2"]            = np.nan

        # Cell-voltage spread trend (rising slope = growing imbalance = worse)
        if has_spread:
            cs = veh["cell_spread_mean"].dropna()
            if len(cs) >= MIN_CYCLES_FOR_FIT:
                cs_days = days_full[veh["cell_spread_mean"].notna().values]
                cs_fit  = fit_degradation(cs_days, cs.values)
                row["spread_slope_per_day"] = round(cs_fit["slope"], 6)
                row["spread_r2"]            = round(cs_fit["r2"], 3)
            else:
                row["spread_slope_per_day"] = np.nan
                row["spread_r2"]            = np.nan

        # Voltage sag trend (rising count/cycle = capacity loss).
        # data_prep_1.py consolidates the three severity buckets (severe/moderate/mild)
        # into a single n_vsag count using per-vehicle median discharge voltage as reference.
        if has_vsag:
            vs = veh["n_vsag"].dropna()
            if len(vs) >= MIN_CYCLES_FOR_FIT:
                vs_fit = fit_degradation(days_full[veh["n_vsag"].notna().values], vs.values)
                row["vsag_slope_per_day"] = round(vs_fit["slope"], 5)
                row["vsag_r2"]            = round(vs_fit["r2"], 3)
            else:
                row["vsag_slope_per_day"] = np.nan
                row["vsag_r2"]            = np.nan
            row["total_n_vsag"] = int(veh["n_vsag"].sum())

        # High internal resistance trend (rising = impedance growth)
        if has_ir:
            ir = veh["n_high_ir"].dropna()
            if len(ir) >= MIN_CYCLES_FOR_FIT:
                ir_fit = fit_degradation(days_full[veh["n_high_ir"].notna().values], ir.values)
                row["ir_slope_per_day"] = round(ir_fit["slope"], 5)
                row["ir_r2"]            = round(ir_fit["r2"], 3)
            else:
                row["ir_slope_per_day"] = np.nan
                row["ir_r2"]            = np.nan
            row["total_high_ir"] = int(veh["n_high_ir"].sum())

        # cycle_soh trend — use all sessions (not just discharge or charging).
        # Fit on the full cycles DataFrame so we maximise data points.
        if has_cycle_soh:
            veh_all = cycles[cycles["registration_number"] == reg].copy()
            veh_all["_days"] = (
                (veh_all["start_time"] - veh_all["start_time"].min()) / 86_400_000.0
            )
            cs_all = veh_all["cycle_soh"].dropna()
            if len(cs_all) >= MIN_CYCLES_FOR_FIT:
                cs_all_days = veh_all.loc[cs_all.index, "_days"].values
                cs_fit = fit_degradation(cs_all_days, cs_all.values)
                row["cycle_soh_slope_per_day"] = round(cs_fit["slope"], 5)
                row["cycle_soh_r2"]            = round(cs_fit["r2"], 3)
                row["cycle_soh_current"]        = round(float(cs_all.iloc[-1]), 3)
            else:
                row["cycle_soh_slope_per_day"] = np.nan
                row["cycle_soh_r2"]            = np.nan
                row["cycle_soh_current"]        = np.nan

        vehicle_results.append(row)

    # ── Fleet-average fallback for vehicles with too few cycles ───────────────
    fleet_mean_soh_slope = float(np.mean([s for s in fleet_soh_slopes if s < 0])) \
                           if fleet_soh_slopes else -0.01

    # Back-fill rul_days for vehicles that had insufficient unique SoH values
    # (use_fleet_avg_ols=True). Their soh_slope and rul_days were left as None.
    for row in vehicle_results:
        if row.get("rul_reliability") == "insufficient_data":
            cur = row.get("current_soh") or np.nan
            row["soh_slope_%per_day"] = round(fleet_mean_soh_slope, 5)
            row["rul_days"]           = _finite(rul_from_fit(cur, fleet_mean_soh_slope)) \
                                        if not np.isnan(cur) else None

    no_fit = cycles[cycles["session_type"] == "discharge"]["registration_number"].value_counts()
    no_fit = no_fit[no_fit < MIN_CYCLES_FOR_FIT].index.tolist()
    for reg in no_fit:
        veh = disc[disc["registration_number"] == reg]
        cur = float(veh["soh"].iloc[-1]) if len(veh) else np.nan
        vehicle_results.append({
            "registration_number": reg,
            "n_cycles":            len(veh),
            "n_unique_soh":        int(veh["soh"].nunique()) if len(veh) else 0,
            "rul_reliability":     "insufficient_data",
            "current_soh":         round(cur, 2) if not np.isnan(cur) else None,
            "soh_slope_%per_day":  round(fleet_mean_soh_slope, 5),
            "rul_days":            _finite(rul_from_fit(cur, fleet_mean_soh_slope))
                                   if not np.isnan(cur) else None,
            "note":                "fleet_average_slope",
        })

    rul_df = pd.DataFrame(vehicle_results)

    # ── Composite degradation score ───────────────────────────────────────────
    # Each slope is normalised to [0,1] across the fleet (higher = worse):
    #   SoH health: lower EKF SoH = worse  -> normalise (100 - ekf_soh)
    #               Falls back to OLS slope (inverted) if ekf_soh.csv not available.
    #   EPK slope: more positive = worse  -> normalise directly
    #   Heat slope: more positive = worse -> normalise directly
    def norm_col(col, invert=False):
        s = rul_df[col].dropna()
        lo, hi = s.min(), s.max()
        if hi == lo:
            return pd.Series(0.5, index=rul_df.index)
        normalised = (rul_df[col] - lo) / (hi - lo)
        return (1 - normalised) if invert else normalised

    # Primary SoH health signal: prefer EKF (physics-filtered) over OLS slope
    rul_df["soh_slope_norm"] = norm_col("soh_slope_%per_day", invert=True)  # kept for diagnostics
    if os.path.exists(EKF_CSV):
        ekf_df = pd.read_csv(EKF_CSV)
        ekf_last_soh = (
            ekf_df.groupby("registration_number")["ekf_soh"]
            .last()
            .reset_index()
            .rename(columns={"ekf_soh": "_ekf_soh"})
        )
        rul_df = rul_df.merge(ekf_last_soh, on="registration_number", how="left")
        # (100 - ekf_soh) = health deficit; larger = worse
        rul_df["ekf_soh_norm"] = norm_col("_ekf_soh", invert=True)
        rul_df["soh_health_norm"] = rul_df["ekf_soh_norm"]
        rul_df.drop(columns=["_ekf_soh"], inplace=True)
        print("  Composite: using EKF SoH as primary health signal")
    else:
        rul_df["ekf_soh_norm"] = np.nan
        rul_df["soh_health_norm"] = rul_df["soh_slope_norm"]
        print("  Composite: ekf_soh.csv not found — falling back to OLS slope")

    rul_df["energy_slope_norm"]     = norm_col("epk_slope_per_day",         invert=False) \
                                      if "epk_slope_per_day"         in rul_df.columns else 0.0
    rul_df["heat_slope_norm"]       = norm_col("heat_slope_per_day",        invert=False) \
                                      if "heat_slope_per_day"        in rul_df.columns else 0.0
    rul_df["spread_slope_norm"]     = norm_col("spread_slope_per_day",      invert=False) \
                                      if "spread_slope_per_day"      in rul_df.columns else 0.0
    rul_df["vsag_slope_norm"]       = norm_col("vsag_slope_per_day",        invert=False) \
                                      if "vsag_slope_per_day"        in rul_df.columns else 0.0
    rul_df["ir_slope_norm"]         = norm_col("ir_slope_per_day",          invert=False) \
                                      if "ir_slope_per_day"          in rul_df.columns else 0.0
    # cycle_soh slope: negative slope = SoH declining = worse → invert so high norm = worse
    rul_df["cycle_soh_slope_norm"]  = norm_col("cycle_soh_slope_per_day",   invert=True) \
                                      if "cycle_soh_slope_per_day"   in rul_df.columns else 0.0

    rul_df["composite_degradation_score"] = (
        COMPOSITE_WEIGHTS["soh_health_norm"]       * rul_df["soh_health_norm"].fillna(0) +
        COMPOSITE_WEIGHTS["cycle_soh_slope_norm"]  * rul_df["cycle_soh_slope_norm"].fillna(0) +
        COMPOSITE_WEIGHTS["vsag_slope_norm"]       * rul_df["vsag_slope_norm"].fillna(0) +
        COMPOSITE_WEIGHTS["ir_slope_norm"]         * rul_df["ir_slope_norm"].fillna(0) +
        COMPOSITE_WEIGHTS["energy_slope_norm"]     * rul_df["energy_slope_norm"].fillna(0) +
        COMPOSITE_WEIGHTS["heat_slope_norm"]       * rul_df["heat_slope_norm"].fillna(0) +
        COMPOSITE_WEIGHTS["spread_slope_norm"]     * rul_df["spread_slope_norm"].fillna(0)
    ).round(4)

    # ── Fleet-flag rates per vehicle ──────────────────────────────────────────
    # Summarise the boolean flags from data_prep as % of sessions flagged.
    # Keeps the session-level detail out of rul_estimates while still surfacing
    # which vehicles have the worst usage patterns.
    flag_cols = ["rapid_heating", "high_energy_per_km", "slow_charging", "fast_charging"]
    existing_flags = [c for c in flag_cols if c in cycles.columns]
    if existing_flags:
        flag_rates = (
            cycles[cycles["session_type"] == "discharge"]
            .groupby("registration_number")[existing_flags]
            .mean()
            .rename(columns={c: f"pct_{c}" for c in existing_flags})
            .round(3)
            .reset_index()
        )
        # slow/fast charging are charging-session flags — compute separately
        chg_flags = [c for c in ["slow_charging", "fast_charging"] if c in cycles.columns]
        if chg_flags:
            chg_rates = (
                cycles[cycles["session_type"] == "charging"]
                .groupby("registration_number")[chg_flags]
                .mean()
                .rename(columns={c: f"pct_{c}" for c in chg_flags})
                .round(3)
                .reset_index()
            )
            # overwrite the discharge-computed charging rates with correct values
            flag_rates = flag_rates.drop(columns=[f"pct_{c}" for c in chg_flags], errors="ignore")
            flag_rates = flag_rates.merge(chg_rates, on="registration_number", how="left")
        rul_df = rul_df.merge(flag_rates, on="registration_number", how="left")

    # ── Merge anomaly summary from anomaly_scores.csv (if available) ──────────
    if os.path.exists(ANOMALY_FILE):
        anom = pd.read_csv(ANOMALY_FILE)
        anom_summary = (
            anom.groupby("registration_number")
            .agg(
                n_if_anomalies    = ("if_anomaly",       "sum"),
                n_cusum_soh       = ("cusum_soh_alarm",        "sum"),
                n_cusum_epk       = ("cusum_epk_alarm",        "sum"),
                n_cusum_heat      = ("cusum_heat_alarm",       "sum"),
                n_cusum_spread    = ("cusum_spread_alarm",     "sum"),
                n_cusum_cycle_soh = ("cusum_cycle_soh_alarm",  "sum"),
                n_combined_anom   = ("anomaly",                "sum"),
                if_score_mean     = ("if_score",               "mean"),
            )
            .reset_index()
        )
        anom_summary["if_score_mean"] = anom_summary["if_score_mean"].round(4)
        rul_df = rul_df.merge(anom_summary, on="registration_number", how="left")
        print(f"Merged anomaly scores from {ANOMALY_FILE}")

    # ── Merge neural reconstruction anomaly summary ───────────────────────────
    if os.path.exists(NEURAL_FILE):
        neural = pd.read_csv(NEURAL_FILE)
        neural_summary = (
            neural.groupby("registration_number")
            .agg(
                neural_rec_err_mean = ("reconstruction_err", "mean"),
                neural_rec_err_p95  = ("reconstruction_err", lambda x: x.quantile(0.95)),
                n_neural_anomalies  = ("is_anomaly",         "sum"),
                neural_anomaly_pct  = ("anomaly_pct",        "mean"),
            )
            .reset_index()
        )
        neural_summary[["neural_rec_err_mean", "neural_rec_err_p95",
                        "neural_anomaly_pct"]] = \
            neural_summary[["neural_rec_err_mean", "neural_rec_err_p95",
                            "neural_anomaly_pct"]].round(4)
        rul_df = rul_df.merge(neural_summary, on="registration_number", how="left")
        print(f"Merged neural predictions from {NEURAL_FILE}")

    rul_df = rul_df.sort_values("composite_degradation_score", ascending=False)
    rul_df.to_csv(RUL_FILE, index=False)

    # ── Console summary ───────────────────────────────────────────────────────
    print("\n" + "=" * 75)
    print("VEHICLE DEGRADATION RANKING  (worst -> best, by composite score)")
    print("=" * 75)
    show_cols = ["registration_number", "n_cycles", "current_soh",
                 "soh_slope_%per_day", "rul_reliability", "rul_days", "soh_r2",
                 "cycle_soh_current", "cycle_soh_slope_per_day", "cycle_soh_r2",
                 "dual_rul_days", "dual_rul_cal_days", "dual_rul_efc_days", "dual_dominant_path",
                 "bayes_rul_days", "bayes_soh_pred", "bayes_soh_std",
                 "vsag_slope_per_day", "ir_slope_per_day",
                 "epk_slope_per_day", "heat_slope_per_day", "spread_slope_per_day",
                 "total_n_vsag", "total_high_ir",
                 "pct_rapid_heating", "pct_high_energy_per_km",
                 "pct_slow_charging", "pct_fast_charging",
                 "composite_degradation_score",
                 "n_combined_anom", "if_score_mean",
                 "n_neural_anomalies", "neural_rec_err_mean"]
    show = rul_df[[c for c in show_cols if c in rul_df.columns]]
    print(show.to_string(index=False))

    print(f"\nFleet mean SoH degradation : {fleet_mean_soh_slope:.5f} %SoH/day")
    print(f"Fleet mean RUL             : {rul_from_fit(98.0, fleet_mean_soh_slope):.0f} days")
    print(f"\nSaved RUL estimates        : {RUL_FILE}")

    # ── Young fleet reliability summary ──────────────────────────────────────
    if "rul_reliability" in rul_df.columns:
        rel_counts = rul_df["rul_reliability"].value_counts()
        print(f"\nRUL reliability distribution:")
        for cat, cnt in rel_counts.items():
            print(f"  {cat:<25}: {cnt} vehicles")
        n_insuff = rel_counts.get("insufficient_data", 0)
        if n_insuff / max(len(rul_df), 1) > 0.5:
            print(f"\n  WARNING: {n_insuff}/{len(rul_df)} vehicles have insufficient_data reliability.")
            print("     This is expected for a young fleet (<6 months).")
            print("     Composite scores and anomaly flags are still valid and actionable.")
