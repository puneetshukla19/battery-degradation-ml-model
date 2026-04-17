import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) if "__file__" in dir() else os.getcwd())

"""
ekf_soh.py — Extended Kalman Filter for per-vehicle SoH tracking.

State vector  x = [SoH (%), IR_drift (Ω), spread_drift (V), temp_drift (°C/hr)]
               ^4-dimensional; temp_drift captures gradual thermal aging effect.

Process model (per charging-session step):
    SoH(k+1)      = SoH(k) − α·I_norm·load_fac·Δefc − β·Δdays·(CAL/365) + w₁
    IR(k+1)       = IR(k)  + γ·Δefc                                       + w₂
    spread(k+1)   = spread(k) + δ·Δefc                                    + w₃
    temp_drift(k+1) = temp_drift(k) + ζ·Δefc                              + w₄

  Where:
    I_norm   = current_mean_discharge / I_NOMINAL  (Peukert current stress)
    load_fac = LOAD_STRESS_FACTOR if is_loaded else 1.0

Observation model (charging sessions; non-NaN entries only):
    z = [cycle_soh, bms_soh, ir_ohm_mean, cell_spread_mean, temp_rise_rate_demeaned]
    H = [[1, 0, 0, 0],   # cycle_soh            ≈ SoH
         [1, 0, 0, 0],   # bms_soh              ≈ SoH (noisier; integer steps)
         [0, 1, 0, 0],   # ir_ohm_mean          ≈ baseline_ir + IR_drift
         [0, 0, 1, 0],   # cell_spread_mean      ≈ baseline_spread + spread_drift
         [0, 0, 0, 1]]   # temp_rise_rate - μ_T ≈ temp_drift

  Notes:
    - capacity_soh REMOVED: replaced by cycle_soh (available every cycle, not just
      full-swing charges). cycle_soh removes the 38%-coverage bias of Coulomb counting.
    - temp_rise_rate is fleet-demeaned before use so x[3]=0 for a typical vehicle.

Outputs
-------
ekf_soh.csv  — one row per charging session with:
    ekf_soh, ekf_soh_std, ekf_ir, ekf_spread, ekf_temp_drift,
    ekf_rul_days, ekf_rul_days_lo, ekf_rul_days_hi,
    cycle_soh_obs, bms_soh_obs, temp_rise_rate_obs
"""

import numpy as np
import pandas as pd
from config import (
    CYCLES_CSV, EKF_CSV, BASE,
    EKF_Q_DIAG, EKF_R_DIAG,
    EOL_SOH, EFC_MAX, CAL_AGING_RATE,
    EKF_ALPHA, PEUKERT_N, I_NOMINAL_A, LOAD_STRESS_FACTOR, ZETA,
    CYCLE_SOH_OBS_CAP, CYCLE_SOH_MIN_BLOCK_DOD, CYCLE_SOH_REF_DOD,
)

# ── EKF physical parameters ────────────────────────────────────────────────────
ALPHA = EKF_ALPHA   # %SoH lost per EFC (base cycle aging)
BETA  = 1.0         # scale on calendar aging term
GAMMA = 0.001       # Ω per EFC  (IR growth)
DELTA = 0.0001      # V per EFC  (spread growth)
# ZETA imported from config  (°C/hr drift per EFC, thermal aging)

# Initial state and covariance — 4D
X0 = np.array([100.0, 0.0, 0.0, 0.0])          # fresh pack: full SoH, no drift
P0 = np.diag([4.0, 1e-3, 1e-4, 1e-3])          # ±2% SoH, small IR/spread/thermal uncertainty

# Noise matrices — built from config so both Q and R always match the state/obs dim
Q     = np.diag(EKF_Q_DIAG)   # process noise (4×4)
R_ALL = np.diag(EKF_R_DIAG)   # observation noise (5×5)

# Observation matrix H  (5 × 4)
H_ALL = np.array([
    [1.0, 0.0, 0.0, 0.0],   # cycle_soh            → SoH
    [1.0, 0.0, 0.0, 0.0],   # bms_soh              → SoH
    [0.0, 1.0, 0.0, 0.0],   # ir_ohm_mean          → IR_drift
    [0.0, 0.0, 1.0, 0.0],   # cell_spread_mean     → spread_drift
    [0.0, 0.0, 0.0, 1.0],   # temp_rise_rate (demeaned) → temp_drift
])

# Observation column names (positions match z vector built in run_ekf_fleet)
OBS_COLS = ["cycle_soh", "soh", "ir_ohm_mean", "cell_spread_mean", "temp_rise_rate"]


# ── EKF core ───────────────────────────────────────────────────────────────────

def _build_F() -> np.ndarray:
    """State transition Jacobian F (4×4) — identity since model is linear in state."""
    return np.eye(4)


def _process(x: np.ndarray, delta_efc: float, delta_days: float,
             current_mean_discharge: float = np.nan,
             is_loaded: bool = False) -> np.ndarray:
    """
    Nonlinear (but linear-in-state) state transition.

    Peukert current stress:
        I_norm = max(current_mean_discharge / I_NOMINAL_A, 1.0)
        alpha_adj = ALPHA * (1 + PEUKERT_N * (I_norm - 1))
    Load stress: multiply effective EFC by LOAD_STRESS_FACTOR when loaded.
    """
    x_new = x.copy()

    # Current-adjusted cycle aging coefficient
    if np.isfinite(current_mean_discharge) and current_mean_discharge > 0:
        I_norm    = current_mean_discharge / I_NOMINAL_A
        alpha_adj = ALPHA * (1.0 + PEUKERT_N * max(0.0, I_norm - 1.0))
    else:
        alpha_adj = ALPHA

    load_fac = LOAD_STRESS_FACTOR if is_loaded else 1.0

    x_new[0] -= (alpha_adj * load_fac * delta_efc +
                 BETA * delta_days * (CAL_AGING_RATE / 365.0))
    x_new[1] += GAMMA * delta_efc
    x_new[2] += DELTA * delta_efc
    x_new[3] += ZETA  * delta_efc   # thermal drift accumulates with cycling
    return x_new


def ekf_step(x: np.ndarray, P: np.ndarray,
             z_obs: np.ndarray,
             delta_efc: float, delta_days: float,
             current_mean_discharge: float = np.nan,
             is_loaded: bool = False,
             r_cycle_soh_override: float = None) -> tuple:
    """
    One EKF predict + update step.

    Parameters
    ----------
    x                     : current state (4,)
    P                     : current covariance (4,4)
    z_obs                 : observation vector (5,) — NaN entries are skipped
    delta_efc             : EFC change since last session
    delta_days            : calendar days since last session
    current_mean_discharge: mean discharge current this session (A)
    is_loaded             : whether vehicle was loaded this session
    r_cycle_soh_override  : if given, replaces R[0,0] for this step (adaptive DoD scaling)
    """
    # ── Predict ──
    F   = _build_F()
    x_p = _process(x, delta_efc, delta_days, current_mean_discharge, is_loaded)
    P_p = F @ P @ F.T + Q

    # ── Update (skip NaN observations) ──
    valid = ~np.isnan(z_obs)
    if not valid.any():
        return x_p, P_p

    H_v = H_ALL[valid]
    R_v = R_ALL[np.ix_(valid, valid)].copy()

    # Apply adaptive R for cycle_soh (observation index 0) if it is valid this step
    if r_cycle_soh_override is not None and valid[0]:
        # Find position of obs-0 in the compressed valid array
        idx_in_valid = np.where(valid)[0].tolist().index(0)
        R_v[idx_in_valid, idx_in_valid] = r_cycle_soh_override

    z_v = z_obs[valid]

    S     = H_v @ P_p @ H_v.T + R_v
    K     = P_p @ H_v.T @ np.linalg.inv(S)
    inn   = z_v - H_v @ x_p              # innovation

    x_new = x_p + K @ inn
    x_new[0] = np.clip(x_new[0], 0.0, 105.0)  # SoH plausible range
    P_new = (np.eye(4) - K @ H_v) @ P_p

    return x_new, P_new


# ── RUL from EKF state ─────────────────────────────────────────────────────────

def _rul_from_ekf(soh: float, soh_std: float, avg_efc_per_day: float,
                  eol: float = EOL_SOH) -> dict:
    """Compute point + uncertainty RUL from EKF SoH and its posterior std."""
    if soh <= eol or not np.isfinite(avg_efc_per_day) or avg_efc_per_day <= 0:
        return {"ekf_rul_days": 0.0, "ekf_rul_days_lo": 0.0, "ekf_rul_days_hi": 0.0}

    daily_soh_rate = (ALPHA * avg_efc_per_day + BETA * CAL_AGING_RATE / 365.0)
    if daily_soh_rate <= 0:
        return {"ekf_rul_days": np.inf, "ekf_rul_days_lo": np.inf, "ekf_rul_days_hi": np.inf}

    remaining    = soh - eol
    rul_point    = remaining / daily_soh_rate
    remaining_lo = max(0.0, remaining - 1.96 * soh_std)
    remaining_hi = remaining + 1.96 * soh_std
    rul_lo       = remaining_lo / daily_soh_rate
    rul_hi       = remaining_hi / daily_soh_rate

    def _cap(v):
        return round(min(v, 36500.0), 0) if np.isfinite(v) else None

    return {
        "ekf_rul_days":    _cap(rul_point),
        "ekf_rul_days_lo": _cap(rul_lo),
        "ekf_rul_days_hi": _cap(rul_hi),
    }


# ── Per-vehicle batch loop ─────────────────────────────────────────────────────

def run_ekf_fleet(cycles: pd.DataFrame) -> pd.DataFrame:
    """
    Run EKF on charging sessions for every vehicle.
    Uses cycle_soh (not capacity_soh) as the primary SoH observation.
    Also uses current_mean_discharge and is_loaded to adjust cycle aging.
    """
    # Charging sessions — cycle_soh is available here and is the reliable SoH proxy
    chg = cycles[cycles["session_type"] == "charging"].copy()
    chg = chg.sort_values(["registration_number", "start_time"]).reset_index(drop=True)

    if "cum_efc" not in chg.columns:
        chg["_efc_s"] = chg["soc_range"].abs() / 100.0
        chg["cum_efc"] = chg.groupby("registration_number")["_efc_s"].transform("cumsum")
        chg.drop(columns=["_efc_s"], inplace=True)

    if "days_since_first" not in chg.columns:
        chg["days_since_first"] = (
            chg.groupby("registration_number")["start_time"]
            .transform(lambda x: (x - x.min()) / 86_400_000.0)
        )

    # Fleet-wide temp_rise_rate baseline for demeaning the thermal observation
    fleet_temp_mu = float(
        chg["temp_rise_rate"].median()
        if "temp_rise_rate" in chg.columns and chg["temp_rise_rate"].notna().any()
        else 0.0
    )
    print(f"  Fleet temp_rise_rate baseline (median): {fleet_temp_mu:.4f} °C/hr")

    results = []

    for reg, veh in chg.groupby("registration_number"):
        veh = veh.sort_values("start_time").reset_index(drop=True)

        days_span = float(veh["days_since_first"].iloc[-1])
        efc_total = float(veh["cum_efc"].iloc[-1])
        avg_efc_per_day = (efc_total / days_span) if days_span > 1.0 else np.nan

        # ── Pre-compute EWM-smoothed cycle_soh per vehicle ───────────────────
        # Only smooth on quality observations (not capped, sufficient block DoD).
        # Sessions outside the quality gate keep their raw value (used to decide NaN later).
        block_dod = veh["block_soc_diff"].abs() if "block_soc_diff" in veh.columns else veh["soc_range"].abs()
        csoh_raw  = veh["cycle_soh"].copy()
        quality_mask = (
            csoh_raw.notna() &
            (csoh_raw < CYCLE_SOH_OBS_CAP) &
            (block_dod >= CYCLE_SOH_MIN_BLOCK_DOD)
        )
        # EWM on valid values only; propagate last valid to non-quality rows for bookkeeping
        csoh_valid = csoh_raw.where(quality_mask)
        csoh_ewm   = csoh_valid.ewm(span=5, min_periods=1).mean()
        # Clamp smoothed values to physical range [88, 101]
        csoh_ewm   = csoh_ewm.clip(lower=88.0, upper=101.0)
        veh = veh.copy()
        veh["_csoh_ewm"]   = csoh_ewm
        veh["_block_dod"]  = block_dod
        veh["_csoh_quality"] = quality_mask

        x = X0.copy()
        P = P0.copy()

        for _, row in veh.iterrows():
            delta_efc  = float(row.get("soc_range", 0.0) or 0.0) / 100.0
            delta_days = float(row.get("days_since_first", 0.0) or 0.0)
            if len(results) > 0 and results[-1]["registration_number"] == reg:
                prev_days  = float(results[-1].get("days_since_first_session", delta_days))
                delta_days = max(0.0, delta_days - prev_days)
            else:
                delta_days = 0.0

            # Session-level current and load
            current_disc = float(row.get("current_mean_discharge", np.nan) or np.nan)
            is_loaded    = bool(row.get("is_loaded", False))

            # ── cycle_soh quality gate ────────────────────────────────────────
            # Use EWM-smoothed value only when quality conditions are met.
            # Cap at 99.5%: a 100% reading is an artefact of the Coulomb-count
            # ceiling, not a true measurement — it provides zero downward signal.
            # Shallow blocks (block_dod < 20%) are too noisy to be informative.
            use_csoh    = bool(row.get("_csoh_quality", False))
            csoh_obs    = float(row["_csoh_ewm"]) if use_csoh and pd.notna(row["_csoh_ewm"]) else np.nan

            # Adaptive R for cycle_soh: deeper DoD → more reliable → lower R
            r_csoh_override = None
            if use_csoh and np.isfinite(csoh_obs):
                dod = float(row["_block_dod"])
                ref = float(CYCLE_SOH_REF_DOD)
                scale = (ref / max(dod, 10.0)) ** 2
                r_csoh_override = float(EKF_R_DIAG[0]) * scale  # R_base * scaling factor

            # Build observation vector — temp_rise_rate demeaned by fleet baseline
            # bms_soh (z[1]) excluded: BMS SoH is integer-stepped and biased;
            # cycle_soh provides a cleaner direct SoH signal.
            z = np.array([
                csoh_obs,
                np.nan,                  # bms_soh excluded
                float(row["ir_ohm_mean"]) if pd.notna(row.get("ir_ohm_mean")) else np.nan,
                float(row["cell_spread_mean"]) if pd.notna(row.get("cell_spread_mean")) else np.nan,
                (float(row["temp_rise_rate"]) - fleet_temp_mu)
                    if pd.notna(row.get("temp_rise_rate")) else np.nan,
            ])

            x, P = ekf_step(x, P, z, delta_efc, delta_days,
                            current_mean_discharge=current_disc,
                            is_loaded=is_loaded,
                            r_cycle_soh_override=r_csoh_override)

            ekf_soh     = float(x[0])
            ekf_soh_std = float(np.sqrt(max(0.0, P[0, 0])))
            rul_dict    = _rul_from_ekf(ekf_soh, ekf_soh_std, avg_efc_per_day)

            results.append({
                "registration_number":      reg,
                "session_id":               row.get("session_id"),
                "start_time":               row.get("start_time"),
                "days_since_first_session": float(row.get("days_since_first", 0.0)),
                "cum_efc":                  float(row.get("cum_efc", 0.0)),
                "cycle_soh_obs":            csoh_obs,   # EWM-smoothed quality-gated value
                "bms_soh_obs":              z[1],
                "temp_rise_rate_obs":       z[4],       # demeaned value
                "current_mean_discharge":   current_disc,
                "is_loaded":                is_loaded,
                "ekf_soh":                  round(ekf_soh, 3),
                "ekf_soh_std":              round(ekf_soh_std, 4),
                "ekf_ir":                   round(float(x[1]), 6),
                "ekf_spread":               round(float(x[2]), 6),
                "ekf_temp_drift":           round(float(x[3]), 4),
                **rul_dict,
            })

    return pd.DataFrame(results)


# ── Main ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print(f"Loading cycles from {CYCLES_CSV} ...")
    cycles = pd.read_csv(CYCLES_CSV)
    print(f"  {len(cycles):,} sessions, {cycles['registration_number'].nunique()} vehicles")

    # ── Young-fleet notice ────────────────────────────────────────────────────
    max_days = cycles.get("days_since_first", pd.Series([0])).max() if "days_since_first" in cycles.columns else 0
    if max_days < 180:
        print(f"\n  NOTE: Young fleet detected (max data span ~{max_days:.0f} days).")
        print("  EKF SoH estimates are dominated by the prior for <6 months of data.")
        print("  Expect EKF SoH near 100% and wide uncertainty bands — this is correct behaviour.")

    print("Running EKF fleet-wide ...")
    ekf_df = run_ekf_fleet(cycles)
    print(f"  EKF output: {len(ekf_df):,} charging sessions")

    ekf_df.to_csv(EKF_CSV, index=False)
    print(f"Saved: {EKF_CSV}")

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n" + "=" * 75)
    print("EKF SOH TRACKING SUMMARY  (per vehicle, sorted by ekf_soh)")
    print("=" * 75)
    summary = (
        ekf_df.groupby("registration_number")
        .agg(
            n_sessions       = ("ekf_soh",         "count"),
            ekf_soh_final    = ("ekf_soh",          "last"),
            ekf_soh_std_mean = ("ekf_soh_std",      "mean"),
            ekf_ir_final     = ("ekf_ir",           "last"),
            ekf_temp_drift   = ("ekf_temp_drift",   "last"),
            ekf_rul_days     = ("ekf_rul_days",     "last"),
            cum_efc_final    = ("cum_efc",          "last"),
        )
        .round(4)
        .sort_values("ekf_soh_final")
    )
    print(summary.to_string())

    fleet_last = ekf_df.groupby("registration_number")
    print(f"\nFleet EKF SoH  : "
          f"mean={fleet_last['ekf_soh'].last().mean():.2f}%  "
          f"std={fleet_last['ekf_soh'].last().std():.2f}%")
    print(f"Fleet EKF RUL  : "
          f"median={fleet_last['ekf_rul_days'].last().median():.0f} days")
    print(f"Fleet EKF temp_drift: "
          f"mean={fleet_last['ekf_temp_drift'].last().mean():.4f} °C/hr "
          "(0 = nominal; positive = heating faster than fleet baseline)")

    # ── Verification ─────────────────────────────────────────────────────────
    print("\n-- Verification checks --")
    ekf_last = ekf_df.groupby("registration_number").last().reset_index()

    # Cycle_soh observation usage rate
    n_total  = len(ekf_df)
    n_csoh   = ekf_df["cycle_soh_obs"].notna().sum()
    print(f"  cycle_soh obs used: {n_csoh:,}/{n_total:,} sessions "
          f"({100*n_csoh/n_total:.1f}%) — quality-gated + EWM-smoothed")

    valid_csoh = ekf_last.dropna(subset=["cycle_soh_obs"])
    if len(valid_csoh):
        diff = (valid_csoh["ekf_soh"] - valid_csoh["cycle_soh_obs"]).abs()
        pct_within_2 = (diff <= 2.0).mean() * 100
        msg = f"  ekf_soh within ±2% of cycle_soh (quality obs): {pct_within_2:.0f}% of vehicles"
        print(msg + " [OK]" if pct_within_2 >= 80 else msg + " [WARN]")

    if len(ekf_df) > 100:
        early = ekf_df[ekf_df["cum_efc"] < ekf_df["cum_efc"].quantile(0.25)]["ekf_soh_std"].mean()
        late  = ekf_df[ekf_df["cum_efc"] > ekf_df["cum_efc"].quantile(0.75)]["ekf_soh_std"].mean()
        print(f"  ekf_soh_std early vs late: {early:.4f} -> {late:.4f} "
              f"({'shrinking [OK]' if late < early else '[WARN] not shrinking - may need more data'})")

    # Young fleet: check RUL isn't misleadingly low
    median_rul = ekf_df.groupby("registration_number")["ekf_rul_days"].last().median()
    if median_rul is not None and np.isfinite(float(median_rul or np.inf)):
        if float(median_rul) < 365:
            print(f"  [WARN] Median EKF RUL = {median_rul:.0f} days. "
                  "For a young fleet this likely reflects prior uncertainty rather than real degradation.")
        else:
            print(f"  Median EKF RUL = {median_rul:.0f} days [OK] (consistent with young fleet)")
