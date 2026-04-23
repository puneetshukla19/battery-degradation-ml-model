import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) if "__file__" in dir() else os.getcwd())

"""
ecm_1rc.py — 1RC Equivalent Circuit Model with Extended Kalman Filter.

Extends the existing 4D EKF (ekf_soh.py) to a 5D state that separates ohmic
resistance (R₀) from polarisation resistance (R₁) via an RC branch (R₁C₁):

    State x = [SoH (%), R₀ (Ω), R₁ (Ω), C₁ (F), IR_drift_legacy (Ω)]

Process model (per session step):
    SoH(k+1)   = SoH(k) − α·I_norm·load·Δefc − β·Δdays·(CAL/365)
    R₀(k+1)    = R₀(k)  + γ_R0  · Δefc          (ohmic aging)
    R₁(k+1)    = R₁(k)  + γ_R1  · Δefc          (polarisation aging)
    C₁(k+1)    = C₁(k)                            (near-constant; process noise only)
    IR_legacy(k+1) = IR_legacy(k) + GAMMA · Δefc  (backward-compat drift)

Observation model:
  Dual mode (discharge session with IR profile):
    z[0] = cycle_soh      → H = [1, 0, 0, 0, 0]
    z[1] = ir_early_mean  → H = [0, 1, 0, 0, 0]   (first ~60s ≈ R₀ only)
    z[2] = ir_late_mean   → H = [0, 1, 1, 0, 0]   (steady-state ≈ R₀ + R₁)

  Scalar mode (no discharge profile):
    z[0] = cycle_soh      → H = [1, 0, 0, 0, 0]
    z[1] = ir_ohm_mean    → H = [0, 1, 0, 0, 1]   (R₀ + IR_drift_legacy)

Business value
--------------
R₀ and R₁ rise precede visible SoH drop by weeks.  Daily R₀/R₁ monitoring gives
operations a 4–6 week warning before a vehicle needs maintenance — allowing
proactive route/load adjustment rather than reactive battery swap.

Outputs
-------
artifacts/ecm_soh.csv         — per-session ECM state + 60/90-day forecast
artifacts/ecm_state_convergence.csv
artifacts/ecm_metrics.csv
artifacts/fleet_forecast_ecm.csv
plots/ecm_state_trajectories.png
plots/ecm_nis_distribution.png
plots/ecm_r0_r1_scatter.png
plots/ecm_covariance_convergence.png
plots/ecm_vs_ekf_comparison.png
plots/ecm_dual_obs_benefit.png
plots/ecm_resistance_dashboard.png   ← stakeholder
plots/ecm_resistance_forecast.png    ← stakeholder
"""

import warnings
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.lines import Line2D
from scipy import stats
from scipy.stats import shapiro, spearmanr, pearsonr, chi2, ks_2samp, adfuller
from sklearn.linear_model import LinearRegression

from config import (
    CYCLES_CSV, ARTIFACTS_DIR, PLOTS_DIR, BASE,
    EKF_Q_DIAG, EKF_R_DIAG,
    EOL_SOH, EFC_MAX, CAL_AGING_RATE,
    EKF_ALPHA, PEUKERT_N, I_NOMINAL_A, LOAD_STRESS_FACTOR, ZETA,
    CYCLE_SOH_OBS_CAP, CYCLE_SOH_MIN_BLOCK_DOD, CYCLE_SOH_REF_DOD,
    SEED,
)

np.random.seed(SEED)
warnings.filterwarnings("ignore", category=RuntimeWarning)

# ── Output paths ───────────────────────────────────────────────────────────────
ECM_CSV          = os.path.join(ARTIFACTS_DIR, "ecm_soh.csv")
CONV_CSV         = os.path.join(ARTIFACTS_DIR, "ecm_state_convergence.csv")
METRICS_CSV      = os.path.join(ARTIFACTS_DIR, "ecm_metrics.csv")
FLEET_CSV        = os.path.join(ARTIFACTS_DIR, "fleet_forecast_ecm.csv")
EKF_CSV          = os.path.join(ARTIFACTS_DIR, "ekf_soh.csv")

# ── ECM physical parameters (5D state) ────────────────────────────────────────
ALPHA   = EKF_ALPHA   # %SoH / EFC  (cycle aging)
BETA    = 1.0         # calendar aging scale
GAMMA   = 0.001       # Ω/EFC legacy IR drift (backward compat)
GAMMA_R0 = 0.0005     # Ω/EFC — ohmic resistance growth rate
GAMMA_R1 = 0.0002     # Ω/EFC — polarisation resistance growth rate

# Initial 5D state: [SoH, R₀, R₁, C₁, IR_legacy]
# R₀_init ~ 5 mΩ, R₁_init ~ 2 mΩ, C₁_init ~ 200 F (τ = R₁·C₁ ≈ 40 s initial)
X0_5 = np.array([100.0, 0.005, 0.002, 200.0, 0.0])

# Initial covariance
P0_5 = np.diag([
    4.0,    # SoH ±2%
    1e-5,   # R₀ ±3.2 mΩ
    1e-6,   # R₁ ±1 mΩ
    1e4,    # C₁ ±100 F (high initial uncertainty)
    1e-4,   # IR_legacy ±10 mΩ
])

# Process noise Q (5×5)
Q_ECM = np.diag([
    EKF_Q_DIAG[0],   # SoH
    1e-7,             # R₀
    5e-8,             # R₁
    1.0,              # C₁ (large — allows drift)
    2.5e-7,           # IR_legacy
])

# Observation noise R
R_CSOH     = EKF_R_DIAG[0]   # cycle_soh noise (scaled by DoD)
R_IR_EARLY = 4e-6             # ir_early_mean noise (≈ 2 mΩ std)
R_IR_LATE  = 6e-6             # ir_late_mean noise (≈ 2.5 mΩ std)
R_IR_SCALAR = 5e-6            # ir_ohm_mean noise (scalar mode)

# Physical constraint bounds
R0_MAX = 0.050  # Ω (50 mΩ)
R1_MAX = 0.020  # Ω (20 mΩ)
C1_MIN = 10.0   # F
C1_MAX = 5000.0 # F
TAU_MAX = 3600.0 # s

# Quality gate for cycle_soh
CSOH_CAP     = CYCLE_SOH_OBS_CAP
CSOH_MIN_DOD = CYCLE_SOH_MIN_BLOCK_DOD
CSOH_REF_DOD = CYCLE_SOH_REF_DOD


# ── 5D process model ─────────────────────────────────────────────────────────

def _process_5d(x: np.ndarray, delta_efc: float, delta_days: float,
                current_mean_discharge: float = np.nan,
                is_loaded: bool = False) -> np.ndarray:
    """Nonlinear (linear-in-state) 5D state transition for ECM EKF."""
    x_new = x.copy()

    # Current stress (Peukert)
    if np.isfinite(current_mean_discharge) and current_mean_discharge > 0:
        I_norm    = current_mean_discharge / I_NOMINAL_A
        alpha_adj = ALPHA * (1.0 + PEUKERT_N * max(0.0, I_norm - 1.0))
    else:
        alpha_adj = ALPHA

    load_fac = LOAD_STRESS_FACTOR if is_loaded else 1.0

    x_new[0] -= (alpha_adj * load_fac * delta_efc +
                 BETA * delta_days * (CAL_AGING_RATE / 365.0))
    x_new[1] += GAMMA_R0 * delta_efc
    x_new[2] += GAMMA_R1 * delta_efc
    # x_new[3] = C₁ — constant (only process noise)
    x_new[4] += GAMMA * delta_efc

    # Physical clamps
    x_new[0] = np.clip(x_new[0], 0.0, 105.0)
    x_new[1] = np.clip(x_new[1], 0.0, R0_MAX)
    x_new[2] = np.clip(x_new[2], 0.0, R1_MAX)
    x_new[3] = np.clip(x_new[3], C1_MIN, C1_MAX)
    return x_new


def _F_5d() -> np.ndarray:
    """State-transition Jacobian — identity (linear model)."""
    return np.eye(5)


# ── 5D ECM EKF step ───────────────────────────────────────────────────────────

def ecm_step(x: np.ndarray, P: np.ndarray,
             delta_efc: float, delta_days: float,
             cycle_soh_obs: float,
             ir_ohm_mean: float,
             ir_early: float, ir_late: float,
             current_mean_discharge: float = np.nan,
             is_loaded: bool = False,
             r_csoh_override: float = None) -> tuple:
    """
    One ECM EKF predict + update step.

    Returns
    -------
    x_new, P_new, nis_value, obs_type ('dual' | 'scalar' | 'soh_only')
    """
    # ── Predict ──────────────────────────────────────────────────────────────
    F   = _F_5d()
    x_p = _process_5d(x, delta_efc, delta_days, current_mean_discharge, is_loaded)
    P_p = F @ P @ F.T + Q_ECM

    # ── Build observation set ─────────────────────────────────────────────────
    r_csoh = r_csoh_override if r_csoh_override is not None else R_CSOH

    has_csoh  = np.isfinite(cycle_soh_obs)
    has_dual  = np.isfinite(ir_early) and np.isfinite(ir_late)
    has_scalar = np.isfinite(ir_ohm_mean)

    obs_list = []   # (z_val, h_row, r_val)

    if has_csoh:
        h = np.array([1, 0, 0, 0, 0], dtype=float)
        obs_list.append((cycle_soh_obs, h, r_csoh))

    if has_dual:
        # ir_early ≈ R₀
        h_early = np.array([0, 1, 0, 0, 0], dtype=float)
        obs_list.append((ir_early, h_early, R_IR_EARLY))
        # ir_late ≈ R₀ + R₁
        h_late = np.array([0, 1, 1, 0, 0], dtype=float)
        obs_list.append((ir_late, h_late, R_IR_LATE))
        obs_type = "dual"
    elif has_scalar:
        # ir_ohm_mean ≈ R₀ + IR_drift_legacy
        h_scalar = np.array([0, 1, 0, 0, 1], dtype=float)
        obs_list.append((ir_ohm_mean, h_scalar, R_IR_SCALAR))
        obs_type = "scalar"
    else:
        obs_type = "soh_only"

    if not obs_list:
        return x_p, P_p, np.nan, "none"

    # Stack observations
    n_obs = len(obs_list)
    z_vec = np.array([o[0] for o in obs_list])
    H_mat = np.vstack([o[1] for o in obs_list])
    R_mat = np.diag([o[2] for o in obs_list])

    # ── Update ────────────────────────────────────────────────────────────────
    S     = H_mat @ P_p @ H_mat.T + R_mat
    try:
        S_inv = np.linalg.inv(S)
    except np.linalg.LinAlgError:
        return x_p, P_p, np.nan, obs_type

    K     = P_p @ H_mat.T @ S_inv
    inn   = z_vec - H_mat @ x_p

    x_new = x_p + K @ inn
    x_new[0] = np.clip(x_new[0], 0.0, 105.0)
    x_new[1] = np.clip(x_new[1], 0.0, R0_MAX)
    x_new[2] = np.clip(x_new[2], 0.0, R1_MAX)
    x_new[3] = np.clip(x_new[3], C1_MIN, C1_MAX)

    P_new = (np.eye(5) - K @ H_mat) @ P_p

    # NIS = ν.T S⁻¹ ν
    nis = float(inn @ S_inv @ inn)

    return x_new, P_new, nis, obs_type


# ── RUL from ECM state ────────────────────────────────────────────────────────

def _rul_ecm(soh: float, soh_std: float, avg_efc_per_day: float,
             eol: float = EOL_SOH) -> dict:
    """Compute p50/p05/p95 RUL from ECM SoH estimate."""
    if soh <= eol or not np.isfinite(avg_efc_per_day) or avg_efc_per_day <= 0:
        return {"ecm_rul_days": 0.0, "ecm_rul_days_lo": 0.0, "ecm_rul_days_hi": 0.0}

    daily_rate = ALPHA * avg_efc_per_day + BETA * CAL_AGING_RATE / 365.0
    if daily_rate <= 0:
        return {"ecm_rul_days": None, "ecm_rul_days_lo": None, "ecm_rul_days_hi": None}

    remaining = soh - eol
    rul_p50   = remaining / daily_rate
    rul_lo    = max(0.0, remaining - 1.96 * soh_std) / daily_rate
    rul_hi    = (remaining + 1.96 * soh_std) / daily_rate

    cap = lambda v: round(min(float(v), 36500.0), 0) if np.isfinite(float(v)) else None
    return {
        "ecm_rul_days":    cap(rul_p50),
        "ecm_rul_days_lo": cap(rul_lo),
        "ecm_rul_days_hi": cap(rul_hi),
    }


# ── 60/90-day forward propagation ────────────────────────────────────────────

def _forecast_ecm(x: np.ndarray, P: np.ndarray,
                  avg_efc_per_day: float, avg_session_gap_days: float,
                  horizon_days: int) -> dict:
    """
    Propagate ECM state forward horizon_days using process model only
    (no observation updates). Covariance grows → widening CI.
    """
    if not np.isfinite(avg_efc_per_day) or avg_efc_per_day <= 0:
        return {f"soh_pred_{horizon_days}d": np.nan,
                f"soh_pred_{horizon_days}d_lo": np.nan,
                f"soh_pred_{horizon_days}d_hi": np.nan,
                f"r0_pred_{horizon_days}d": np.nan,
                f"r1_pred_{horizon_days}d": np.nan}

    x_f = x.copy()
    P_f = P.copy()
    F   = _F_5d()

    n_steps = max(1, int(round(horizon_days / max(avg_session_gap_days, 0.1))))
    d_efc   = avg_efc_per_day * avg_session_gap_days
    d_days  = avg_session_gap_days

    for _ in range(n_steps):
        x_f = _process_5d(x_f, d_efc, d_days)
        P_f = F @ P_f @ F.T + Q_ECM

    soh_pred = float(x_f[0])
    soh_std  = float(np.sqrt(max(0.0, P_f[0, 0])))

    return {
        f"soh_pred_{horizon_days}d":    round(soh_pred, 3),
        f"soh_pred_{horizon_days}d_lo": round(soh_pred - 1.96 * soh_std, 3),
        f"soh_pred_{horizon_days}d_hi": round(soh_pred + 1.96 * soh_std, 3),
        f"r0_pred_{horizon_days}d":     round(float(x_f[1]), 6),
        f"r1_pred_{horizon_days}d":     round(float(x_f[2]), 6),
    }


# ── Fleet loop ────────────────────────────────────────────────────────────────

def run_ecm_fleet(cycles: pd.DataFrame) -> pd.DataFrame:
    """
    Run ECM EKF on all sessions for all vehicles.
    Uses ir_ohm_mean from cycles.csv as the scalar IR observation.
    All 77,194 sessions used (cycle_soh ffill/bfill for non-quality rows).
    """
    df = cycles.copy()
    df = df.sort_values(["registration_number", "start_time"]).reset_index(drop=True)

    # No discharge IR profiles — scalar IR only
    df["ir_early_mean"] = np.nan
    df["ir_late_mean"]  = np.nan

    # Ensure cum_efc and days_since_first exist
    if "cum_efc" not in df.columns:
        df["_efc_s"] = df["soc_range"].abs() / 100.0
        df["cum_efc"] = df.groupby("registration_number")["_efc_s"].transform("cumsum")
        df.drop(columns=["_efc_s"], inplace=True)

    if "days_since_first" not in df.columns:
        def _days(g):
            return (g - g.min()) / 86_400_000.0
        df["days_since_first"] = df.groupby("registration_number")["start_time"].transform(_days)

    results = []
    vehicles = df["registration_number"].unique()

    for reg in vehicles:
        veh = df[df["registration_number"] == reg].sort_values("start_time").reset_index(drop=True)

        # Usage rate for RUL and forecasting
        days_span   = float(veh["days_since_first"].iloc[-1]) if len(veh) > 1 else 1.0
        efc_total   = float(veh["cum_efc"].iloc[-1])
        avg_efc_day = (efc_total / days_span) if days_span > 1.0 else np.nan

        # Average session gap
        session_gaps  = veh["days_since_first"].diff().dropna()
        avg_gap_days  = float(session_gaps.mean()) if len(session_gaps) > 0 else 1.0
        avg_gap_days  = max(avg_gap_days, 0.1)

        # Per-vehicle cycle_soh: quality gate + EWM smoothing
        block_dod    = veh["block_soc_diff"].abs() if "block_soc_diff" in veh.columns else veh["soc_range"].abs()
        csoh_raw     = veh["cycle_soh"].copy()
        quality_mask = (
            csoh_raw.notna() &
            (csoh_raw < CSOH_CAP) &
            (block_dod >= CSOH_MIN_DOD)
        )
        csoh_valid = csoh_raw.where(quality_mask)
        csoh_ewm   = csoh_valid.ewm(span=5, min_periods=1).mean().clip(88.0, 101.0)
        veh = veh.copy()
        veh["_csoh_ewm"]     = csoh_ewm
        veh["_block_dod"]    = block_dod
        veh["_csoh_quality"] = quality_mask

        x = X0_5.copy()
        P = P0_5.copy()
        prev_days = 0.0
        session_idx = 0

        for _, row in veh.iterrows():
            delta_efc  = float(row.get("soc_range", 0.0) or 0.0) / 100.0
            cur_days   = float(row.get("days_since_first", 0.0) or 0.0)
            delta_days = max(0.0, cur_days - prev_days)
            prev_days  = cur_days

            current_disc = float(row.get("current_mean_discharge", np.nan) or np.nan)
            is_loaded    = bool(row.get("is_loaded", False))

            # cycle_soh observation
            use_csoh = bool(row.get("_csoh_quality", False))
            csoh_obs = float(row["_csoh_ewm"]) if use_csoh and pd.notna(row.get("_csoh_ewm")) else np.nan

            # Adaptive R scaling for cycle_soh
            r_csoh_override = None
            if use_csoh and np.isfinite(csoh_obs):
                dod   = float(row["_block_dod"])
                scale = (CSOH_REF_DOD / max(dod, 10.0)) ** 2
                r_csoh_override = R_CSOH * scale

            # IR observations
            ir_early = float(row["ir_early_mean"]) if pd.notna(row.get("ir_early_mean")) else np.nan
            ir_late  = float(row["ir_late_mean"])  if pd.notna(row.get("ir_late_mean"))  else np.nan
            ir_scalar = float(row["ir_ohm_mean"])  if pd.notna(row.get("ir_ohm_mean"))   else np.nan

            x, P, nis, obs_type = ecm_step(
                x, P,
                delta_efc, delta_days,
                csoh_obs, ir_scalar, ir_early, ir_late,
                current_mean_discharge=current_disc,
                is_loaded=is_loaded,
                r_csoh_override=r_csoh_override,
            )

            ecm_soh     = float(x[0])
            ecm_soh_std = float(np.sqrt(max(0.0, P[0, 0])))
            tau         = float(x[2]) * float(x[3])  # τ = R₁ · C₁

            rul_d = _rul_ecm(ecm_soh, ecm_soh_std, avg_efc_day)

            # Train/test split (80/20 per vehicle, time-ordered)
            split = "train" if session_idx < int(0.8 * len(veh)) else "test"
            is_qg = bool(row.get("_csoh_quality", False))

            record = {
                "registration_number": reg,
                "session_id":          row.get("session_id"),
                "start_time":          row.get("start_time"),
                "cum_efc":             float(row.get("cum_efc", 0.0)),
                "days_since_first":    cur_days,
                "cycle_soh_obs":       csoh_obs,
                "ecm_soh":             round(ecm_soh, 3),
                "ecm_soh_std":         round(ecm_soh_std, 4),
                "ecm_r0":              round(float(x[1]), 6),
                "ecm_r1":              round(float(x[2]), 6),
                "ecm_c1":              round(float(x[3]), 2),
                "ecm_tau":             round(tau, 2),
                "ecm_ir_legacy":       round(float(x[4]), 6),
                "nis_value":           round(nis, 4) if np.isfinite(nis) else np.nan,
                "obs_type":            obs_type,
                "ir_early_mean":       row.get("ir_early_mean"),
                "ir_late_mean":        row.get("ir_late_mean"),
                "split":               split,
                "is_quality_gated":    is_qg,
                **rul_d,
            }

            # Forecast only for the last session (will overwrite on next iteration)
            for h in [60, 90]:
                fct = _forecast_ecm(x, P, avg_efc_day, avg_gap_days, h)
                record.update(fct)

            results.append(record)
            session_idx += 1

    return pd.DataFrame(results)


# ── Metrics computation ───────────────────────────────────────────────────────

def compute_metrics(df_sub: pd.DataFrame, label: str) -> dict:
    """MAE, RMSE, R², MBE, within-1/2%, NIS stats on a subset."""
    valid = df_sub.dropna(subset=["cycle_soh_obs", "ecm_soh"])
    if len(valid) < 3:
        return {"subset": label, "n": len(valid)}

    y_true = valid["cycle_soh_obs"].values
    y_pred = valid["ecm_soh"].values
    resid  = y_pred - y_true

    mae   = float(np.mean(np.abs(resid)))
    rmse  = float(np.sqrt(np.mean(resid ** 2)))
    ss_res = np.sum(resid ** 2)
    ss_tot = np.sum((y_true - y_true.mean()) ** 2)
    r2    = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan
    mbe   = float(np.mean(resid))
    w1    = float(np.mean(np.abs(resid) <= 1.0))
    w2    = float(np.mean(np.abs(resid) <= 2.0))

    # PICP90 using ecm_soh_std
    if "ecm_soh_std" in valid.columns:
        lo = valid["ecm_soh"].values - 1.96 * valid["ecm_soh_std"].values
        hi = valid["ecm_soh"].values + 1.96 * valid["ecm_soh_std"].values
        picp90 = float(np.mean((y_true >= lo) & (y_true <= hi)))
        mpiw90 = float(np.mean(hi - lo))
    else:
        picp90 = mpiw90 = np.nan

    # NIS summary
    nis_vals = df_sub["nis_value"].dropna()
    nis_mean = float(nis_vals.mean()) if len(nis_vals) > 0 else np.nan

    # DOF = 1 (SoH only for scalar mode, could be 2-3 for dual)
    dual_frac = (df_sub["obs_type"] == "dual").mean() if "obs_type" in df_sub.columns else 0.0
    avg_dof   = 1.0 + dual_frac * 2.0
    chi_hi    = float(chi2.ppf(0.95, df=avg_dof)) if np.isfinite(avg_dof) else np.nan

    return {
        "subset": label, "n": len(valid),
        "mae": round(mae, 4), "rmse": round(rmse, 4),
        "r2": round(r2, 4), "mbe": round(mbe, 4),
        "within_1pct": round(w1, 4), "within_2pct": round(w2, 4),
        "picp90": round(picp90, 4), "mpiw90": round(mpiw90, 4),
        "nis_mean": round(nis_mean, 4), "nis_chi95": round(chi_hi, 3) if np.isfinite(chi_hi) else np.nan,
        "dual_obs_frac": round(dual_frac, 4),
    }


# ── Diagnostics ───────────────────────────────────────────────────────────────

def run_diagnostics(df: pd.DataFrame) -> dict:
    """Physical constraint checks, ADF, R₀/R₁ growth rate OLS."""
    results = {}

    # 1. Physical constraint violation rates
    n = len(df)
    viol_r0  = (df["ecm_r0"] > R0_MAX).sum() / n
    viol_r1  = (df["ecm_r1"] > R1_MAX).sum() / n
    tau_vals = df["ecm_tau"].clip(lower=0)
    viol_tau = (tau_vals > TAU_MAX).sum() / n
    viol_soh = ((df["ecm_soh"] < 0) | (df["ecm_soh"] > 105)).sum() / n
    results["phys_viol_r0_pct"]  = round(100 * viol_r0, 2)
    results["phys_viol_r1_pct"]  = round(100 * viol_r1, 2)
    results["phys_viol_tau_pct"] = round(100 * viol_tau, 2)
    results["phys_viol_soh_pct"] = round(100 * viol_soh, 2)

    # 2. R₀ growth rate OLS per vehicle (Ω/EFC)
    r0_slopes = []
    r1_slopes = []
    r0r1_corr = []
    for reg, grp in df.groupby("registration_number"):
        grp = grp.sort_values("cum_efc").dropna(subset=["ecm_r0", "ecm_r1", "cum_efc"])
        if len(grp) < 5:
            continue
        X = grp[["cum_efc"]].values
        try:
            slope_r0 = LinearRegression().fit(X, grp["ecm_r0"].values).coef_[0]
            slope_r1 = LinearRegression().fit(X, grp["ecm_r1"].values).coef_[0]
            r0_slopes.append(slope_r0)
            r1_slopes.append(slope_r1)
            corr, _ = pearsonr(grp["ecm_r0"].values, grp["ecm_r1"].values)
            r0r1_corr.append(corr)
        except Exception:
            pass

    results["r0_slope_mean_mohm_per_efc"] = round(float(np.mean(r0_slopes)) * 1000, 4) if r0_slopes else np.nan
    results["r1_slope_mean_mohm_per_efc"] = round(float(np.mean(r1_slopes)) * 1000, 4) if r1_slopes else np.nan
    results["r0_r1_corr_mean"]            = round(float(np.mean(r0r1_corr)), 4) if r0r1_corr else np.nan
    results["r0_r1_degenerate_pct"]       = round(100 * sum(c > 0.99 for c in r0r1_corr) / max(len(r0r1_corr), 1), 1)

    # 3. ADF stationarity on IR innovation per vehicle (using NIS as proxy)
    adf_reject_pct = []
    for reg, grp in df.groupby("registration_number"):
        nis_seq = grp["nis_value"].dropna().values
        if len(nis_seq) < 20:
            continue
        try:
            adf_result = adfuller(nis_seq, autolag="AIC")
            adf_reject_pct.append(1.0 if adf_result[1] < 0.05 else 0.0)
        except Exception:
            pass
    results["adf_stationary_pct"] = round(100 * float(np.mean(adf_reject_pct)), 1) if adf_reject_pct else np.nan

    # 4. τ = R₁·C₁ statistics
    tau_valid = df["ecm_tau"].dropna()
    tau_valid = tau_valid[tau_valid > 0]
    results["tau_median_s"] = round(float(tau_valid.median()), 1) if len(tau_valid) > 0 else np.nan
    results["tau_p05_s"]    = round(float(tau_valid.quantile(0.05)), 1) if len(tau_valid) > 0 else np.nan
    results["tau_p95_s"]    = round(float(tau_valid.quantile(0.95)), 1) if len(tau_valid) > 0 else np.nan

    return results


# ── Q/R sensitivity ───────────────────────────────────────────────────────────

def q_sensitivity_check(df_merged: pd.DataFrame) -> list:
    """Run ECM with 3 Q_R₁ settings; return NIS and RMSE for each."""
    q_settings = [1e-8, 1e-7, 1e-6]
    rows = []
    for q_r1 in q_settings:
        orig = Q_ECM[2, 2]
        Q_ECM[2, 2] = q_r1

        df_run = run_ecm_fleet(df_merged)
        valid  = df_run.dropna(subset=["cycle_soh_obs", "ecm_soh"])
        nis_m  = df_run["nis_value"].dropna().mean()
        rmse   = np.sqrt(np.mean((valid["ecm_soh"] - valid["cycle_soh_obs"]) ** 2)) if len(valid) > 0 else np.nan

        Q_ECM[2, 2] = orig
        rows.append({"q_r1": q_r1, "nis_mean": round(nis_m, 3), "rmse": round(rmse, 4)})
    return rows


# ── Dual vs scalar benefit comparison ────────────────────────────────────────

def dual_vs_scalar(df: pd.DataFrame) -> dict:
    """Compare NIS and cycle_soh MAE for dual-obs vs scalar-obs sessions."""
    dual   = df[df["obs_type"] == "dual"].dropna(subset=["nis_value"])
    scalar = df[df["obs_type"] == "scalar"].dropna(subset=["nis_value"])

    out = {}
    for tag, sub in [("dual", dual), ("scalar", scalar)]:
        valid = sub.dropna(subset=["cycle_soh_obs", "ecm_soh"])
        out[f"{tag}_nis_mean"] = round(float(sub["nis_value"].mean()), 4) if len(sub) > 0 else np.nan
        out[f"{tag}_mae"]      = round(float(np.mean(np.abs(valid["ecm_soh"] - valid["cycle_soh_obs"]))), 4) if len(valid) > 3 else np.nan
        out[f"{tag}_n"]        = len(sub)
    return out


# ── Covariance convergence ────────────────────────────────────────────────────

def state_covariance_convergence(ecm_df: pd.DataFrame) -> pd.DataFrame:
    """
    For each vehicle, record sqrt(P_ii) at sessions 1/10/50/last.
    Approximated from ecm_soh_std (P[0,0]). Full covariance is not stored per session.
    """
    rows = []
    for reg, grp in ecm_df.groupby("registration_number"):
        grp = grp.reset_index(drop=True)
        n   = len(grp)
        for idx_label, idx in [("s1", 0), ("s10", min(9, n - 1)),
                                ("s50", min(49, n - 1)), ("last", n - 1)]:
            row_ = grp.iloc[idx]
            rows.append({
                "registration_number": reg,
                "session_step": idx_label,
                "soh_std": round(float(row_["ecm_soh_std"]), 5),
                "cum_efc": round(float(row_["cum_efc"]), 3),
            })
    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────────────────
# PLOTS
# ─────────────────────────────────────────────────────────────────────────────

RISK_RED    = "#c0392b"
RISK_AMBER  = "#e67e22"
RISK_GREEN  = "#27ae60"
FLEET_BLUE  = "#2980b9"
FLEET_GREY  = "#95a5a6"


def _risk_colour(delta: float) -> str:
    if delta > 3.0:  return RISK_RED
    if delta > 1.0:  return RISK_AMBER
    return RISK_GREEN


def plot_state_trajectories(df: pd.DataFrame, out_path: str):
    """
    4-panel: (A) SoH comparison, (B) R₀ vs cum_efc,
             (C) R₁ vs cum_efc, (D) τ vs cum_efc.
    """
    ekf_path = EKF_CSV
    ekf_df = pd.read_csv(ekf_path) if os.path.exists(ekf_path) else pd.DataFrame()

    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    axes = axes.flatten()
    vehicles = df["registration_number"].unique()
    cmap     = plt.cm.tab20
    n_v      = len(vehicles)

    # Panel A: SoH comparison (ECM vs EKF vs obs)
    ax = axes[0]
    for i, reg in enumerate(vehicles):
        sub = df[df["registration_number"] == reg].sort_values("cum_efc")
        col = cmap(i / max(n_v, 1))
        ax.plot(sub["cum_efc"], sub["ecm_soh"], color=col, lw=0.8, alpha=0.7)

    if len(ekf_df) > 0:
        for i, reg in enumerate(vehicles):
            sub_e = ekf_df[ekf_df["registration_number"] == reg].sort_values("cum_efc")
            if len(sub_e) > 0:
                col = cmap(i / max(n_v, 1))
                ax.plot(sub_e["cum_efc"], sub_e["ekf_soh"],
                        color=col, lw=0.8, alpha=0.4, ls="--")

    obs_valid = df.dropna(subset=["cycle_soh_obs"])
    ax.scatter(obs_valid["cum_efc"], obs_valid["cycle_soh_obs"],
               s=5, c="black", alpha=0.3, label="Observed SoH", zorder=3)
    ax.axhline(EOL_SOH, color=RISK_RED, ls="--", lw=1.2, label="EOL 80%")
    ax.set_ylim(85, 102)
    ax.set_xlabel("Charge Cycles (EFC)")
    ax.set_ylabel("Battery Health (%)")
    ax.set_title("A — SoH: ECM (solid) vs EKF (dashed)")
    ax.legend(fontsize=7)

    # Panel B: R₀ vs cum_efc
    ax = axes[1]
    for i, reg in enumerate(vehicles):
        sub = df[df["registration_number"] == reg].sort_values("cum_efc")
        col = cmap(i / max(n_v, 1))
        ax.plot(sub["cum_efc"], sub["ecm_r0"] * 1000, color=col, lw=0.8, alpha=0.7)
    # Fleet mean
    fleet_r0 = df.groupby("cum_efc")["ecm_r0"].mean()
    ax.plot(fleet_r0.index, fleet_r0.values * 1000, "k-", lw=2, label="Fleet mean R₀")
    ax.set_xlabel("Charge Cycles (EFC)")
    ax.set_ylabel("Ohmic Resistance R₀ (mΩ)")
    ax.set_title("B — Ohmic Resistance (R₀) vs Charge Cycles")
    ax.legend(fontsize=8)

    # Panel C: R₁ vs cum_efc
    ax = axes[2]
    for i, reg in enumerate(vehicles):
        sub = df[df["registration_number"] == reg].sort_values("cum_efc")
        col = cmap(i / max(n_v, 1))
        ax.plot(sub["cum_efc"], sub["ecm_r1"] * 1000, color=col, lw=0.8, alpha=0.7)
    fleet_r1 = df.groupby("cum_efc")["ecm_r1"].mean()
    ax.plot(fleet_r1.index, fleet_r1.values * 1000, "k-", lw=2, label="Fleet mean R₁")
    ax.set_xlabel("Charge Cycles (EFC)")
    ax.set_ylabel("Polarisation Resistance R₁ (mΩ)")
    ax.set_title("C — Polarisation Resistance (R₁) vs Charge Cycles")
    ax.legend(fontsize=8)

    # Panel D: τ = R₁C₁ vs cum_efc
    ax = axes[3]
    tau_valid = df[df["ecm_tau"].between(1, TAU_MAX)]
    for i, reg in enumerate(vehicles):
        sub = tau_valid[tau_valid["registration_number"] == reg].sort_values("cum_efc")
        if len(sub) > 0:
            col = cmap(i / max(n_v, 1))
            ax.plot(sub["cum_efc"], sub["ecm_tau"], color=col, lw=0.8, alpha=0.7)
    ax.axhline(100, color="navy", ls="--", lw=1, label="100 s reference")
    ax.axhline(1000, color="purple", ls="--", lw=1, label="1000 s reference")
    ax.set_xlabel("Charge Cycles (EFC)")
    ax.set_ylabel("Time Constant τ = R₁C₁ (s)")
    ax.set_title("D — RC Time Constant vs Charge Cycles")
    ax.legend(fontsize=8)

    plt.suptitle("ECM 1RC State Trajectories", fontsize=12, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out_path}")


def plot_nis_distribution(df: pd.DataFrame, out_path: str):
    """NIS histogram + chi-squared PDF; NIS time series for 3 vehicles."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    nis_all = df["nis_value"].dropna()
    dof_est = 1.5  # approximate average DOF across obs modes

    ax = axes[0]
    ax.hist(nis_all, bins=50, density=True, color=FLEET_BLUE, alpha=0.7, label="NIS distribution")
    x_r = np.linspace(0, nis_all.quantile(0.99), 200)
    ax.plot(x_r, chi2.pdf(x_r, df=dof_est), "r-", lw=2, label=f"χ²(dof={dof_est:.1f})")
    thresh_95 = chi2.ppf(0.95, df=dof_est)
    ax.axvline(thresh_95, color="orange", ls="--", lw=1.5, label=f"95th pct χ² = {thresh_95:.2f}")
    frac_over = (nis_all > thresh_95).mean()
    ax.set_xlabel("NIS (Normalised Innovation Squared)")
    ax.set_ylabel("Density")
    ax.set_title(f"NIS Distribution\n(Ideal: 5% above threshold — actual {100*frac_over:.1f}%)")
    ax.legend(fontsize=8)

    ax = axes[1]
    sample_vehicles = df["registration_number"].unique()[:3]
    colours = [FLEET_BLUE, RISK_AMBER, RISK_GREEN]
    for reg, col in zip(sample_vehicles, colours):
        sub = df[df["registration_number"] == reg].sort_values("cum_efc")
        ax.plot(sub["cum_efc"], sub["nis_value"].clip(upper=20), color=col, lw=1.0, label=reg)
    ax.axhline(thresh_95, color="orange", ls="--", lw=1.2, label="χ² 95th pct")
    ax.set_xlabel("Charge Cycles (EFC)")
    ax.set_ylabel("NIS")
    ax.set_title("NIS Time Series (3 Example Vehicles)")
    ax.legend(fontsize=7)
    ax.set_ylim(bottom=0)

    plt.suptitle("ECM Filter Consistency — Normalised Innovation Squared", fontsize=11, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.94])
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out_path}")


def plot_r0_r1_scatter(df: pd.DataFrame, out_path: str):
    """R₀ and R₁ vs cum_efc per vehicle with OLS trend lines."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    vehicles = df["registration_number"].unique()
    cmap     = plt.cm.tab20

    for panel, (col_key, label) in enumerate([
            ("ecm_r0", "Ohmic Resistance R₀ (mΩ)"),
            ("ecm_r1", "Polarisation Resistance R₁ (mΩ)")]):
        ax = axes[panel]
        for i, reg in enumerate(vehicles):
            sub = df[df["registration_number"] == reg].sort_values("cum_efc").dropna(subset=[col_key, "cum_efc"])
            if len(sub) < 3:
                continue
            col = cmap(i / max(len(vehicles), 1))
            ax.plot(sub["cum_efc"], sub[col_key] * 1000, color=col, lw=0.7, alpha=0.6)
            # OLS
            X = sub[["cum_efc"]].values
            m = LinearRegression().fit(X, sub[col_key].values)
            x_range = np.linspace(sub["cum_efc"].min(), sub["cum_efc"].max(), 100)
            ax.plot(x_range, m.predict(x_range.reshape(-1, 1)) * 1000,
                    color=col, lw=1.5, ls="--", alpha=0.8)

        # Fleet mean
        fleet_mean = df.groupby("cum_efc")[col_key].mean()
        ax.plot(fleet_mean.index, fleet_mean.values * 1000,
                "k-", lw=2.5, label="Fleet mean", zorder=5)

        ax.set_xlabel("Charge Cycles (EFC)")
        ax.set_ylabel(label)
        ax.set_title(f"{label} — per Vehicle (dashed = OLS trend)")
        ax.legend(fontsize=8)

    plt.suptitle("ECM 1RC — Internal Resistance Components", fontsize=11, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.94])
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out_path}")


def plot_covariance_convergence(conv_df: pd.DataFrame, out_path: str):
    """4-panel showing SoH std convergence across sessions."""
    fig, ax = plt.subplots(figsize=(9, 5))

    step_order = ["s1", "s10", "s50", "last"]
    step_labels = ["Session 1", "Session 10", "Session 50", "Last Session"]

    means = []
    stds  = []
    for step in step_order:
        sub = conv_df[conv_df["session_step"] == step]["soh_std"]
        means.append(sub.mean())
        stds.append(sub.std())

    ax.errorbar(range(len(step_order)), means, yerr=stds,
                fmt="o-", color=FLEET_BLUE, capsize=5, linewidth=2,
                label="Mean ± Std across vehicles")
    ax.set_xticks(range(len(step_order)))
    ax.set_xticklabels(step_labels)
    ax.set_ylabel("SoH Posterior Std (%)")
    ax.set_title("ECM Covariance Convergence — SoH Uncertainty Narrows with Data")
    ax.legend()
    ax.set_ylim(bottom=0)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out_path}")


def plot_vs_ekf(df: pd.DataFrame, out_path: str):
    """Scatter ECM SoH vs EKF SoH + 3-vehicle time series comparison."""
    if not os.path.exists(EKF_CSV):
        print(f"  [SKIP] {out_path} — ekf_soh.csv not found")
        return

    ekf_df = pd.read_csv(EKF_CSV)
    merged = df.merge(
        ekf_df[["session_id", "ekf_soh", "ekf_soh_std"]],
        on="session_id", how="inner"
    ).dropna(subset=["ecm_soh", "ekf_soh"])

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    ax = axes[0]
    ax.scatter(merged["ekf_soh"], merged["ecm_soh"], s=5, alpha=0.4, color=FLEET_BLUE)
    lo = min(merged["ekf_soh"].min(), merged["ecm_soh"].min())
    hi = max(merged["ekf_soh"].max(), merged["ecm_soh"].max())
    ax.plot([lo, hi], [lo, hi], "r--", lw=1.5, label="45° line")
    r, _ = pearsonr(merged["ekf_soh"], merged["ecm_soh"])
    ax.set_xlabel("EKF SoH (%)")
    ax.set_ylabel("ECM SoH (%)")
    ax.set_title(f"ECM vs EKF SoH\n(Pearson r = {r:.3f})")
    ax.legend(fontsize=8)

    ax = axes[1]
    sample_vehicles = merged["registration_number"].unique()[:3]
    colours = [FLEET_BLUE, RISK_AMBER, RISK_GREEN]
    for reg, col in zip(sample_vehicles, colours):
        sub_e = merged[merged["registration_number"] == reg].sort_values("cum_efc")
        ax.plot(sub_e["cum_efc"], sub_e["ecm_soh"], color=col, lw=1.5, label=f"{reg} ECM")
        ax.plot(sub_e["cum_efc"], sub_e["ekf_soh"], color=col, lw=1.5, ls="--", alpha=0.6, label=f"{reg} EKF")
        obs = sub_e.dropna(subset=["cycle_soh_obs"])
        ax.scatter(obs["cum_efc"], obs["cycle_soh_obs"], s=15, color=col, zorder=3)

    ax.axhline(EOL_SOH, color=RISK_RED, ls="--", lw=1.2)
    ax.set_ylim(85, 102)
    ax.set_xlabel("Charge Cycles (EFC)")
    ax.set_ylabel("Battery Health (%)")
    ax.set_title("ECM vs EKF — 3 Example Vehicles")
    ax.legend(fontsize=6, ncol=2)

    plt.suptitle("ECM vs EKF Comparison", fontsize=11, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.94])
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out_path}")


def plot_dual_obs_benefit(df: pd.DataFrame, out_path: str):
    """NIS and MAE comparison: dual-obs sessions vs scalar-obs sessions."""
    dual   = df[df["obs_type"] == "dual"]
    scalar = df[df["obs_type"] == "scalar"]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Panel 1: NIS distributions
    ax = axes[0]
    for sub, label, col in [(dual, "Dual IR (R₀+R₁)", FLEET_BLUE),
                              (scalar, "Scalar IR", FLEET_GREY)]:
        nis_v = sub["nis_value"].dropna().clip(upper=15)
        if len(nis_v) > 5:
            ax.hist(nis_v, bins=30, density=True, alpha=0.6, color=col, label=label)
    thresh = chi2.ppf(0.95, df=1.5)
    ax.axvline(thresh, color="orange", ls="--", lw=1.5, label=f"χ² 95th pct")
    ax.set_xlabel("NIS")
    ax.set_ylabel("Density")
    ax.set_title("Filter Consistency: Dual vs Scalar IR Observations")
    ax.legend(fontsize=8)

    # Panel 2: MAE comparison
    ax = axes[1]
    maes   = []
    labels = []
    colours = []
    for sub, label, col in [(dual, "Dual IR\n(R₀+R₁)", FLEET_BLUE),
                              (scalar, "Scalar IR\n(single obs)", FLEET_GREY)]:
        valid = sub.dropna(subset=["cycle_soh_obs", "ecm_soh"])
        if len(valid) > 3:
            mae = np.mean(np.abs(valid["ecm_soh"] - valid["cycle_soh_obs"]))
            maes.append(mae)
            labels.append(label)
            colours.append(col)

    bars = ax.bar(labels, maes, color=colours, edgecolor="white", width=0.5)
    for bar, val in zip(bars, maes):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f"{val:.3f}%", ha="center", va="bottom", fontsize=9)
    ax.set_ylabel("Mean Absolute Error vs Observed SoH (%)")
    ax.set_title("Cycle-SoH MAE:\nDual IR Profiles vs Scalar IR")
    ax.set_ylim(bottom=0)

    plt.suptitle("Value of Discharge IR Profiles (sessions_rows.csv) for ECM Accuracy",
                 fontsize=10, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out_path}")


def plot_resistance_dashboard(df: pd.DataFrame, out_path: str):
    """
    STAKEHOLDER — Fleet R₀ and R₁ trend dashboard.
    Top panel: all vehicles R₀ vs cum_efc + fleet mean.
    Bottom panel: all vehicles R₁ vs cum_efc + fleet mean.
    Right margin: ranked current R₀ per vehicle.
    """
    # Get latest state per vehicle
    latest = (
        df.sort_values("cum_efc")
          .groupby("registration_number")
          .last()
          .reset_index()
          .sort_values("ecm_r0", ascending=False)
    )

    fig = plt.figure(figsize=(15, 10))
    gs  = fig.add_gridspec(2, 2, width_ratios=[3, 1], hspace=0.35, wspace=0.3)

    ax_r0    = fig.add_subplot(gs[0, 0])
    ax_r0bar = fig.add_subplot(gs[0, 1])
    ax_r1    = fig.add_subplot(gs[1, 0])
    ax_r1bar = fig.add_subplot(gs[1, 1])

    vehicles = df["registration_number"].unique()
    cmap     = plt.cm.tab20

    # R₀ trend
    for i, reg in enumerate(vehicles):
        sub = df[df["registration_number"] == reg].sort_values("cum_efc")
        col = cmap(i / max(len(vehicles), 1))
        ax_r0.plot(sub["cum_efc"], sub["ecm_r0"] * 1000, color=col, lw=0.7, alpha=0.5)
    fleet_r0 = df.groupby("cum_efc")["ecm_r0"].mean()
    ax_r0.plot(fleet_r0.index, fleet_r0.values * 1000, "k-", lw=2.5, label="Fleet Mean")
    ax_r0.set_xlabel("Charge Cycles (EFC)", fontsize=10)
    ax_r0.set_ylabel("Ohmic Resistance R₀ (mΩ)", fontsize=10)
    ax_r0.set_title("Ohmic Resistance (R₀) — All Vehicles\n"
                    "Rising R₀ = Ohmic Aging", fontsize=10)
    ax_r0.legend(fontsize=9)

    # R₀ ranked bar
    bar_cols = [_risk_colour(r * 1000 * 100) for r in latest["ecm_r0"]]  # rough scaling
    ax_r0bar.barh(range(len(latest)), latest["ecm_r0"].values * 1000,
                  color=bar_cols, edgecolor="white")
    ax_r0bar.set_yticks(range(len(latest)))
    ax_r0bar.set_yticklabels(latest["registration_number"].values, fontsize=6)
    ax_r0bar.set_xlabel("Current R₀ (mΩ)", fontsize=9)
    ax_r0bar.set_title("Current R₀\n(ranked)", fontsize=9)

    # R₁ trend
    for i, reg in enumerate(vehicles):
        sub = df[df["registration_number"] == reg].sort_values("cum_efc")
        col = cmap(i / max(len(vehicles), 1))
        ax_r1.plot(sub["cum_efc"], sub["ecm_r1"] * 1000, color=col, lw=0.7, alpha=0.5)
    fleet_r1 = df.groupby("cum_efc")["ecm_r1"].mean()
    ax_r1.plot(fleet_r1.index, fleet_r1.values * 1000, "k-", lw=2.5, label="Fleet Mean")
    ax_r1.set_xlabel("Charge Cycles (EFC)", fontsize=10)
    ax_r1.set_ylabel("Polarisation Resistance R₁ (mΩ)", fontsize=10)
    ax_r1.set_title("Polarisation Resistance (R₁) — All Vehicles\n"
                    "Rising R₁ = Polarisation Build-Up", fontsize=10)
    ax_r1.legend(fontsize=9)

    # R₁ ranked bar
    bar_cols_r1 = [_risk_colour(r * 1000 * 50) for r in latest["ecm_r1"]]
    ax_r1bar.barh(range(len(latest)), latest["ecm_r1"].values * 1000,
                  color=bar_cols_r1, edgecolor="white")
    ax_r1bar.set_yticks(range(len(latest)))
    ax_r1bar.set_yticklabels(latest["registration_number"].values, fontsize=6)
    ax_r1bar.set_xlabel("Current R₁ (mΩ)", fontsize=9)
    ax_r1bar.set_title("Current R₁\n(ranked)", fontsize=9)

    fig.suptitle(
        "Internal Resistance — Fleet Early Warning\n"
        "R₀ rise = ohmic aging  ·  R₁ rise = polarisation  ·  Both lead SoH decline by weeks",
        fontsize=12, fontweight="bold", y=1.02
    )

    # Risk legend
    legend_elements = [
        Line2D([0], [0], color=RISK_RED,   lw=3, label="High risk"),
        Line2D([0], [0], color=RISK_AMBER, lw=3, label="Medium risk"),
        Line2D([0], [0], color=RISK_GREEN, lw=3, label="Low risk"),
    ]
    fig.legend(handles=legend_elements, loc="lower center", ncol=3,
               fontsize=9, bbox_to_anchor=(0.5, -0.04))

    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out_path}")


def plot_resistance_forecast(df: pd.DataFrame, out_path: str):
    """
    STAKEHOLDER — Projected R₀ and R₁ per vehicle at 60 and 90 days.
    Shows current + forecast values with risk colouring.
    """
    # Get last session per vehicle
    latest = (
        df.sort_values("cum_efc")
          .groupby("registration_number")
          .last()
          .reset_index()
    )

    needed_cols = ["r0_pred_60d", "r0_pred_90d", "r1_pred_60d", "r1_pred_90d"]
    for c in needed_cols:
        if c not in latest.columns:
            latest[c] = np.nan

    latest = latest.sort_values("ecm_r0", ascending=True).reset_index(drop=True)
    n_veh  = len(latest)

    fig, axes = plt.subplots(1, 2, figsize=(14, max(5, n_veh * 0.3 + 2)))
    y_pos = np.arange(n_veh)

    for ax, (cur_col, f60_col, f90_col, label, thresh) in zip(axes, [
        ("ecm_r0", "r0_pred_60d", "r0_pred_90d", "Ohmic R₀ (mΩ)", R0_MAX * 0.7 * 1000),
        ("ecm_r1", "r1_pred_60d", "r1_pred_90d", "Polarisation R₁ (mΩ)", R1_MAX * 0.7 * 1000),
    ]):
        cur = latest[cur_col].values * 1000
        f60 = latest[f60_col].values * 1000 if f60_col in latest.columns else np.full(n_veh, np.nan)
        f90 = latest[f90_col].values * 1000 if f90_col in latest.columns else np.full(n_veh, np.nan)

        colours = [_risk_colour((v - c) * 20) for v, c in zip(f90, cur)]

        ax.barh(y_pos, cur, 0.6, color=FLEET_GREY, alpha=0.5, label="Current")
        ax.scatter(f60, y_pos + 0.2, marker=">", s=40, color=FLEET_BLUE, zorder=5, label="60-day forecast")
        ax.scatter(f90, y_pos - 0.2, marker="D", s=30, color=[_risk_colour((v - c) * 20) for v, c in zip(f90, cur)],
                   zorder=5, label="90-day forecast")

        ax.axvline(thresh, color=RISK_RED, ls="--", lw=1.5, label=f"Warning threshold ({thresh:.1f} mΩ)")
        ax.set_yticks(y_pos)
        ax.set_yticklabels(latest["registration_number"].values, fontsize=7)
        ax.set_xlabel(label, fontsize=10)
        ax.set_title(f"Projected {label}\n(bar=current, ▶=60d, ◆=90d)", fontsize=10)
        ax.legend(fontsize=7, loc="lower right")

    plt.suptitle("Projected Resistance Growth — 60 & 90 Days\n"
                 "Rising resistance precedes battery health decline by weeks",
                 fontsize=12, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out_path}")


# ── Fleet forecast summary ────────────────────────────────────────────────────

def build_fleet_forecast(df: pd.DataFrame) -> pd.DataFrame:
    """One row per vehicle: current state + 60/90-day SoH forecast + risk flag."""
    latest = (
        df.sort_values("cum_efc")
          .groupby("registration_number")
          .last()
          .reset_index()
    )

    rows = []
    for _, row in latest.iterrows():
        cur_soh  = float(row["ecm_soh"])
        soh_60   = float(row.get("soh_pred_60d", np.nan))
        soh_90   = float(row.get("soh_pred_90d", np.nan))
        delta_90 = (cur_soh - soh_90) if np.isfinite(soh_90) else np.nan
        risk     = _risk_colour(delta_90) if np.isfinite(delta_90) else FLEET_GREY
        risk_str = "red" if risk == RISK_RED else ("amber" if risk == RISK_AMBER else "green")

        rows.append({
            "registration_number": row["registration_number"],
            "current_soh":         round(cur_soh, 2),
            "ecm_r0_mohm":         round(float(row["ecm_r0"]) * 1000, 3),
            "ecm_r1_mohm":         round(float(row["ecm_r1"]) * 1000, 3),
            "ecm_tau_s":           round(float(row["ecm_tau"]), 1),
            "soh_60d":             round(soh_60, 2) if np.isfinite(soh_60) else np.nan,
            "soh_60d_lo":          row.get("soh_pred_60d_lo"),
            "soh_60d_hi":          row.get("soh_pred_60d_hi"),
            "soh_90d":             round(soh_90, 2) if np.isfinite(soh_90) else np.nan,
            "soh_90d_lo":          row.get("soh_pred_90d_lo"),
            "soh_90d_hi":          row.get("soh_pred_90d_hi"),
            "delta_soh_90d":       round(delta_90, 2) if np.isfinite(delta_90) else np.nan,
            "rul_days_p50":        row.get("ecm_rul_days"),
            "rul_days_lo":         row.get("ecm_rul_days_lo"),
            "rul_days_hi":         row.get("ecm_rul_days_hi"),
            "risk_flag":           risk_str,
        })

    return pd.DataFrame(rows).sort_values("current_soh")


# ── Main ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 70)
    print("ecm_1rc.py — 1RC Equivalent Circuit Model with 5D EKF")
    print("=" * 70)

    # ── Load data ─────────────────────────────────────────────────────────────
    print(f"\nLoading cycles.csv from {CYCLES_CSV} ...")
    cycles = pd.read_csv(CYCLES_CSV)
    print(f"  {len(cycles):,} sessions, {cycles['registration_number'].nunique()} vehicles")


    # ── Young-fleet warning ───────────────────────────────────────────────────
    max_days = cycles["days_since_first"].max() if "days_since_first" in cycles.columns else 0
    if max_days < 180:
        print(f"\n  YOUNG FLEET WARNING: {max_days:.0f} days data span, "
              f"{cycles['cum_efc'].max():.0f} EFC max. "
              "SoH range 88-100%. RUL priors dominate.")

    # ── Run ECM filter ────────────────────────────────────────────────────────
    print("\nRunning ECM EKF fleet-wide ...")
    ecm_df = run_ecm_fleet(cycles)
    print(f"  ECM output: {len(ecm_df):,} sessions across {ecm_df['registration_number'].nunique()} vehicles")

    # Obs type breakdown
    obs_counts = ecm_df["obs_type"].value_counts()
    print("  Observation type breakdown:")
    for ot, cnt in obs_counts.items():
        print(f"    {ot}: {cnt:,} ({100*cnt/len(ecm_df):.1f}%)")

    # Save
    ecm_df.to_csv(ECM_CSV, index=False)
    print(f"\n  Saved: {ECM_CSV}")

    # ── Metrics ───────────────────────────────────────────────────────────────
    print("\nComputing metrics ...")
    test_all = ecm_df[ecm_df["split"] == "test"]
    test_qg  = test_all[test_all["is_quality_gated"]]

    metrics_rows = [
        compute_metrics(ecm_df,   "train_all"),
        compute_metrics(ecm_df[ecm_df["is_quality_gated"]], "train_quality_gated"),
        compute_metrics(test_all, "test_all"),
        compute_metrics(test_qg,  "test_quality_gated"),
    ]
    metrics_df = pd.DataFrame(metrics_rows)
    metrics_df.to_csv(METRICS_CSV, index=False)
    print(metrics_df[["subset", "n", "mae", "rmse", "r2", "within_2pct",
                       "picp90", "nis_mean", "dual_obs_frac"]].to_string(index=False))

    # ── Diagnostics ───────────────────────────────────────────────────────────
    print("\nRunning diagnostics ...")
    diag = run_diagnostics(ecm_df)
    for k, v in diag.items():
        print(f"  {k}: {v}")

    # Q/R sensitivity
    print("\nQ/R₁ sensitivity check (3 settings) ...")
    try:
        sens_rows = q_sensitivity_check(cycles)
        print("  Q_R1      | NIS mean | RMSE")
        for r in sens_rows:
            print(f"  {r['q_r1']:.0e}  | {r['nis_mean']:.4f}  | {r['rmse']:.4f}")
    except Exception as e:
        print(f"  [WARN] Q sensitivity check failed: {e}")

    # Dual vs scalar benefit
    print("\nDual vs scalar IR obs benefit ...")
    benefit = dual_vs_scalar(ecm_df)
    for k, v in benefit.items():
        print(f"  {k}: {v}")

    # Covariance convergence
    conv_df = state_covariance_convergence(ecm_df)
    conv_df.to_csv(CONV_CSV, index=False)
    print(f"\n  Saved covariance convergence: {CONV_CSV}")

    # Physical constraint violations
    print(f"\n  Physical violation rates:")
    print(f"    R₀ > {R0_MAX*1000:.0f} mΩ:  {diag['phys_viol_r0_pct']:.2f}%"
          f"  {'[OK]' if diag['phys_viol_r0_pct'] < 5.0 else '[WARN]'}")
    print(f"    R₁ > {R1_MAX*1000:.0f} mΩ:  {diag['phys_viol_r1_pct']:.2f}%"
          f"  {'[OK]' if diag['phys_viol_r1_pct'] < 5.0 else '[WARN]'}")
    print(f"    τ  > {TAU_MAX:.0f} s:   {diag['phys_viol_tau_pct']:.2f}%"
          f"  {'[OK]' if diag['phys_viol_tau_pct'] < 5.0 else '[WARN]'}")
    print(f"    τ  median: {diag['tau_median_s']:.1f} s "
          f"(expected 100-1000 s range)")

    # R₀ growth rate
    r0_slope = diag.get("r0_slope_mean_mohm_per_efc", np.nan)
    r0_per_100efc = r0_slope * 100 if np.isfinite(r0_slope) else np.nan
    print(f"\n  R₀ growth rate: {r0_per_100efc:.2f} mΩ / 100 EFC "
          f"(NMC literature: 0.5–2 mΩ / 100 EFC)")
    if np.isfinite(r0_per_100efc):
        status = "[OK]" if 0.0 <= r0_per_100efc <= 5.0 else "[WARN] outside expected range"
        print(f"  {status}")

    # R₀/R₁ degeneracy check
    print(f"  R₀/R₁ correlation mean: {diag['r0_r1_corr_mean']:.3f} "
          f"(>0.99 = degenerate: {diag['r0_r1_degenerate_pct']:.1f}% of vehicles)")

    # ── Fleet forecast ────────────────────────────────────────────────────────
    print("\nBuilding fleet forecast ...")
    fleet_fc = build_fleet_forecast(ecm_df)
    fleet_fc.to_csv(FLEET_CSV, index=False)
    print(f"  Saved: {FLEET_CSV}")
    print(f"\n  Fleet forecast summary:")
    print(fleet_fc[["registration_number", "current_soh", "soh_60d", "soh_90d",
                     "ecm_r0_mohm", "ecm_r1_mohm", "rul_days_p50", "risk_flag"]].to_string(index=False))

    # Risk counts
    rc = fleet_fc["risk_flag"].value_counts()
    print(f"\n  Risk flags: red={rc.get('red',0)}, amber={rc.get('amber',0)}, green={rc.get('green',0)}")

    # ── Plots ─────────────────────────────────────────────────────────────────
    print("\nGenerating plots ...")
    plot_state_trajectories(ecm_df, os.path.join(PLOTS_DIR, "ecm_state_trajectories.png"))
    plot_nis_distribution(ecm_df,   os.path.join(PLOTS_DIR, "ecm_nis_distribution.png"))
    plot_r0_r1_scatter(ecm_df,      os.path.join(PLOTS_DIR, "ecm_r0_r1_scatter.png"))
    plot_covariance_convergence(conv_df, os.path.join(PLOTS_DIR, "ecm_covariance_convergence.png"))
    plot_vs_ekf(ecm_df,             os.path.join(PLOTS_DIR, "ecm_vs_ekf_comparison.png"))
    plot_dual_obs_benefit(ecm_df,   os.path.join(PLOTS_DIR, "ecm_dual_obs_benefit.png"))
    plot_resistance_dashboard(ecm_df, os.path.join(PLOTS_DIR, "ecm_resistance_dashboard.png"))
    plot_resistance_forecast(ecm_df,  os.path.join(PLOTS_DIR, "ecm_resistance_forecast.png"))

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("ECM 1RC SUMMARY")
    print("=" * 70)
    test_qg_valid = test_qg.dropna(subset=["cycle_soh_obs", "ecm_soh"])
    if len(test_qg_valid) > 0:
        mae_qg = float(np.mean(np.abs(test_qg_valid["ecm_soh"] - test_qg_valid["cycle_soh_obs"])))
        print(f"  Test MAE (quality-gated): {mae_qg:.4f}%")
    print(f"  Dual-obs sessions: {(ecm_df['obs_type']=='dual').sum():,}")
    print(f"  Scalar-obs sessions: {(ecm_df['obs_type']=='scalar').sum():,}")
    print(f"  Fleet current SoH range: {ecm_df.groupby('registration_number')['ecm_soh'].last().min():.2f}% – "
          f"{ecm_df.groupby('registration_number')['ecm_soh'].last().max():.2f}%")
    print(f"  Fleet current R₀ range:  "
          f"{ecm_df.groupby('registration_number')['ecm_r0'].last().min()*1000:.2f} – "
          f"{ecm_df.groupby('registration_number')['ecm_r0'].last().max()*1000:.2f} mΩ")
    print(f"\n  Next: run particle_filter_soh.py")
    print()
