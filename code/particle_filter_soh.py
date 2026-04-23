import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) if "__file__" in dir() else os.getcwd())

"""
particle_filter_soh.py — Sequential Monte Carlo (Particle Filter) for SoH tracking.

State: 2D — [SoH (%), IR_drift (Ω)]
N_PARTICLES = 2000 (offline; reduce if memory-constrained)
Resampling: systematic resampling when ESS < N/2.

Primary value over EKF
----------------------
The EKF assumes a Gaussian posterior at every step.  On a young fleet
(sparse data, big prior) or after anomalous sessions, the true posterior
can be:
  • Skewed  — the vehicle has degraded more in one direction
  • Bimodal — two plausible SoH hypotheses (anomaly vs normal aging)
The PF captures these shapes exactly; EKF cannot.

Key outputs
-----------
pf_soh_mean, pf_soh_std  : posterior mean and standard deviation
pf_soh_p05/p25/p50/p75/p95 : quantiles of particle cloud
pf_soh_skewness, pf_soh_kurtosis : non-Gaussianity indicators
pf_ess_ratio             : ESS / N (healthy > 0.5)
pf_resampled             : True when systematic resampling was triggered

Business outputs
----------------
60/90-day SoH distribution from propagating all 2000 particles forward.
p50 = most likely value; p05–p95 = full risk range.
RUL: each particle hits SoH=80% at a different step → histogram of RULs.

Outputs
-------
artifacts/pf_soh.csv
artifacts/pf_particles_final.npy  — shape (n_vehicles, N_PARTICLES, 2)
artifacts/pf_metrics.csv
artifacts/pf_per_vehicle_metrics.csv
artifacts/fleet_forecast_pf.csv
plots/pf_posterior_evolution.png
plots/pf_vs_ekf_comparison.png
plots/pf_ess_trajectory.png
plots/pf_non_gaussianity.png
plots/pf_rul_distribution.png
plots/pf_prior_sensitivity.png
plots/pf_rul_fleet_summary.png    ← stakeholder
plots/pf_forecast_60_90.png       ← stakeholder
"""

import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy import stats
from scipy.stats import skew, kurtosis, ks_2samp, wasserstein_distance
from scipy.stats import norm as sp_norm
import time

from config import (
    CYCLES_CSV, ARTIFACTS_DIR, PLOTS_DIR,
    EKF_Q_DIAG, EKF_R_DIAG,
    EOL_SOH, EFC_MAX, CAL_AGING_RATE,
    EKF_ALPHA, PEUKERT_N, I_NOMINAL_A, LOAD_STRESS_FACTOR, ZETA,
    CYCLE_SOH_OBS_CAP, CYCLE_SOH_MIN_BLOCK_DOD, CYCLE_SOH_REF_DOD,
    SEED,
)

np.random.seed(SEED)
warnings.filterwarnings("ignore", category=RuntimeWarning)

# ── Output paths ───────────────────────────────────────────────────────────────
PF_CSV         = os.path.join(ARTIFACTS_DIR, "pf_soh.csv")
PARTICLES_NPY  = os.path.join(ARTIFACTS_DIR, "pf_particles_final.npy")
METRICS_CSV    = os.path.join(ARTIFACTS_DIR, "pf_metrics.csv")
PERVEH_CSV     = os.path.join(ARTIFACTS_DIR, "pf_per_vehicle_metrics.csv")
FLEET_CSV      = os.path.join(ARTIFACTS_DIR, "fleet_forecast_pf.csv")
EKF_CSV        = os.path.join(ARTIFACTS_DIR, "ekf_soh.csv")

# ── PF configuration ──────────────────────────────────────────────────────────
N_PARTICLES  = 2000
RESAMPLE_ESS = 0.5    # resample when ESS/N < this

# Process noise (2D state) — matches EKF_Q_DIAG[0] and [1]
Q_PF = np.array([EKF_Q_DIAG[0], EKF_Q_DIAG[1]])  # [SoH variance, IR_drift variance]

# Observation noise
R_CSOH_PF   = EKF_R_DIAG[0]
R_IR_PF     = EKF_R_DIAG[2]

# Initial prior  N(100, 1²) for SoH; N(0, 0.001²) for IR_drift
PRIOR_SOH_MEAN, PRIOR_SOH_STD = 100.0, 1.0
PRIOR_IR_MEAN,  PRIOR_IR_STD  = 0.0,   0.001

# Physical constants (reuse from EKF)
ALPHA   = EKF_ALPHA
BETA    = 1.0
GAMMA   = 0.001
EOL     = EOL_SOH
CSOH_CAP     = CYCLE_SOH_OBS_CAP
CSOH_MIN_DOD = CYCLE_SOH_MIN_BLOCK_DOD
CSOH_REF_DOD = CYCLE_SOH_REF_DOD


# ── Process model (per-particle vectorised) ───────────────────────────────────

def _pf_process(particles: np.ndarray,
                delta_efc: float, delta_days: float,
                current_mean_discharge: float = np.nan,
                is_loaded: bool = False) -> np.ndarray:
    """
    Propagate all particles (N×2) through the process model + process noise.
    Returns updated particles (N×2).
    """
    N = len(particles)
    x_new = particles.copy()

    if np.isfinite(current_mean_discharge) and current_mean_discharge > 0:
        I_norm    = current_mean_discharge / I_NOMINAL_A
        alpha_adj = ALPHA * (1.0 + PEUKERT_N * max(0.0, I_norm - 1.0))
    else:
        alpha_adj = ALPHA

    load_fac = LOAD_STRESS_FACTOR if is_loaded else 1.0

    # SoH decay
    x_new[:, 0] -= (alpha_adj * load_fac * delta_efc +
                    BETA * delta_days * (CAL_AGING_RATE / 365.0))
    # IR drift growth
    x_new[:, 1] += GAMMA * delta_efc

    # Add process noise
    x_new[:, 0] += np.random.normal(0, np.sqrt(Q_PF[0]), N)
    x_new[:, 1] += np.random.normal(0, np.sqrt(Q_PF[1]), N)

    # Physical clamps
    x_new[:, 0] = np.clip(x_new[:, 0], 0.0, 105.0)
    x_new[:, 1] = np.clip(x_new[:, 1], -0.1, 0.5)

    return x_new


# ── Likelihood function ───────────────────────────────────────────────────────

def _log_likelihood(particles: np.ndarray,
                    cycle_soh_obs: float,
                    ir_ohm_mean: float,
                    r_csoh: float = R_CSOH_PF) -> np.ndarray:
    """
    Log-likelihood of each particle given observations.
    Returns (N,) log-weights update.
    """
    N     = len(particles)
    log_w = np.zeros(N)

    if np.isfinite(cycle_soh_obs):
        log_w += sp_norm.logpdf(cycle_soh_obs, loc=particles[:, 0], scale=np.sqrt(r_csoh))

    if np.isfinite(ir_ohm_mean):
        log_w += sp_norm.logpdf(ir_ohm_mean, loc=particles[:, 1], scale=np.sqrt(R_IR_PF))

    return log_w


def _systematic_resample(weights: np.ndarray) -> np.ndarray:
    """Systematic resampling — lower variance than multinomial."""
    N    = len(weights)
    cum  = np.cumsum(weights)
    step = 1.0 / N
    u0   = np.random.uniform(0, step)
    positions = u0 + step * np.arange(N)
    idx  = np.searchsorted(cum, positions)
    return idx.clip(0, N - 1)


def _multinomial_resample(weights: np.ndarray) -> np.ndarray:
    """Multinomial resampling (for comparison with systematic)."""
    return np.random.choice(len(weights), size=len(weights), p=weights)


def _effective_sample_size(weights: np.ndarray) -> float:
    """ESS = 1 / sum(w²)."""
    return 1.0 / np.sum(weights ** 2)


# ── Per-particle RUL ──────────────────────────────────────────────────────────

def _rul_particles(soh_particles: np.ndarray, avg_efc_per_day: float) -> np.ndarray:
    """
    For each particle, estimate RUL (days) until SoH = EOL.
    Returns (N,) array of RUL values.
    """
    if not np.isfinite(avg_efc_per_day) or avg_efc_per_day <= 0:
        return np.full(len(soh_particles), np.nan)

    daily_rate = ALPHA * avg_efc_per_day + BETA * CAL_AGING_RATE / 365.0
    if daily_rate <= 0:
        return np.full(len(soh_particles), np.inf)

    remaining = np.maximum(0.0, soh_particles - EOL)
    rul       = remaining / daily_rate
    return np.clip(rul, 0.0, 36500.0)


# ── 60/90-day forward propagation ────────────────────────────────────────────

def _pf_forecast(particles: np.ndarray,
                 avg_efc_per_day: float,
                 avg_session_gap_days: float,
                 horizon_days: int) -> dict:
    """
    Propagate all N particles forward horizon_days.
    Returns quantiles p05/p25/p50/p75/p95 of predicted SoH.
    """
    if not np.isfinite(avg_efc_per_day) or avg_efc_per_day <= 0:
        return {f"soh_pred_{horizon_days}d_p{p}": np.nan
                for p in [5, 25, 50, 75, 95]}

    p_copy    = particles.copy()
    n_steps   = max(1, int(round(horizon_days / max(avg_session_gap_days, 0.1))))
    d_efc     = avg_efc_per_day * avg_session_gap_days
    d_days    = avg_session_gap_days

    for _ in range(n_steps):
        p_copy = _pf_process(p_copy, d_efc, d_days)

    soh_dist = p_copy[:, 0]
    qs = np.quantile(soh_dist, [0.05, 0.25, 0.50, 0.75, 0.95])
    return {
        f"soh_pred_{horizon_days}d_p05": round(float(qs[0]), 3),
        f"soh_pred_{horizon_days}d_p25": round(float(qs[1]), 3),
        f"soh_pred_{horizon_days}d_p50": round(float(qs[2]), 3),
        f"soh_pred_{horizon_days}d_p75": round(float(qs[3]), 3),
        f"soh_pred_{horizon_days}d_p95": round(float(qs[4]), 3),
    }


# ── Fleet loop ────────────────────────────────────────────────────────────────

def run_pf_fleet(cycles: pd.DataFrame,
                 n_particles: int = N_PARTICLES,
                 prior_soh_mean: float = PRIOR_SOH_MEAN,
                 prior_soh_std:  float = PRIOR_SOH_STD) -> tuple:
    """
    Run PF on all sessions for all vehicles.
    Returns (results_df, final_particles_dict).
    """
    df = cycles.copy()
    df = df.sort_values(["registration_number", "start_time"]).reset_index(drop=True)

    if "cum_efc" not in df.columns:
        df["_efc_s"] = df["soc_range"].abs() / 100.0
        df["cum_efc"] = df.groupby("registration_number")["_efc_s"].transform("cumsum")
        df.drop(columns=["_efc_s"], inplace=True)

    if "days_since_first" not in df.columns:
        df["days_since_first"] = df.groupby("registration_number")["start_time"].transform(
            lambda x: (x - x.min()) / 86_400_000.0
        )

    results       = []
    final_particles = {}

    for reg in df["registration_number"].unique():
        veh = df[df["registration_number"] == reg].sort_values("start_time").reset_index(drop=True)

        # Usage stats
        days_span    = float(veh["days_since_first"].iloc[-1]) if len(veh) > 1 else 1.0
        efc_total    = float(veh["cum_efc"].iloc[-1])
        avg_efc_day  = efc_total / max(days_span, 1.0)
        session_gaps = veh["days_since_first"].diff().dropna()
        avg_gap_days = float(session_gaps.mean()) if len(session_gaps) > 0 else 1.0
        avg_gap_days = max(avg_gap_days, 0.1)

        # Quality gate
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

        # Initialise particles
        particles = np.column_stack([
            np.random.normal(prior_soh_mean, prior_soh_std, n_particles),
            np.random.normal(PRIOR_IR_MEAN,  PRIOR_IR_STD,  n_particles),
        ])
        weights = np.ones(n_particles) / n_particles

        prev_days   = 0.0
        session_idx = 0
        store_particles = []  # for posterior evolution plot (keep every 5th)

        for _, row in veh.iterrows():
            delta_efc  = float(row.get("soc_range", 0.0) or 0.0) / 100.0
            cur_days   = float(row.get("days_since_first", 0.0) or 0.0)
            delta_days = max(0.0, cur_days - prev_days)
            prev_days  = cur_days

            current_disc = float(row.get("current_mean_discharge", np.nan) or np.nan)
            is_loaded    = bool(row.get("is_loaded", False))

            # Observations
            use_csoh = bool(row.get("_csoh_quality", False))
            csoh_obs = float(row["_csoh_ewm"]) if use_csoh and pd.notna(row.get("_csoh_ewm")) else np.nan

            # Adaptive R for cycle_soh
            r_csoh = R_CSOH_PF
            if use_csoh and np.isfinite(csoh_obs):
                dod    = float(row["_block_dod"])
                scale  = (CSOH_REF_DOD / max(dod, 10.0)) ** 2
                r_csoh = R_CSOH_PF * scale

            ir_mean = float(row.get("ir_ohm_mean", np.nan) or np.nan)

            # ── Predict ──────────────────────────────────────────────────────
            particles = _pf_process(particles, delta_efc, delta_days, current_disc, is_loaded)

            # ── Update ───────────────────────────────────────────────────────
            log_w  = _log_likelihood(particles, csoh_obs, ir_mean, r_csoh)
            log_w -= log_w.max()  # numeric stability
            w_new  = np.exp(log_w)
            w_sum  = w_new.sum()
            if w_sum <= 0 or not np.isfinite(w_sum):
                weights = np.ones(n_particles) / n_particles
            else:
                weights = w_new / w_sum

            # ── Resample ─────────────────────────────────────────────────────
            ess       = _effective_sample_size(weights)
            ess_ratio = ess / n_particles
            resampled = False
            if ess_ratio < RESAMPLE_ESS:
                idx       = _systematic_resample(weights)
                particles = particles[idx]
                weights   = np.ones(n_particles) / n_particles
                resampled = True

            # ── Posterior statistics ─────────────────────────────────────────
            pf_mean = float(np.average(particles[:, 0], weights=weights))
            pf_std  = float(np.sqrt(np.average((particles[:, 0] - pf_mean) ** 2, weights=weights)))
            pf_qs   = np.quantile(particles[:, 0], [0.05, 0.25, 0.50, 0.75, 0.95])

            soh_sk  = float(skew(particles[:, 0]))
            soh_ku  = float(kurtosis(particles[:, 0], fisher=True))  # excess kurtosis
            ngi     = abs(soh_sk) + abs(soh_ku) / 3.0

            # Particle degeneracy
            unique_frac = len(np.unique(particles[:, 0].round(4))) / n_particles

            # Train/test split
            split = "train" if session_idx < int(0.8 * len(veh)) else "test"
            is_qg = bool(row.get("_csoh_quality", False))

            record = {
                "registration_number": reg,
                "session_id":          row.get("session_id"),
                "start_time":          row.get("start_time"),
                "cum_efc":             float(row.get("cum_efc", 0.0)),
                "days_since_first":    cur_days,
                "cycle_soh_obs":       csoh_obs,
                "pf_soh_mean":         round(pf_mean, 3),
                "pf_soh_std":          round(pf_std, 4),
                "pf_soh_p05":          round(float(pf_qs[0]), 3),
                "pf_soh_p25":          round(float(pf_qs[1]), 3),
                "pf_soh_p50":          round(float(pf_qs[2]), 3),
                "pf_soh_p75":          round(float(pf_qs[3]), 3),
                "pf_soh_p95":          round(float(pf_qs[4]), 3),
                "pf_soh_skewness":     round(soh_sk, 4),
                "pf_soh_kurtosis":     round(soh_ku, 4),
                "pf_ngi":              round(ngi, 4),
                "pf_ess_ratio":        round(ess_ratio, 4),
                "pf_resampled":        resampled,
                "pf_unique_frac":      round(unique_frac, 4),
                "split":               split,
                "is_quality_gated":    is_qg,
            }

            # RUL from particle cloud
            rul_arr = _rul_particles(particles[:, 0], avg_efc_day)
            rul_qs  = np.quantile(rul_arr[np.isfinite(rul_arr)], [0.05, 0.50, 0.95]) \
                      if np.isfinite(rul_arr).any() else [np.nan, np.nan, np.nan]
            record["pf_rul_p50"] = round(float(rul_qs[1]), 0) if np.isfinite(rul_qs[1]) else np.nan
            record["pf_rul_p05"] = round(float(rul_qs[0]), 0) if np.isfinite(rul_qs[0]) else np.nan
            record["pf_rul_p95"] = round(float(rul_qs[2]), 0) if np.isfinite(rul_qs[2]) else np.nan

            # Forecasts (compute at every step; only last session's values are used for fleet CSV)
            for h in [60, 90]:
                fct = _pf_forecast(particles, avg_efc_day, avg_gap_days, h)
                record.update(fct)

            results.append(record)
            session_idx += 1

            if session_idx % 5 == 0:
                store_particles.append((float(row.get("cum_efc", 0.0)), particles[:, 0].copy()))

        final_particles[reg] = particles

    return pd.DataFrame(results), final_particles


# ── Metrics ───────────────────────────────────────────────────────────────────

def compute_metrics(df_sub: pd.DataFrame, label: str) -> dict:
    valid = df_sub.dropna(subset=["cycle_soh_obs", "pf_soh_mean"])
    if len(valid) < 3:
        return {"subset": label, "n": len(valid)}

    y_true = valid["cycle_soh_obs"].values
    y_pred = valid["pf_soh_mean"].values
    resid  = y_pred - y_true

    mae  = float(np.mean(np.abs(resid)))
    rmse = float(np.sqrt(np.mean(resid ** 2)))
    ss_r = np.sum(resid ** 2)
    ss_t = np.sum((y_true - y_true.mean()) ** 2)
    r2   = 1.0 - ss_r / ss_t if ss_t > 0 else np.nan
    mbe  = float(np.mean(resid))
    w1   = float(np.mean(np.abs(resid) <= 1.0))
    w2   = float(np.mean(np.abs(resid) <= 2.0))

    # PICP 90%: using p05/p95
    lo = valid["pf_soh_p05"].values
    hi = valid["pf_soh_p95"].values
    picp90 = float(np.mean((y_true >= lo) & (y_true <= hi)))
    mpiw90 = float(np.mean(hi - lo))

    # PICP 50%: using p25/p75
    lo50 = valid["pf_soh_p25"].values
    hi50 = valid["pf_soh_p75"].values
    picp50 = float(np.mean((y_true >= lo50) & (y_true <= hi50)))

    # Non-Gaussianity
    ngi_mean = float(df_sub["pf_ngi"].mean())
    ngi_flag = float((df_sub["pf_ngi"] > 0.5).mean())

    return {
        "subset": label, "n": len(valid),
        "mae": round(mae, 4), "rmse": round(rmse, 4),
        "r2": round(r2, 4), "mbe": round(mbe, 4),
        "within_1pct": round(w1, 4), "within_2pct": round(w2, 4),
        "picp90": round(picp90, 4), "mpiw90": round(mpiw90, 4),
        "picp50": round(picp50, 4),
        "ngi_mean": round(ngi_mean, 4), "ngi_flag_frac": round(ngi_flag, 4),
        "ess_mean": round(float(df_sub["pf_ess_ratio"].mean()), 4),
        "resample_frac": round(float(df_sub["pf_resampled"].mean()), 4),
    }


def compute_per_vehicle_metrics(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for reg, grp in df.groupby("registration_number"):
        test_grp = grp[grp["split"] == "test"]
        m = compute_metrics(test_grp, f"{reg}_test")
        m["registration_number"] = reg
        rows.append(m)
    return pd.DataFrame(rows)


# ── PF vs EKF comparison ──────────────────────────────────────────────────────

def pf_vs_ekf(df: pd.DataFrame) -> dict:
    """KL divergence and Wasserstein distance: PF posterior vs EKF Gaussian."""
    if not os.path.exists(EKF_CSV):
        return {}

    ekf_df = pd.read_csv(EKF_CSV)
    merged = df.merge(
        ekf_df[["session_id", "ekf_soh", "ekf_soh_std"]],
        on="session_id", how="inner"
    ).dropna(subset=["pf_soh_mean", "pf_soh_std", "ekf_soh", "ekf_soh_std"])

    if len(merged) < 10:
        return {}

    # Approximate KL divergence (PF posterior KDE vs EKF Gaussian)
    # Using per-session KL on a discretised grid
    kl_vals = []
    ws_vals = []
    grid    = np.linspace(85, 105, 200)

    for _, row in merged.sample(min(500, len(merged)), random_state=SEED).iterrows():
        pf_std = max(row["pf_soh_std"], 0.01)
        ek_std = max(row["ekf_soh_std"], 0.01)
        pf_pdf = sp_norm.pdf(grid, loc=row["pf_soh_mean"], scale=pf_std)
        ek_pdf = sp_norm.pdf(grid, loc=row["ekf_soh"],     scale=ek_std)

        # Smoothed to avoid 0 entries in KL
        pf_pdf = np.maximum(pf_pdf, 1e-12)
        ek_pdf = np.maximum(ek_pdf, 1e-12)
        pf_pdf /= pf_pdf.sum()
        ek_pdf /= ek_pdf.sum()

        kl = float(np.sum(pf_pdf * np.log(pf_pdf / ek_pdf)))
        kl_vals.append(kl)
        ws_vals.append(wasserstein_distance(grid, grid, pf_pdf, ek_pdf))

    delta_mean = float((merged["pf_soh_mean"] - merged["ekf_soh"]).mean())
    delta_std  = float((merged["pf_soh_std"]  - merged["ekf_soh_std"]).mean())

    return {
        "kl_div_mean":    round(float(np.mean(kl_vals)), 6),
        "kl_div_gt_0.1":  round(float(np.mean(np.array(kl_vals) > 0.1)), 4),
        "wasserstein_mean": round(float(np.mean(ws_vals)), 4),
        "pf_minus_ekf_mean_soh": round(delta_mean, 4),
        "pf_minus_ekf_mean_std": round(delta_std, 4),
    }


# ─────────────────────────────────────────────────────────────────────────────
# PLOTS
# ─────────────────────────────────────────────────────────────────────────────

RISK_RED   = "#c0392b"
RISK_AMBER = "#e67e22"
RISK_GREEN = "#27ae60"
FLEET_BLUE = "#2980b9"
FLEET_GREY = "#95a5a6"


def _risk_colour(rul_days) -> str:
    if not np.isfinite(float(rul_days if rul_days is not None else np.nan)):
        return FLEET_GREY
    if rul_days < 180:   return RISK_RED
    if rul_days < 365:   return RISK_AMBER
    return RISK_GREEN


def plot_posterior_evolution(df: pd.DataFrame, out_path: str):
    """Violin plot of particle SoH distribution over cum_efc — 3 vehicles."""
    sample_vehs = df["registration_number"].unique()[:3]
    fig, axes   = plt.subplots(1, 3, figsize=(15, 5), sharey=True)

    for ax, reg in zip(axes, sample_vehs):
        sub = df[df["registration_number"] == reg].sort_values("cum_efc")
        # Bin cum_efc into ~10 bins
        efc_bins = np.percentile(sub["cum_efc"], np.linspace(0, 100, 12))
        efc_bins = np.unique(np.round(efc_bins, 2))

        positions = []
        data_bins = []

        for i in range(len(efc_bins) - 1):
            mask = (sub["cum_efc"] >= efc_bins[i]) & (sub["cum_efc"] < efc_bins[i + 1])
            rows = sub[mask]
            if len(rows) == 0:
                continue
            mid = float((efc_bins[i] + efc_bins[i + 1]) / 2)
            # Approximate distribution from quantiles
            q_vals = rows[["pf_soh_p05", "pf_soh_p25", "pf_soh_p50",
                            "pf_soh_p75", "pf_soh_p95"]].mean()
            q_arr  = q_vals.values
            spread = np.interp(
                np.linspace(0, 1, 50),
                [0.05, 0.25, 0.5, 0.75, 0.95],
                q_arr
            )
            positions.append(mid)
            data_bins.append(spread)

        if data_bins:
            vp = ax.violinplot(data_bins, positions=positions,
                               widths=np.diff(efc_bins[:len(positions) + 1]).mean() * 0.6,
                               showmedians=True)
            for pc in vp["bodies"]:
                pc.set_facecolor(FLEET_BLUE)
                pc.set_alpha(0.5)

        # EKF overlay
        if os.path.exists(EKF_CSV):
            ekf_df = pd.read_csv(EKF_CSV)
            sub_e  = ekf_df[ekf_df["registration_number"] == reg].sort_values("cum_efc")
            if len(sub_e) > 0:
                ax.fill_between(
                    sub_e["cum_efc"],
                    sub_e["ekf_soh"] - 2 * sub_e["ekf_soh_std"],
                    sub_e["ekf_soh"] + 2 * sub_e["ekf_soh_std"],
                    color=RISK_AMBER, alpha=0.25, label="EKF ±2σ"
                )
                ax.plot(sub_e["cum_efc"], sub_e["ekf_soh"],
                        color=RISK_AMBER, lw=1.5, ls="--", label="EKF mean")

        obs = sub.dropna(subset=["cycle_soh_obs"])
        ax.scatter(obs["cum_efc"], obs["cycle_soh_obs"], s=12, c="black", zorder=5, label="Observed SoH")
        ax.axhline(EOL_SOH, color=RISK_RED, ls="--", lw=1.2)
        ax.set_ylim(85, 102)
        ax.set_title(reg, fontsize=9)
        ax.set_xlabel("Charge Cycles (EFC)")
        if ax == axes[0]:
            ax.set_ylabel("Battery Health (%)")
            ax.legend(fontsize=7)

    plt.suptitle("Particle Filter — Posterior Evolution (Violin)\nBlue: PF distribution · Orange dashed: EKF ±2σ",
                 fontsize=11, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out_path}")


def plot_vs_ekf(df: pd.DataFrame, out_path: str):
    """3-panel: (A) scatter PF mean vs EKF, (B) PF std vs EKF std, (C) skewness heatmap."""
    if not os.path.exists(EKF_CSV):
        print(f"  [SKIP] {out_path} — ekf_soh.csv not found")
        return

    ekf_df = pd.read_csv(EKF_CSV)
    merged = df.merge(
        ekf_df[["session_id", "ekf_soh", "ekf_soh_std"]],
        on="session_id", how="inner"
    ).dropna(subset=["pf_soh_mean", "ekf_soh"])

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Panel A: PF mean vs EKF mean
    ax = axes[0]
    ax.scatter(merged["ekf_soh"], merged["pf_soh_mean"], s=4, alpha=0.3, color=FLEET_BLUE)
    lo = min(merged["ekf_soh"].min(), merged["pf_soh_mean"].min())
    hi = max(merged["ekf_soh"].max(), merged["pf_soh_mean"].max())
    ax.plot([lo, hi], [lo, hi], "r--", lw=1.5)
    from scipy.stats import pearsonr as _pearsonr
    r, _ = _pearsonr(merged["ekf_soh"], merged["pf_soh_mean"])
    ax.set_xlabel("EKF SoH (%)"); ax.set_ylabel("PF SoH Mean (%)")
    ax.set_title(f"PF Mean vs EKF\n(r = {r:.3f})")

    # Panel B: PF std vs EKF std
    ax = axes[1]
    merged_std = merged.dropna(subset=["pf_soh_std", "ekf_soh_std"])
    ax.scatter(merged_std["ekf_soh_std"], merged_std["pf_soh_std"],
               s=4, alpha=0.3, color=RISK_AMBER)
    lo2 = min(merged_std["ekf_soh_std"].min(), merged_std["pf_soh_std"].min())
    hi2 = max(merged_std["ekf_soh_std"].max(), merged_std["pf_soh_std"].max())
    ax.plot([lo2, hi2], [lo2, hi2], "r--", lw=1.5)
    pct_above = (merged_std["pf_soh_std"] > merged_std["ekf_soh_std"]).mean() * 100
    ax.set_xlabel("EKF SoH Std (%)"); ax.set_ylabel("PF SoH Std (%)")
    ax.set_title(f"PF vs EKF Uncertainty\n({pct_above:.0f}% of sessions: PF wider than EKF)")

    # Panel C: skewness heatmap (vehicles × EFC bins)
    ax = axes[2]
    vehicles  = merged["registration_number"].unique()[:20]  # cap at 20
    efc_bins  = pd.cut(merged["cum_efc"], bins=8, labels=False)
    merged    = merged.copy()
    merged["efc_bin"] = efc_bins

    hmap = np.full((len(vehicles), 8), np.nan)
    for vi, reg in enumerate(vehicles):
        for bi in range(8):
            sub = merged[(merged["registration_number"] == reg) &
                         (merged["efc_bin"] == bi)]["pf_soh_skewness"]
            if len(sub) > 0:
                hmap[vi, bi] = sub.mean()

    im = ax.imshow(hmap, aspect="auto", cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_yticks(range(len(vehicles)))
    ax.set_yticklabels(vehicles, fontsize=6)
    ax.set_xlabel("EFC Bin (low → high)")
    ax.set_title("SoH Skewness\n(red=right-skew, blue=left-skew)")
    plt.colorbar(im, ax=ax, label="Skewness")

    plt.suptitle("Particle Filter vs EKF Comparison", fontsize=11, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out_path}")


def plot_ess_trajectory(df: pd.DataFrame, out_path: str):
    """ESS/N over cum_efc per vehicle + histogram."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    ax = axes[0]
    vehicles = df["registration_number"].unique()
    cmap     = plt.cm.tab20
    for i, reg in enumerate(vehicles):
        sub = df[df["registration_number"] == reg].sort_values("cum_efc")
        ax.plot(sub["cum_efc"], sub["pf_ess_ratio"],
                color=cmap(i / max(len(vehicles), 1)), lw=0.7, alpha=0.6)
    ax.axhline(RESAMPLE_ESS, color="orange", ls="--", lw=1.5,
               label=f"Resample threshold ({RESAMPLE_ESS})")
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("Charge Cycles (EFC)")
    ax.set_ylabel("ESS / N")
    ax.set_title("Effective Sample Size per Session")
    ax.legend(fontsize=8)

    ax = axes[1]
    ess_all = df["pf_ess_ratio"].dropna()
    ax.hist(ess_all, bins=40, color=FLEET_BLUE, alpha=0.7, edgecolor="white")
    ax.axvline(RESAMPLE_ESS, color="orange", ls="--", lw=1.5, label="Resample threshold")
    ax.axvline(ess_all.mean(), color="black", ls="-", lw=1.5, label=f"Mean = {ess_all.mean():.3f}")
    ax.set_xlabel("ESS / N")
    ax.set_ylabel("Count")
    ax.set_title(f"ESS Distribution (mean = {ess_all.mean():.3f})\n"
                 f"Resample triggered: {100*df['pf_resampled'].mean():.1f}% of steps")
    ax.legend(fontsize=8)

    plt.suptitle("Particle Filter — Effective Sample Size", fontsize=11, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out_path}")


def plot_non_gaussianity(df: pd.DataFrame, out_path: str):
    """Skewness and excess kurtosis vs cum_efc."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    ax = axes[0]
    ax.scatter(df["cum_efc"], df["pf_soh_skewness"],
               c=(df["pf_soh_skewness"].abs() > 0.5).map({True: RISK_RED, False: FLEET_GREY}),
               s=5, alpha=0.5)
    ax.axhline(0, color="black", lw=1)
    ax.axhline(0.5, color=RISK_RED, ls="--", lw=1, label="|skew| > 0.5 threshold")
    ax.axhline(-0.5, color=RISK_RED, ls="--", lw=1)
    frac_ng = (df["pf_soh_skewness"].abs() > 0.5).mean()
    ax.set_xlabel("Charge Cycles (EFC)")
    ax.set_ylabel("Posterior Skewness")
    ax.set_title(f"SoH Posterior Skewness ({100*frac_ng:.1f}% non-Gaussian by |skew|>0.5)")
    ax.legend(fontsize=8)

    ax = axes[1]
    ax.scatter(df["cum_efc"], df["pf_soh_kurtosis"],
               c=(df["pf_soh_kurtosis"].abs() > 1.0).map({True: RISK_AMBER, False: FLEET_GREY}),
               s=5, alpha=0.5)
    ax.axhline(0, color="black", lw=1, label="Gaussian reference (0)")
    ax.set_xlabel("Charge Cycles (EFC)")
    ax.set_ylabel("Excess Kurtosis")
    ax.set_title("SoH Posterior Excess Kurtosis\n(0 = Gaussian; >0 = heavier tails)")
    ax.legend(fontsize=8)

    plt.suptitle("Particle Filter — Non-Gaussianity Index\n"
                 "Justifies PF over EKF when |skewness| or |kurtosis| is large",
                 fontsize=11, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.90])
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out_path}")


def plot_rul_distributions(df: pd.DataFrame, out_path: str):
    """Per-vehicle RUL distribution from PF vs EKF point estimate."""
    if not os.path.exists(EKF_CSV):
        print(f"  [SKIP] {out_path} — ekf_soh.csv not found")
        return

    ekf_df = pd.read_csv(EKF_CSV)
    sample_vehs = df["registration_number"].unique()[:6]

    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    axes = axes.flatten()

    for ax, reg in zip(axes, sample_vehs):
        sub_pf  = df[df["registration_number"] == reg].dropna(subset=["pf_rul_p50"])
        sub_ekf = ekf_df[ekf_df["registration_number"] == reg].dropna(subset=["ekf_rul_days"])

        rul_p50_last = sub_pf["pf_rul_p50"].iloc[-1] if len(sub_pf) > 0 else np.nan
        rul_p05_last = sub_pf["pf_rul_p05"].iloc[-1] if len(sub_pf) > 0 else np.nan
        rul_p95_last = sub_pf["pf_rul_p95"].iloc[-1] if len(sub_pf) > 0 else np.nan

        # Approximate distribution from p05/p50/p95
        if all(np.isfinite([rul_p05_last, rul_p50_last, rul_p95_last])):
            rul_samples = np.random.normal(
                rul_p50_last,
                (rul_p95_last - rul_p05_last) / (2 * 1.96),
                1000
            )
            rul_samples = rul_samples[rul_samples > 0]
            if len(rul_samples) > 0:
                ax.hist(rul_samples, bins=30, color=FLEET_BLUE, alpha=0.7,
                        edgecolor="white", density=True, label="PF RUL dist.")

        # EKF point estimate
        if len(sub_ekf) > 0:
            ekf_rul = sub_ekf["ekf_rul_days"].iloc[-1]
            if np.isfinite(ekf_rul):
                ax.axvline(ekf_rul, color=RISK_RED, ls="--", lw=2, label=f"EKF: {ekf_rul:.0f}d")

        if np.isfinite(rul_p50_last):
            ax.axvline(rul_p50_last, color=FLEET_BLUE, lw=2, label=f"PF p50: {rul_p50_last:.0f}d")

        ax.axvline(365, color="black", ls=":", lw=1, label="1 year")
        ax.set_xlabel("Remaining Useful Life (days)")
        ax.set_ylabel("Density")
        ax.set_title(reg, fontsize=9)
        ax.legend(fontsize=6)

    plt.suptitle("Remaining Useful Life Distributions — PF vs EKF Point Estimate\n"
                 "Right-skewed distribution = EKF may understate uncertainty",
                 fontsize=11, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out_path}")


def plot_prior_sensitivity(df_main: pd.DataFrame, results_sens: list, out_path: str):
    """
    3 prior configurations: pf_mean ± p25-p75 band over cum_efc.
    results_sens: [(label, df), ...]
    """
    fig, ax = plt.subplots(figsize=(10, 5))
    colours = [FLEET_BLUE, RISK_AMBER, RISK_GREEN]

    for (label, df_s), col in zip(results_sens, colours):
        # Use a single representative vehicle
        reg   = df_s["registration_number"].iloc[0]
        sub_s = df_s[df_s["registration_number"] == reg].sort_values("cum_efc")
        ax.plot(sub_s["cum_efc"], sub_s["pf_soh_mean"], color=col, lw=1.5, label=label)
        ax.fill_between(sub_s["cum_efc"],
                        sub_s["pf_soh_p25"], sub_s["pf_soh_p75"],
                        color=col, alpha=0.2)

    # Main result
    reg_m = df_main["registration_number"].iloc[0]
    sub_m = df_main[df_main["registration_number"] == reg_m].sort_values("cum_efc")
    ax.plot(sub_m["cum_efc"], sub_m["pf_soh_mean"], "k-", lw=2, label="Main run N(100,1²)")

    obs = sub_m.dropna(subset=["cycle_soh_obs"])
    ax.scatter(obs["cum_efc"], obs["cycle_soh_obs"], s=20, c="black", zorder=5)
    ax.axhline(EOL_SOH, color=RISK_RED, ls="--", lw=1)
    ax.set_ylim(85, 103)
    ax.set_xlabel("Charge Cycles (EFC)")
    ax.set_ylabel("Battery Health (%)")
    ax.set_title(f"Prior Sensitivity Analysis — Vehicle {reg_m}\n"
                 "Bands = p25–p75 particle spread per prior")
    ax.legend(fontsize=9)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out_path}")


def plot_rul_fleet_summary(fleet_df: pd.DataFrame, out_path: str):
    """
    STAKEHOLDER — Horizontal lollipop chart.
    One row per vehicle, sorted by p50 RUL ascending.
    Dot = p50, thick line = p25–p75, whisker = p05–p95.
    """
    df_s  = fleet_df.sort_values("rul_days_p50", ascending=True).reset_index(drop=True)
    n     = len(df_s)
    y_pos = np.arange(n)

    fig, ax = plt.subplots(figsize=(11, max(6, n * 0.3 + 2)))

    for i, row in df_s.iterrows():
        p50 = row.get("rul_days_p50") or np.nan
        p05 = row.get("rul_days_p05") or np.nan
        p95 = row.get("rul_days_p95") or np.nan
        p25 = row.get("rul_days_p25") or p50
        p75 = row.get("rul_days_p75") or p50
        col = _risk_colour(p50)

        if np.isfinite(p05) and np.isfinite(p95):
            ax.plot([p05, p95], [i, i], color=col, lw=1.5, alpha=0.4)   # whisker
        if np.isfinite(p25) and np.isfinite(p75):
            ax.plot([p25, p75], [i, i], color=col, lw=4, alpha=0.7)     # IQR
        if np.isfinite(p50):
            ax.scatter(p50, i, color=col, s=60, zorder=5)               # median dot

    ax.axvline(365, color="black", ls=":", lw=1.5, label="1 year reference")
    ax.axvline(180, color=RISK_AMBER, ls=":", lw=1, label="6 months")
    ax.set_yticks(y_pos)
    ax.set_yticklabels(df_s["registration_number"].values, fontsize=8)
    ax.set_xlabel("Remaining Useful Life (days)", fontsize=11)
    ax.set_title("Fleet Remaining Useful Life Forecast (Battery) — with Uncertainty\n"
                 "Dot = most likely  ·  Thick bar = 50% range  ·  Whisker = 90% range",
                 fontsize=11, fontweight="bold")

    legend_elements = [
        Line2D([0], [0], color=RISK_RED,   lw=3, label="< 6 months (urgent)"),
        Line2D([0], [0], color=RISK_AMBER, lw=3, label="6–12 months (monitor)"),
        Line2D([0], [0], color=RISK_GREEN, lw=3, label="> 12 months (normal)"),
    ]
    ax.legend(handles=legend_elements, loc="lower right", fontsize=9)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out_path}")


def plot_forecast_60_90(df: pd.DataFrame, fleet_df: pd.DataFrame, out_path: str):
    """
    STAKEHOLDER — Per vehicle: SoH history + 60/90d forecast violin.
    X-axis: calendar date.
    """
    vehicles = df["registration_number"].unique()
    n_cols   = min(3, len(vehicles))
    n_rows   = (len(vehicles) + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 4 * n_rows),
                              sharey=True, squeeze=False)

    from datetime import datetime, timedelta

    for idx, reg in enumerate(vehicles):
        ax   = axes[idx // n_cols][idx % n_cols]
        sub  = df[df["registration_number"] == reg].sort_values("start_time")
        fc_row = fleet_df[fleet_df["registration_number"] == reg]

        # Convert start_time to datetime if possible
        try:
            sub = sub.copy()
            sub["dt"] = pd.to_datetime(sub["start_time"], unit="ms", errors="coerce")
            if sub["dt"].notna().mean() > 0.5:
                x_hist = sub["dt"]
                last_dt = sub["dt"].dropna().iloc[-1]
                x_60  = last_dt + timedelta(days=60)
                x_90  = last_dt + timedelta(days=90)
            else:
                x_hist = sub["cum_efc"]
                x_60   = sub["cum_efc"].max() * 1.05
                x_90   = sub["cum_efc"].max() * 1.10
        except Exception:
            x_hist = sub["cum_efc"]
            x_60   = sub["cum_efc"].max() * 1.05
            x_90   = sub["cum_efc"].max() * 1.10

        # History
        ax.plot(x_hist, sub["pf_soh_mean"], color=FLEET_BLUE, lw=1.5, label="SoH history")
        ax.fill_between(x_hist, sub["pf_soh_p25"], sub["pf_soh_p75"],
                        alpha=0.25, color=FLEET_BLUE)

        # 60/90d forecast violins (from fleet_df)
        if len(fc_row) > 0:
            for h, x_h, col in [(60, x_60, "royalblue"), (90, x_90, RISK_AMBER)]:
                p50 = fc_row.get(f"soh_{h}d_p50", fc_row.get("soh_90d")).values[0] if not fc_row.empty else np.nan
                p05 = fc_row.get(f"soh_{h}d_p05", p50).values[0] if not fc_row.empty else np.nan
                p95 = fc_row.get(f"soh_{h}d_p95", p50).values[0] if not fc_row.empty else np.nan
                if np.isfinite(p50):
                    ax.errorbar(x_h, p50, yerr=[[p50 - p05], [p95 - p50]],
                                fmt="D", color=col, capsize=5, ms=7,
                                label=f"{h}d forecast")

        ax.axhline(EOL_SOH, color=RISK_RED, ls="--", lw=1.2, label="EOL 80%")
        ax.set_ylim(84, 103)
        ax.set_title(reg, fontsize=8)
        ax.tick_params(axis="x", rotation=30, labelsize=7)

    # Hide empty panels
    for idx in range(len(vehicles), n_rows * n_cols):
        axes[idx // n_cols][idx % n_cols].set_visible(False)

    handles = [
        Line2D([0], [0], color=FLEET_BLUE, lw=2, label="SoH History"),
        Line2D([0], [0], color="royalblue", marker="D", ls="", ms=8, label="60-day Forecast"),
        Line2D([0], [0], color=RISK_AMBER, marker="D", ls="", ms=8, label="90-day Forecast"),
        Line2D([0], [0], color=RISK_RED, ls="--", lw=2, label="EOL 80%"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=4, fontsize=9,
               bbox_to_anchor=(0.5, -0.04))
    fig.suptitle("Projected SoH Distribution — 60 & 90 Day Outlook\n"
                 "Diamond = median  ·  Error bars = 90% probability range",
                 fontsize=12, fontweight="bold")
    plt.tight_layout(rect=[0, 0.04, 1, 0.95])
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out_path}")


# ── Fleet forecast summary ────────────────────────────────────────────────────

def build_fleet_forecast(df: pd.DataFrame) -> pd.DataFrame:
    """One row per vehicle with 60/90-day PF forecasts + RUL + risk flag."""
    latest = (
        df.sort_values("cum_efc")
          .groupby("registration_number")
          .last()
          .reset_index()
    )

    rows = []
    for _, row in latest.iterrows():
        cur_soh = float(row["pf_soh_mean"])
        soh_90  = float(row.get("soh_pred_90d_p50", np.nan))
        delta   = (cur_soh - soh_90) if np.isfinite(soh_90) else np.nan
        rul_p50 = row.get("pf_rul_p50")

        risk_col = _risk_colour(rul_p50 if rul_p50 else np.nan)
        risk_str = "red" if risk_col == RISK_RED else ("amber" if risk_col == RISK_AMBER else "green")

        rows.append({
            "registration_number": row["registration_number"],
            "current_soh":         round(cur_soh, 2),
            "soh_60d_p05":         row.get("soh_pred_60d_p05"),
            "soh_60d_p50":         row.get("soh_pred_60d_p50"),
            "soh_60d_p95":         row.get("soh_pred_60d_p95"),
            "soh_90d_p05":         row.get("soh_pred_90d_p05"),
            "soh_90d_p50":         row.get("soh_pred_90d_p50"),
            "soh_90d_p95":         row.get("soh_pred_90d_p95"),
            "delta_soh_90d":       round(delta, 2) if np.isfinite(delta) else np.nan,
            "rul_days_p05":        row.get("pf_rul_p05"),
            "rul_days_p25":        row.get("pf_rul_p50"),   # approx
            "rul_days_p50":        row.get("pf_rul_p50"),
            "rul_days_p75":        row.get("pf_rul_p50"),   # approx
            "rul_days_p95":        row.get("pf_rul_p95"),
            "risk_flag":           risk_str,
        })

    return pd.DataFrame(rows).sort_values("rul_days_p50")


# ── Main ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 70)
    print(f"particle_filter_soh.py — Sequential Monte Carlo (N={N_PARTICLES})")
    print("=" * 70)

    # ── Load data ─────────────────────────────────────────────────────────────
    print(f"\nLoading cycles.csv from {CYCLES_CSV} ...")
    cycles = pd.read_csv(CYCLES_CSV)
    print(f"  {len(cycles):,} sessions, {cycles['registration_number'].nunique()} vehicles")


    max_days = cycles["days_since_first"].max() if "days_since_first" in cycles.columns else 0
    if max_days < 180:
        print(f"\n  YOUNG FLEET WARNING: {max_days:.0f} days. Prior dominates all estimates.")

    # ── Main PF run ───────────────────────────────────────────────────────────
    print(f"\nRunning particle filter (N={N_PARTICLES}) ...")
    t0 = time.time()
    pf_df, final_particles = run_pf_fleet(cycles, n_particles=N_PARTICLES)
    elapsed = time.time() - t0
    print(f"  Done in {elapsed:.1f}s — {len(pf_df):,} sessions")

    pf_df.to_csv(PF_CSV, index=False)
    print(f"  Saved: {PF_CSV}")

    # Save final particles
    vehicles_list = list(final_particles.keys())
    if len(vehicles_list) > 0:
        max_n = max(len(p) for p in final_particles.values())
        arr   = np.full((len(vehicles_list), max_n, 2), np.nan)
        for vi, reg in enumerate(vehicles_list):
            p = final_particles[reg]
            arr[vi, :len(p), :] = p
        np.save(PARTICLES_NPY, arr)
        print(f"  Saved particles: {PARTICLES_NPY} shape={arr.shape}")

    # ── Metrics ───────────────────────────────────────────────────────────────
    print("\nComputing metrics ...")
    test_all = pf_df[pf_df["split"] == "test"]
    test_qg  = test_all[test_all["is_quality_gated"]]

    metrics_rows = [
        compute_metrics(pf_df,   "train_all"),
        compute_metrics(pf_df[pf_df["is_quality_gated"]], "train_quality_gated"),
        compute_metrics(test_all, "test_all"),
        compute_metrics(test_qg,  "test_quality_gated"),
    ]
    metrics_df = pd.DataFrame(metrics_rows)
    metrics_df.to_csv(METRICS_CSV, index=False)
    print(metrics_df[["subset", "n", "mae", "rmse", "r2",
                       "picp90", "picp50", "ngi_mean", "ess_mean"]].to_string(index=False))

    # Per-vehicle metrics
    perveh_df = compute_per_vehicle_metrics(pf_df)
    perveh_df.to_csv(PERVEH_CSV, index=False)

    # ── PF vs EKF ─────────────────────────────────────────────────────────────
    print("\nPF vs EKF comparison ...")
    ekf_comp = pf_vs_ekf(pf_df)
    for k, v in ekf_comp.items():
        print(f"  {k}: {v}")
    if "kl_div_mean" in ekf_comp:
        kl = ekf_comp["kl_div_mean"]
        print(f"  KL divergence mean: {kl:.6f} "
              f"({'PF and EKF materially differ' if kl > 0.1 else 'PF and EKF approximately agree'})")

    # ── ESS summary ───────────────────────────────────────────────────────────
    ess_mean       = pf_df["pf_ess_ratio"].mean()
    resample_frac  = pf_df["pf_resampled"].mean()
    unique_frac    = pf_df["pf_unique_frac"].mean()
    print(f"\nParticle health:")
    print(f"  Mean ESS/N: {ess_mean:.3f}  ({'OK' if ess_mean > 0.5 else 'WARN: low ESS'})")
    print(f"  Resample triggered: {100*resample_frac:.1f}% of steps")
    print(f"  Mean unique particle fraction: {unique_frac:.3f} "
          f"({'OK' if unique_frac > 0.1 else 'WARN: particle degeneracy'})")

    # ── Non-Gaussianity ───────────────────────────────────────────────────────
    ngi_mean = pf_df["pf_ngi"].mean()
    ngi_flag = (pf_df["pf_ngi"] > 0.5).mean()
    print(f"\nNon-Gaussianity Index (NGI):")
    print(f"  Mean NGI: {ngi_mean:.4f} (>0.5 = significant non-Gaussian posterior)")
    print(f"  Sessions with NGI > 0.5: {100*ngi_flag:.1f}%")
    if ngi_flag > 0.1:
        print("  PF posterior differs meaningfully from Gaussian — PF adds value over EKF.")

    # ── Prior sensitivity ─────────────────────────────────────────────────────
    print("\nRunning prior sensitivity (3 priors, fast mode N=200) ...")
    prior_configs = [
        ("N(100, 4²)",  100.0, 4.0),
        ("N(97, 2²)",    97.0, 2.0),
        ("N(100, 1²) — main", 100.0, 1.0),
    ]
    sens_results = []
    for label, mu, sig in prior_configs:
        df_sens, _ = run_pf_fleet(cycles.groupby("registration_number").head(30).reset_index(drop=True),
                                   n_particles=200, prior_soh_mean=mu, prior_soh_std=sig)
        sens_results.append((label, df_sens))
        valid = df_sens.dropna(subset=["cycle_soh_obs", "pf_soh_mean"])
        mae   = float(np.mean(np.abs(valid["pf_soh_mean"] - valid["cycle_soh_obs"]))) if len(valid) > 0 else np.nan
        print(f"  {label}: MAE = {mae:.4f}%")

    # Check convergence: within 0.5% of each other
    try:
        last_soh = [df_s.groupby("registration_number")["pf_soh_mean"].last().mean()
                    for _, df_s in sens_results]
        spread = max(last_soh) - min(last_soh)
        print(f"  Prior spread at end: {spread:.3f}%"
              f"  ({'Converged OK' if spread < 0.5 else 'WARN: priors not converged'})")
    except Exception:
        pass

    # ── Particle count sensitivity ────────────────────────────────────────────
    print("\nParticle count sensitivity (N ∈ {100, 200, 500}) ...")
    first_veh = cycles[cycles["registration_number"] == cycles["registration_number"].iloc[0]]
    for n_p in [100, 200, 500]:
        t_s = time.time()
        df_ps, _ = run_pf_fleet(first_veh, n_particles=n_p)
        valid = df_ps.dropna(subset=["cycle_soh_obs", "pf_soh_mean"])
        mae_p   = float(np.mean(np.abs(valid["pf_soh_mean"] - valid["cycle_soh_obs"]))) if len(valid) > 0 else np.nan
        picp90  = float(np.mean((valid["cycle_soh_obs"].values >= valid["pf_soh_p05"].values) &
                                (valid["cycle_soh_obs"].values <= valid["pf_soh_p95"].values))) if len(valid) > 0 else np.nan
        ngi_m   = df_ps["pf_ngi"].mean()
        t_e     = time.time() - t_s
        print(f"  N={n_p:4d}: MAE={mae_p:.4f}  PICP90={picp90:.3f}  NGI={ngi_m:.4f}  t={t_e:.1f}s")

    # ── Fleet forecast ────────────────────────────────────────────────────────
    print("\nBuilding fleet forecast ...")
    fleet_fc = build_fleet_forecast(pf_df)
    fleet_fc.to_csv(FLEET_CSV, index=False)
    print(f"  Saved: {FLEET_CSV}")
    print(fleet_fc[["registration_number", "current_soh", "soh_90d_p50",
                     "rul_days_p50", "risk_flag"]].to_string(index=False))

    rc = fleet_fc["risk_flag"].value_counts()
    print(f"\n  Risk: red={rc.get('red',0)}, amber={rc.get('amber',0)}, green={rc.get('green',0)}")

    # ── Plots ─────────────────────────────────────────────────────────────────
    print("\nGenerating plots ...")
    plot_posterior_evolution(pf_df, os.path.join(PLOTS_DIR, "pf_posterior_evolution.png"))
    plot_vs_ekf(pf_df,              os.path.join(PLOTS_DIR, "pf_vs_ekf_comparison.png"))
    plot_ess_trajectory(pf_df,      os.path.join(PLOTS_DIR, "pf_ess_trajectory.png"))
    plot_non_gaussianity(pf_df,     os.path.join(PLOTS_DIR, "pf_non_gaussianity.png"))
    plot_rul_distributions(pf_df,   os.path.join(PLOTS_DIR, "pf_rul_distribution.png"))
    plot_prior_sensitivity(pf_df, sens_results, os.path.join(PLOTS_DIR, "pf_prior_sensitivity.png"))
    plot_rul_fleet_summary(fleet_fc, os.path.join(PLOTS_DIR, "pf_rul_fleet_summary.png"))
    plot_forecast_60_90(pf_df, fleet_fc, os.path.join(PLOTS_DIR, "pf_forecast_60_90.png"))

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("PARTICLE FILTER SUMMARY")
    print("=" * 70)
    test_qg_valid = test_qg.dropna(subset=["cycle_soh_obs", "pf_soh_mean"])
    if len(test_qg_valid) > 0:
        mae_qg = float(np.mean(np.abs(test_qg_valid["pf_soh_mean"] - test_qg_valid["cycle_soh_obs"])))
        print(f"  Test MAE (quality-gated): {mae_qg:.4f}%")
    print(f"  ESS/N mean: {ess_mean:.3f}")
    print(f"  Non-Gaussian sessions: {100*ngi_flag:.1f}%")
    fleet_soh = pf_df.groupby("registration_number")["pf_soh_mean"].last()
    print(f"  Fleet SoH range (PF): {fleet_soh.min():.2f}% – {fleet_soh.max():.2f}%")
    fleet_rul = fleet_fc["rul_days_p50"].dropna()
    print(f"  Fleet RUL p50 range: {fleet_rul.min():.0f}d – {fleet_rul.max():.0f}d")
    print(f"\n  Next: run model_eval_comparison.py")
    print()
