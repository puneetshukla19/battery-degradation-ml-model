import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) if "__file__" in dir() else os.getcwd())

"""
gpr_soh.py — Gaussian Process Regression for fleet SoH estimation & forecasting.

Model
-----
GaussianProcessRegressor with ARD Matérn-5/2 kernel (one length-scale per feature).
  - Input  : 9 session-level features from cycles.csv
  - Target : cycle_soh (ffill/bfill interpolated to all sessions; quality-gated
             sessions are the reliable anchor points)
  - Scope  : Fleet-wide (single model; per-vehicle N too small for individual fits)

Business outputs
----------------
  • Current SoH estimate with ±σ uncertainty band per vehicle
  • 60-day and 90-day SoH forecast per vehicle (query GPR at projected feature values)
  • Feature importance chart — which operational factors drive degradation

Outputs
-------
  artifacts/gpr_predictions.csv          — per-session predictions + 60/90-day forecasts
  artifacts/gpr_metrics.csv              — MAE / RMSE / R² / calibration per split
  artifacts/gpr_per_vehicle_metrics.csv  — per-vehicle breakdown
  artifacts/gpr_kernel_params.json       — optimised ARD length-scales + log-ML
  artifacts/fleet_forecast_gpr.csv       — one row per vehicle, risk-flagged forecast
  plots/gpr_fit_surface.png
  plots/gpr_residuals.png
  plots/gpr_calibration.png
  plots/gpr_per_vehicle.png
  plots/gpr_learning_curve.png
  plots/gpr_feature_importance.png       ← stakeholder-facing
  plots/gpr_fleet_forecast_60_90.png     ← stakeholder-facing
"""

import json
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from scipy import stats
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel, RBF
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

from config import (
    CYCLES_CSV, ARTIFACTS_DIR, PLOTS_DIR,
    EOL_SOH, CYCLE_SOH_OBS_CAP, CYCLE_SOH_MIN_BLOCK_DOD, SEED,
)

np.random.seed(SEED)
warnings.filterwarnings("ignore")

# ── Paths ──────────────────────────────────────────────────────────────────────
EKF_CSV         = os.path.join(ARTIFACTS_DIR, "ekf_soh.csv")
PRED_CSV        = os.path.join(ARTIFACTS_DIR, "gpr_predictions.csv")
METRICS_CSV     = os.path.join(ARTIFACTS_DIR, "gpr_metrics.csv")
VEH_METRICS_CSV = os.path.join(ARTIFACTS_DIR, "gpr_per_vehicle_metrics.csv")
KERNEL_JSON     = os.path.join(ARTIFACTS_DIR, "gpr_kernel_params.json")
FORECAST_CSV    = os.path.join(ARTIFACTS_DIR, "fleet_forecast_gpr.csv")

# ── Feature configuration ──────────────────────────────────────────────────────
# Plain-English labels for stakeholder plots (map from column name)
FEATURE_LABELS = {
    # Aging / cumulative load
    "cum_efc":                "Charge Cycles (EFC)",
    "aging_index":            "Composite Aging Index",
    # EKF state estimates (trajectory-aware priors)
    "ekf_soh":                "EKF State of Health (%)",
    "ekf_soh_std":            "EKF SoH Uncertainty (σ)",
    # Trend / EWM signals (capture degradation trajectory)
    "soh_trend_slope":        "SoH Degradation Slope (per day)",
    "ir_ohm_mean_ewm10":      "Internal Resistance EWM-10 (Ω)",
    "cell_spread_mean_ewm10": "Cell Imbalance EWM-10 (V)",
    # Instantaneous health signals
    "ir_ohm_mean":            "Internal Resistance (Ω, session avg)",
    "cell_spread_mean":       "Cell Voltage Imbalance (V, avg)",
    "temp_rise_rate":         "Temperature Rise Rate (°C/hr)",
    "n_vsag":                 "Voltage Sag Events (count)",
    # Usage / operational context
    "energy_per_km":          "Energy Consumption (kWh/km)",
    "soc_range":              "State-of-Charge Swing (%)",
    "dod_stress":             "Depth-of-Discharge Stress",
}

FEATURE_COLS = list(FEATURE_LABELS.keys())

# Features excluded from the GPR kernel: they saturated at the ARD upper bound (100.0)
# in the initial fit, meaning the optimizer found them uninformative.  Keeping them wastes
# kernel dimensions and slows convergence.
_GPR_EXCLUDE = {"aging_index", "soc_range", "dod_stress"}
GPR_FEATURE_COLS = [c for c in FEATURE_COLS if c not in _GPR_EXCLUDE]
GPR_FEAT_IDX     = [FEATURE_COLS.index(c) for c in GPR_FEATURE_COLS]
EKF_FEAT_IDX     = FEATURE_COLS.index("ekf_soh")   # used in collinearity check

# Feature categories for colour-coding
FEATURE_CATEGORY = {
    "cum_efc":                "Aging",
    "aging_index":            "Aging",
    "ekf_soh":                "Aging",
    "ekf_soh_std":            "Aging",
    "soh_trend_slope":        "Aging",
    "ir_ohm_mean_ewm10":      "Health",
    "cell_spread_mean_ewm10": "Health",
    "ir_ohm_mean":            "Health",
    "cell_spread_mean":       "Health",
    "n_vsag":                 "Health",
    "temp_rise_rate":         "Thermal",
    "energy_per_km":          "Efficiency",
    "soc_range":              "Usage",
    "dod_stress":             "Usage",
}

CATEGORY_COLORS = {
    "Aging":      "#E15759",
    "Health":     "#4E79A7",
    "Thermal":    "#F28E2B",
    "Efficiency": "#59A14F",
    "Usage":      "#B07AA1",
}

TRAIN_FRAC = 0.80
SOH_Y_LIM  = (85, 102)


# ── Data loading ───────────────────────────────────────────────────────────────

def _clean_soh_labels(df: pd.DataFrame, iqr_k: float = 1.2, rolling_window: int = 3) -> pd.DataFrame:
    """
    Per-vehicle SOH label cleaning applied before quality-gate computation:
      1. IQR outlier removal (k=1.2, tighter than default 1.5) — sets implausible
         cycle_soh values to NaN so they are excluded from quality-gated training.
      2. Rolling-median smoothing (window=3, centred) — removes session-to-session
         noise while preserving the slow degradation trend.

    Both steps operate per-vehicle so the fleet-wide mean does not suppress
    within-vehicle variance. Uses transform() to avoid pandas apply() column-drop
    issues in pandas 2.x.
    """
    df = df.copy()
    before_qg = df["cycle_soh"].notna().sum()

    group_col = "registration_number" if "registration_number" in df.columns else None

    def _iqr_filter(s: pd.Series) -> pd.Series:
        valid = s.dropna()
        if len(valid) < 4:
            return s
        q1, q3 = valid.quantile(0.25), valid.quantile(0.75)
        iqr = q3 - q1
        return s.where((s >= q1 - iqr_k * iqr) & (s <= q3 + iqr_k * iqr))

    def _rolling_median(s: pd.Series) -> pd.Series:
        if s.notna().sum() < 2:
            return s
        return s.rolling(rolling_window, min_periods=1, center=True).median()

    if group_col:
        soh_filtered = df.groupby(group_col)["cycle_soh"].transform(_iqr_filter)
        df["cycle_soh"] = soh_filtered
        soh_smoothed = df.groupby(group_col)["cycle_soh"].transform(_rolling_median)
        df["cycle_soh"] = soh_smoothed
    else:
        df["cycle_soh"] = _rolling_median(_iqr_filter(df["cycle_soh"]))

    after_qg = df["cycle_soh"].notna().sum()
    removed = before_qg - after_qg
    print(f"  SOH label cleaning: {removed:,} outlier rows nulled "
          f"({removed / max(before_qg, 1) * 100:.1f}%), "
          f"rolling-median (window={rolling_window}) applied per vehicle.")
    return df


def load_data() -> pd.DataFrame:
    print(f"Loading cycles.csv ...")
    cycles = pd.read_csv(CYCLES_CSV, low_memory=False)
    print(f"  {len(cycles):,} sessions, {cycles['registration_number'].nunique()} vehicles")

    # Merge EKF SOH estimates as features (Kalman-filtered, trajectory-aware)
    ekf_path = os.path.join(ARTIFACTS_DIR, "ekf_soh.csv")
    if os.path.exists(ekf_path):
        ekf_cols = ["registration_number", "session_id", "ekf_soh", "ekf_soh_std"]
        ekf = pd.read_csv(ekf_path, low_memory=False)
        ekf = ekf[[c for c in ekf_cols if c in ekf.columns]]
        cycles = cycles.merge(ekf, on=["registration_number", "session_id"], how="left",
                              suffixes=("", "_ekf"))
        print(f"  EKF merge: {cycles['ekf_soh'].notna().sum():,} / {len(cycles):,} sessions have EKF SOH")
    else:
        cycles["ekf_soh"] = np.nan
        cycles["ekf_soh_std"] = np.nan
        print("  [WARN] ekf_soh.csv not found — ekf_soh feature will be NaN (run ekf_soh.py first)")

    # Ensure all feature columns exist
    for col in FEATURE_COLS:
        if col not in cycles.columns:
            cycles[col] = np.nan

    # Clean SOH labels before quality-gating: IQR outlier removal + rolling-median smooth
    cycles = _clean_soh_labels(cycles)

    # Quality gate flag (computed after label cleaning)
    # Hard floor at 80% rejects corrupt readings (observed min was 9.1% — clearly erroneous)
    block_dod = cycles.get("block_soc_diff", cycles.get("soc_range", pd.Series(0.0, index=cycles.index))).abs()
    cycles["is_quality_gated"] = (
        cycles["cycle_soh"].notna() &
        (cycles["cycle_soh"] >= 80.0) &
        (cycles["cycle_soh"] < CYCLE_SOH_OBS_CAP) &
        (block_dod >= CYCLE_SOH_MIN_BLOCK_DOD)
    )

    # Convert start_time to datetime for calendar-date plots
    try:
        cycles["start_dt"] = pd.to_datetime(cycles["start_time"], unit="ms", utc=True).dt.tz_convert("Asia/Kolkata")
    except Exception:
        try:
            cycles["start_dt"] = pd.to_datetime(cycles["start_time"])
        except Exception:
            cycles["start_dt"] = pd.NaT

    return cycles


def make_train_test_split(df: pd.DataFrame) -> pd.DataFrame:
    """Time-based 80/20 per-vehicle split (consistent with existing pipeline)."""
    df = df.sort_values(["registration_number", "start_time"]).copy()
    df["split"] = "train"
    for _, grp in df.groupby("registration_number"):
        n = len(grp)
        cutoff = int(n * TRAIN_FRAC)
        df.loc[grp.index[cutoff:], "split"] = "test"
    return df


# ── Metrics ────────────────────────────────────────────────────────────────────

def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray,
                    y_std: np.ndarray = None) -> dict:
    if len(y_true) == 0:
        return {}
    mae   = mean_absolute_error(y_true, y_pred)
    rmse  = np.sqrt(mean_squared_error(y_true, y_pred))
    r2    = r2_score(y_true, y_pred) if len(np.unique(y_true)) > 1 else np.nan
    mbe   = float(np.mean(y_pred - y_true))
    max_e = float(np.max(np.abs(y_pred - y_true)))
    w1    = float(np.mean(np.abs(y_pred - y_true) <= 1.0))
    w2    = float(np.mean(np.abs(y_pred - y_true) <= 2.0))
    out   = dict(n_obs=len(y_true), mae=round(mae, 4), rmse=round(rmse, 4),
                 r2=round(r2, 4), mbe=round(mbe, 4), max_abs_err=round(max_e, 4),
                 within_1pct=round(w1, 4), within_2pct=round(w2, 4))
    if y_std is not None and np.isfinite(y_std).any():
        for z, label in [(1.645, "picp90"), (1.0, "picp68")]:
            lo = y_pred - z * y_std
            hi = y_pred + z * y_std
            out[label] = round(float(np.mean((y_true >= lo) & (y_true <= hi))), 4)
        out["mpiw90"]    = round(float(np.mean(3.29 * y_std)), 4)
        out["sharpness"] = round(float(np.mean(y_std)), 4)
    return out


# ── 60 / 90-day forecasting ────────────────────────────────────────────────────

def build_forecast_features(cycles: pd.DataFrame, gpr: GaussianProcessRegressor,
                             scaler: StandardScaler, horizons: list = [60, 90]) -> pd.DataFrame:
    """
    For each vehicle's most recent session, project features forward by 60 and 90 days,
    then query the GPR to get forecast SoH ± std.
    """
    rows = []
    for reg, grp in cycles.groupby("registration_number"):
        grp = grp.sort_values("start_time")
        last = grp.iloc[-1]

        # Usage rate: EFC per day
        days_span = float(last.get("days_since_first", 0) or 0)
        efc_total = float(last.get("cum_efc", 0) or 0)
        efc_per_day = (efc_total / days_span) if days_span > 1.0 else 0.5  # fallback 0.5 EFC/day

        # Current feature vector (use last session's values)
        current_feats = {}
        for col in FEATURE_COLS:
            current_feats[col] = float(last.get(col, np.nan) or np.nan)

        current_soh = float(last.get("cycle_soh") or last.get("ekf_soh", np.nan))
        row = {
            "registration_number": reg,
            "current_soh": round(current_soh, 3) if np.isfinite(current_soh) else np.nan,
            "efc_rate_per_day": round(efc_per_day, 4),
        }

        for h in horizons:
            feat_h = current_feats.copy()
            feat_h["cum_efc"] = current_feats["cum_efc"] + efc_per_day * h
            if "days_since_first" in feat_h:
                feat_h["days_since_first"] = current_feats["days_since_first"] + h

            X_h = np.array([[feat_h.get(c, np.nan) for c in FEATURE_COLS]])
            # Impute missing features with column median from training data
            X_h = np.where(np.isnan(X_h), scaler.mean_, X_h)
            X_h_sc = scaler.transform(X_h)

            try:
                p_res, p_std = gpr.predict(X_h_sc[:, GPR_FEAT_IDX], return_std=True)
                p_mean = float(p_res[0])
                p_std  = float(p_std[0])
                lo = p_mean - 1.645 * p_std
                hi = p_mean + 1.645 * p_std
            except Exception:
                p_mean, p_std, lo, hi = np.nan, np.nan, np.nan, np.nan

            row[f"soh_pred_{h}d"]    = round(p_mean, 3) if np.isfinite(p_mean) else np.nan
            row[f"soh_pred_{h}d_lo"] = round(lo, 3) if np.isfinite(lo) else np.nan
            row[f"soh_pred_{h}d_hi"] = round(hi, 3) if np.isfinite(hi) else np.nan

        # Risk flag based on 90d forecast
        soh_90 = row.get("soh_pred_90d", np.nan)
        delta  = (current_soh - soh_90) if (np.isfinite(current_soh) and np.isfinite(soh_90)) else np.nan
        if not np.isfinite(delta):
            row["risk_flag"] = "unknown"
        elif delta > 3.0:
            row["risk_flag"] = "red"
        elif delta > 1.0:
            row["risk_flag"] = "amber"
        else:
            row["risk_flag"] = "green"

        rows.append(row)

    return pd.DataFrame(rows)


# ── Plots ──────────────────────────────────────────────────────────────────────

def _save(fig, name: str):
    path = os.path.join(PLOTS_DIR, name)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {name}")


def plot_fit_surface(df: pd.DataFrame, gpr: GaussianProcessRegressor,
                     scaler: StandardScaler):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    x_keys = ["cum_efc", "ekf_soh"]
    x_labels = ["Charge Cycles (EFC)", "EKF State of Health (%)"]

    for ax, xk, xl in zip(axes, x_keys, x_labels):
        qg = df[df["is_quality_gated"]]
        ax.scatter(df[df["split"] == "train"][xk],
                   df[df["split"] == "train"]["cycle_soh"],
                   c="#4E79A7", alpha=0.15, s=8, label="Train (all)")
        ax.scatter(df[df["split"] == "test"][xk],
                   df[df["split"] == "test"]["cycle_soh"],
                   c="#F28E2B", alpha=0.25, s=8, label="Test (all)")
        ax.scatter(qg[xk], qg["cycle_soh"],
                   c="#E15759", alpha=0.5, s=18, label="Quality-gated", zorder=4)

        # GPR mean + band over x grid
        x_min, x_max = df[xk].quantile(0.01), df[xk].quantile(0.99)
        x_grid = np.linspace(x_min, x_max, 200)

        # Build feature grid: set all features to training median, vary xk
        med = df[FEATURE_COLS].median()
        grid_feats = np.tile(med.values, (200, 1))
        xi = FEATURE_COLS.index(xk)
        grid_feats[:, xi] = x_grid
        grid_sc = scaler.transform(grid_feats)

        try:
            g_mean, g_std = gpr.predict(grid_sc[:, GPR_FEAT_IDX], return_std=True)
            ax.plot(x_grid, g_mean, "k-", lw=1.5, label="GPR mean")
            ax.fill_between(x_grid, g_mean - 2*g_std, g_mean + 2*g_std,
                            alpha=0.15, color="steelblue", label="±2σ")
        except Exception:
            pass

        ax.set_xlim(x_min, x_max)
        ax.set_ylim(*SOH_Y_LIM)
        ax.set_xlabel(xl)
        ax.set_ylabel("Battery Health (%)")
        ax.legend(fontsize=7, ncol=2)
        ax.grid(True, alpha=0.3)

    # Annotate with metrics
    test_qg = df[(df["split"] == "test") & df["is_quality_gated"]]
    if "gpr_soh_pred" in df.columns and len(test_qg) > 0:
        mae  = mean_absolute_error(test_qg["cycle_soh"], test_qg["gpr_soh_pred"])
        rmse = np.sqrt(mean_squared_error(test_qg["cycle_soh"], test_qg["gpr_soh_pred"]))
        axes[0].set_title(f"GPR — quality-gated test  MAE={mae:.3f}%  RMSE={rmse:.3f}%",
                          fontsize=10)
    else:
        axes[0].set_title("GPR Matérn-5/2 ARD — SoH vs Charge Cycles", fontsize=10)
    axes[1].set_title("GPR Matérn-5/2 ARD — SoH vs Days in Service", fontsize=10)
    fig.suptitle("GPR SoH Fit Surface", fontsize=12, fontweight="bold")
    fig.tight_layout()
    _save(fig, "gpr_fit_surface.png")


def plot_residuals(df: pd.DataFrame):
    test_qg = df[(df["split"] == "test") & df["is_quality_gated"] & df["gpr_residual"].notna()]
    if len(test_qg) < 5:
        print("  [SKIP] Not enough quality-gated test rows for residual plot.")
        return

    resid = test_qg["gpr_residual"].values
    pred  = test_qg["gpr_soh_pred"].values
    efc   = test_qg["cum_efc"].values

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    fig.suptitle("GPR Residual Diagnostics (quality-gated test set)", fontsize=12, fontweight="bold")

    # Residuals vs EFC
    ax = axes[0, 0]
    ax.scatter(efc, resid, alpha=0.4, s=10, c="#4E79A7")
    ax.axhline(0, c="red", lw=1)
    ax.set_xlabel("Charge Cycles (EFC)")
    ax.set_ylabel("Residual (actual − predicted) %")
    ax.set_title("Residuals vs Charge Cycles")
    ax.grid(True, alpha=0.3)

    # Shapiro-Wilk test
    if len(resid) >= 8:
        try:
            sw_stat, sw_p = stats.shapiro(resid[:5000])
            ax.text(0.02, 0.97, f"Shapiro-Wilk W={sw_stat:.3f}  p={sw_p:.3f}",
                    transform=ax.transAxes, va="top", fontsize=8,
                    color="green" if sw_p > 0.05 else "red")
        except Exception:
            pass

    # Residuals vs fitted
    ax = axes[0, 1]
    ax.scatter(pred, resid, alpha=0.4, s=10, c="#F28E2B")
    ax.axhline(0, c="red", lw=1)
    ax.set_xlabel("Predicted SoH (%)")
    ax.set_ylabel("Residual %")
    ax.set_title("Residuals vs Fitted (heteroscedasticity check)")
    ax.grid(True, alpha=0.3)
    try:
        rho, p_rho = stats.spearmanr(np.abs(resid), efc)
        ax.text(0.02, 0.97, f"Spearman |resid| vs EFC: ρ={rho:.3f} p={p_rho:.3f}",
                transform=ax.transAxes, va="top", fontsize=8,
                color="red" if (abs(rho) > 0.3 and p_rho < 0.05) else "green")
    except Exception:
        pass

    # Q-Q plot
    ax = axes[1, 0]
    try:
        (osm, osr), (slope, intercept, r) = stats.probplot(resid)
        ax.plot(osm, osr, "o", alpha=0.4, ms=3, c="#59A14F")
        ax.plot(osm, slope * np.array(osm) + intercept, "r-", lw=1.5)
    except Exception:
        ax.text(0.5, 0.5, "Q-Q unavailable", transform=ax.transAxes, ha="center")
    ax.set_title("Q-Q Plot (normality of residuals)")
    ax.set_xlabel("Theoretical Quantiles")
    ax.set_ylabel("Residual Quantiles")
    ax.grid(True, alpha=0.3)

    # Histogram
    ax = axes[1, 1]
    ax.hist(resid, bins=40, color="#B07AA1", alpha=0.7, density=True)
    x_plot = np.linspace(resid.min(), resid.max(), 200)
    ax.plot(x_plot, stats.norm.pdf(x_plot, resid.mean(), resid.std()), "k-", lw=1.5)
    ax.set_xlabel("Residual %")
    ax.set_ylabel("Density")
    ax.set_title(f"Residual Distribution  skew={stats.skew(resid):.3f}  kurt={stats.kurtosis(resid):.3f}")
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    _save(fig, "gpr_residuals.png")


def plot_calibration(df: pd.DataFrame):
    test_qg = df[(df["split"] == "test") & df["is_quality_gated"] &
                 df["gpr_soh_pred"].notna() & df["gpr_soh_std"].notna()]
    if len(test_qg) < 10:
        print("  [SKIP] Not enough quality-gated test rows for calibration plot.")
        return

    y    = test_qg["cycle_soh"].values
    mu   = test_qg["gpr_soh_pred"].values
    sig  = test_qg["gpr_soh_std"].values

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("GPR Uncertainty Calibration", fontsize=12, fontweight="bold")

    # Reliability diagram
    ax = axes[0]
    nominals  = np.arange(0.05, 1.0, 0.05)
    empirical = []
    for p in nominals:
        z_val = stats.norm.ppf((1 + p) / 2)
        inside = np.mean((y >= mu - z_val * sig) & (y <= mu + z_val * sig))
        empirical.append(inside)
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Perfect calibration")
    ax.plot(nominals, empirical, "o-", c="#4E79A7", lw=2, ms=6, label="GPR")
    picp90 = empirical[np.argmin(np.abs(nominals - 0.90))]
    ax.axvline(0.90, c="red", lw=0.8, linestyle=":")
    ax.axhline(picp90, c="#4E79A7", lw=0.8, linestyle=":")
    ax.text(0.92, 0.02, f"PICP90={picp90:.2f}", c="#4E79A7", fontsize=9)
    ax.set_xlabel("Nominal Coverage")
    ax.set_ylabel("Empirical Coverage")
    ax.set_title("Reliability Diagram")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # y_std vs cum_efc
    ax = axes[1]
    sc = ax.scatter(test_qg["cum_efc"], sig,
                    c=test_qg["cycle_soh"], cmap="RdYlGn_r",
                    alpha=0.5, s=12)
    plt.colorbar(sc, ax=ax, label="Actual Battery Health (%)")
    ax.set_xlabel("Charge Cycles (EFC)")
    ax.set_ylabel("GPR Posterior Std (σ)")
    ax.set_title("Uncertainty vs Cycle Age\n(higher σ where data is sparse)")
    ax.grid(True, alpha=0.3)

    # Rug of training density
    train_efc = df[df["split"] == "train"]["cum_efc"]
    ax.plot(train_efc, np.full(len(train_efc), ax.get_ylim()[0]),
            "|", c="steelblue", alpha=0.05, ms=6)

    fig.tight_layout()
    _save(fig, "gpr_calibration.png")


def plot_per_vehicle(df: pd.DataFrame):
    vehicles = sorted(df["registration_number"].unique())
    n = len(vehicles)
    ncols = 6
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 3.5, nrows * 2.8))
    axes = np.array(axes).flatten()

    fig.suptitle("GPR Per-Vehicle SoH (quality-gated obs + GPR mean ± 1σ)",
                 fontsize=11, fontweight="bold")

    for i, reg in enumerate(vehicles):
        ax   = axes[i]
        vdf  = df[df["registration_number"] == reg].sort_values("cum_efc")
        qg   = vdf[vdf["is_quality_gated"]]
        test = vdf[vdf["split"] == "test"]

        ax.fill_between(vdf["cum_efc"].values,
                        vdf["gpr_soh_pred"].values - vdf["gpr_soh_std"].values,
                        vdf["gpr_soh_pred"].values + vdf["gpr_soh_std"].values,
                        alpha=0.2, color="steelblue")
        ax.plot(vdf["cum_efc"], vdf["gpr_soh_pred"], c="steelblue", lw=1.2, label="GPR mean")
        ax.scatter(qg["cum_efc"], qg["cycle_soh"], c="#E15759", s=14, zorder=4, label="Quality obs")

        # Train/test boundary
        if len(test) > 0:
            cutoff_efc = test["cum_efc"].min()
            ax.axvline(cutoff_efc, c="grey", lw=0.8, linestyle="--", alpha=0.6)

        # Annotate MAE on test quality-gated
        tq = vdf[(vdf["split"] == "test") & vdf["is_quality_gated"]]
        if len(tq) >= 2:
            mae_v = mean_absolute_error(tq["cycle_soh"], tq["gpr_soh_pred"])
            ax.set_title(f"{reg}\nMAE={mae_v:.2f}%", fontsize=7, pad=2)
        else:
            ax.set_title(reg, fontsize=7, pad=2)

        ax.set_ylim(*SOH_Y_LIM)
        ax.axhline(EOL_SOH, c="red", lw=0.7, linestyle=":", alpha=0.6)
        ax.set_xlabel("EFC", fontsize=6)
        ax.set_ylabel("SoH %", fontsize=6)
        ax.tick_params(labelsize=5)
        ax.grid(True, alpha=0.25)

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    fig.tight_layout()
    _save(fig, "gpr_per_vehicle.png")


def plot_learning_curve(cycles: pd.DataFrame, scaler_full: StandardScaler,
                        kernel_base, X_test_sc: np.ndarray, y_test: np.ndarray,
                        y_std_test: np.ndarray):
    train_qg = cycles[(cycles["split"] == "train") & cycles["is_quality_gated"]]
    sizes = [s for s in [500, 1500, len(train_qg)] if s <= len(train_qg)]
    sizes = sorted(set(sizes))

    rmses, picp90s = [], []
    print("  Learning curve ...")
    for sz in sizes:
        idx   = np.random.choice(len(train_qg), size=min(sz, len(train_qg)), replace=False)
        sub   = train_qg.iloc[idx]
        X_sub_full = sub[FEATURE_COLS].values.copy()
        y_sub_full = sub["cycle_soh"].values

        # Impute
        col_medians = np.nanmedian(X_sub_full, axis=0)
        for ci in range(X_sub_full.shape[1]):
            X_sub_full[np.isnan(X_sub_full[:, ci]), ci] = col_medians[ci]

        sc_tmp    = StandardScaler().fit(X_sub_full)
        X_sub_sc  = sc_tmp.transform(X_sub_full)[:, GPR_FEAT_IDX]

        gp_tmp = GaussianProcessRegressor(
            kernel=kernel_base.clone_with_theta(kernel_base.theta),
            n_restarts_optimizer=0,
            normalize_y=True,
            random_state=SEED,
        )
        try:
            gp_tmp.fit(X_sub_sc, y_sub_full)
            X_t_full = scaler_full.inverse_transform(X_test_sc)
            X_t_tmp  = sc_tmp.transform(X_t_full)[:, GPR_FEAT_IDX]
            p_m, p_s = gp_tmp.predict(X_t_tmp, return_std=True)
            rmse_v = np.sqrt(mean_squared_error(y_test, p_m))
            lo = p_m - 1.645 * p_s
            hi = p_m + 1.645 * p_s
            picp = float(np.mean((y_test >= lo) & (y_test <= hi)))
        except Exception:
            rmse_v, picp = np.nan, np.nan
        rmses.append(rmse_v)
        picp90s.append(picp)
        print(f"    N={sz:>6,}  RMSE={rmse_v:.4f}  PICP90={picp:.3f}")

    fig, ax1 = plt.subplots(figsize=(8, 5))
    ax2 = ax1.twinx()
    ax1.plot(sizes, rmses, "o-", c="#E15759", lw=2, ms=7, label="Test RMSE (%)")
    ax2.plot(sizes, picp90s, "s--", c="#4E79A7", lw=2, ms=7, label="PICP90")
    ax2.axhline(0.90, c="#4E79A7", lw=0.8, linestyle=":", alpha=0.5)
    ax1.axvline(len(train_qg), c="grey", lw=1, linestyle="--", alpha=0.7,
                label=f"Current N={len(train_qg):,}")
    ax1.set_xlabel("Training Set Size (quality-gated rows)")
    ax1.set_ylabel("Test RMSE (%)", color="#E15759")
    ax2.set_ylabel("PICP90", color="#4E79A7")
    ax1.set_title("GPR Learning Curve", fontsize=11, fontweight="bold")
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, fontsize=9)
    ax1.grid(True, alpha=0.3)
    fig.tight_layout()
    _save(fig, "gpr_learning_curve.png")


def plot_feature_importance(gpr: GaussianProcessRegressor):
    """Stakeholder-facing: ARD length scales → feature importance."""
    try:
        kernel = gpr.kernel_
        # Navigate to Matern kernel to find length_scale (may be wrapped in Product/Sum)
        from sklearn.gaussian_process.kernels import Product, Sum
        def _find_matern(k):
            if isinstance(k, Matern):
                return k
            if isinstance(k, (Product, Sum)):
                for sub in [k.k1, k.k2]:
                    found = _find_matern(sub)
                    if found is not None:
                        return found
            return None

        matern_k = _find_matern(kernel)
        if matern_k is None or not hasattr(matern_k, "length_scale"):
            print("  [SKIP] Could not extract ARD length scales from fitted kernel.")
            return

        ls = np.atleast_1d(matern_k.length_scale)
        if len(ls) != len(FEATURE_COLS):
            print(f"  [SKIP] length_scale dim {len(ls)} != n_features {len(FEATURE_COLS)}")
            return

        importance = 1.0 / ls
        importance = importance / importance.max()  # normalise to [0, 1]

        labels   = [FEATURE_LABELS.get(c, c) for c in FEATURE_COLS]
        cats     = [FEATURE_CATEGORY.get(c, "Other") for c in FEATURE_COLS]
        colors   = [CATEGORY_COLORS.get(cat, "#666666") for cat in cats]

        order = np.argsort(importance)
        imp_s  = importance[order]
        lbl_s  = [labels[i] for i in order]
        col_s  = [colors[i] for i in order]
        cat_s  = [cats[i] for i in order]

        fig, ax = plt.subplots(figsize=(9, 6))
        bars = ax.barh(range(len(imp_s)), imp_s, color=col_s, height=0.65, alpha=0.85)
        ax.set_yticks(range(len(imp_s)))
        ax.set_yticklabels(lbl_s, fontsize=9)
        ax.set_xlabel("Relative Feature Importance (ARD kernel, normalised)", fontsize=10)
        ax.set_title("Factors Driving Battery Degradation\n(GPR ARD Kernel — higher bar = stronger driver)",
                     fontsize=11, fontweight="bold")
        ax.set_xlim(0, 1.12)
        for b, v in zip(bars, imp_s):
            ax.text(v + 0.01, b.get_y() + b.get_height() / 2,
                    f"{v:.3f}", va="center", fontsize=8)

        # Legend for categories
        from matplotlib.patches import Patch
        legend_els = [Patch(facecolor=CATEGORY_COLORS[c], label=c, alpha=0.85)
                      for c in sorted(set(cats)) if c in CATEGORY_COLORS]
        ax.legend(handles=legend_els, loc="lower right", fontsize=8, title="Category")
        ax.grid(True, axis="x", alpha=0.3)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        fig.tight_layout()
        _save(fig, "gpr_feature_importance.png")

    except Exception as e:
        print(f"  [WARN] Feature importance plot failed: {e}")


def plot_fleet_forecast(forecast_df: pd.DataFrame, cycles: pd.DataFrame,
                        gpr: GaussianProcessRegressor, scaler: StandardScaler):
    """Stakeholder-facing: 60 & 90-day SoH forecast per vehicle, calendar date x-axis."""
    vehicles = sorted(forecast_df["registration_number"].unique())
    n = len(vehicles)
    ncols = min(6, n)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 3.8, nrows * 3.0))
    axes = np.array(axes).flatten()

    fig.suptitle("GPR SoH Forecast — 60 & 90 Day Horizon\n"
                 "Solid: historical  |  Dashed: forecast  |  Red line: end-of-life (80%)",
                 fontsize=11, fontweight="bold")

    risk_color = {"red": "#E15759", "amber": "#F28E2B", "green": "#59A14F", "unknown": "#888888"}

    for i, reg in enumerate(vehicles):
        ax  = axes[i]
        vdf = cycles[cycles["registration_number"] == reg].sort_values("start_dt")
        frow = forecast_df[forecast_df["registration_number"] == reg].iloc[0]
        rflag = frow.get("risk_flag", "unknown")

        # Historical SoH line
        if vdf["start_dt"].notna().any():
            ax.plot(vdf["start_dt"], vdf["cycle_soh"], c="#4E79A7", lw=1.4,
                    alpha=0.7, label="Battery Health")

            # Forecast points
            last_dt = vdf["start_dt"].max()
            for h, col_fc, col_band in [(60, "#4E79A7", "#4E79A7"),
                                         (90, "#E15759", "#E15759")]:
                soh_fc = frow.get(f"soh_pred_{h}d", np.nan)
                lo_fc  = frow.get(f"soh_pred_{h}d_lo", np.nan)
                hi_fc  = frow.get(f"soh_pred_{h}d_hi", np.nan)
                if np.isfinite(soh_fc):
                    dt_fc = last_dt + pd.Timedelta(days=h)
                    ax.plot([last_dt, dt_fc], [frow.get("current_soh", soh_fc), soh_fc],
                            "--", c=col_fc, lw=1.6)
                    ax.errorbar(dt_fc, soh_fc,
                                yerr=[[soh_fc - lo_fc], [hi_fc - soh_fc]] if np.isfinite(lo_fc) else None,
                                fmt="o", c=col_fc, ms=5, capsize=3)
                    ax.text(dt_fc, soh_fc - 0.8, f"+{h}d: {soh_fc:.1f}%",
                            fontsize=6, ha="center", color=col_fc)

        ax.axhline(EOL_SOH, c="red", lw=0.8, linestyle=":", alpha=0.8)
        ax.set_ylim(*SOH_Y_LIM)
        ax.set_title(f"{reg}", fontsize=7, color=risk_color.get(rflag, "black"),
                     fontweight="bold", pad=2)
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b%y"))
        ax.tick_params(labelsize=5, axis="x", rotation=30)
        ax.tick_params(labelsize=5, axis="y")
        ax.set_ylabel("Battery Health (%)", fontsize=6)
        ax.grid(True, alpha=0.25)

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D
    legend_els = [
        Line2D([0], [0], color="#4E79A7", lw=1.4, label="Historical SoH"),
        Line2D([0], [0], color="#4E79A7", lw=1.6, linestyle="--", label="60-day forecast"),
        Line2D([0], [0], color="#E15759", lw=1.6, linestyle="--", label="90-day forecast"),
        Line2D([0], [0], color="red", lw=0.8, linestyle=":", label="End-of-life 80%"),
        Patch(facecolor="#E15759", label="High risk (Δ>3%)"),
        Patch(facecolor="#F28E2B", label="Medium risk"),
        Patch(facecolor="#59A14F", label="Low risk"),
    ]
    fig.legend(handles=legend_els, loc="lower center", ncol=4, fontsize=8,
               bbox_to_anchor=(0.5, -0.01), framealpha=0.9)
    fig.tight_layout()
    _save(fig, "gpr_fleet_forecast_60_90.png")


# ── Main ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 70)
    print("gpr_soh.py — Gaussian Process Regression for SoH")
    print("=" * 70)

    # ── Load ──────────────────────────────────────────────────────────────────
    cycles = load_data()
    cycles = make_train_test_split(cycles)

    max_days = cycles["days_since_first"].max() if "days_since_first" in cycles.columns else 0
    max_efc  = cycles["cum_efc"].max() if "cum_efc" in cycles.columns else 0
    if max_days < 180:
        print(f"\n  YOUNG FLEET WARNING: {max_days:.0f} days, {max_efc:.0f} EFC. "
              "SoH range 88-100%. RUL priors dominate data.\n")

    # ── Feature matrix ─────────────────────────────────────────────────────────
    print(f"Building feature matrix ({len(FEATURE_COLS)} features) ...")

    X_all = cycles[FEATURE_COLS].values.copy().astype(float)
    y_all = cycles["cycle_soh"].values.copy().astype(float)

    # Impute missing features with column-wise median (per-feature, computed from all rows)
    col_medians = np.nanmedian(X_all, axis=0)
    for ci in range(X_all.shape[1]):
        nan_mask = np.isnan(X_all[:, ci])
        X_all[nan_mask, ci] = col_medians[ci]

    train_mask = cycles["split"].values == "train"
    test_mask  = ~train_mask
    qg_mask    = cycles["is_quality_gated"].values

    X_train, y_train = X_all[train_mask], y_all[train_mask]
    X_test,  y_test  = X_all[test_mask],  y_all[test_mask]

    # Quality-gated subsets
    X_train_qg = X_all[train_mask & qg_mask]
    y_train_qg = y_all[train_mask & qg_mask]
    X_test_qg  = X_all[test_mask & qg_mask]
    y_test_qg  = y_all[test_mask & qg_mask]

    # GPR predicts cycle_soh directly; ekf_soh is retained as a feature (trajectory prior).
    # Residual formulation was removed: ekf_soh ≈ cycle_soh by EKF design, so the
    # residual was near-zero noise and destroyed the learnable signal.

    # Registration numbers for stratified subsampling
    regs_train_qg = cycles.loc[train_mask & qg_mask, "registration_number"].values

    print(f"  Train: {train_mask.sum():,} rows  ({(train_mask & qg_mask).sum():,} quality-gated)")
    print(f"  Test : {test_mask.sum():,} rows   ({(test_mask & qg_mask).sum():,} quality-gated)")

    # ── Collinearity check ────────────────────────────────────────────────────
    efc_idx   = FEATURE_COLS.index("cum_efc")
    ekf_idx   = FEATURE_COLS.index("ekf_soh") if "ekf_soh" in FEATURE_COLS else None
    if ekf_idx is not None:
        ekf_vals = X_all[:, ekf_idx]
        efc_vals = X_all[:, efc_idx]
        finite   = np.isfinite(ekf_vals) & np.isfinite(efc_vals)
        if finite.sum() > 10:
            r_col, _ = stats.pearsonr(efc_vals[finite], ekf_vals[finite])
            print(f"  Pearson r(cum_efc, ekf_soh) = {r_col:.3f}"
                  f"  {'[collinear]' if abs(r_col) > 0.9 else '[OK]'}")

    # ── Scale ─────────────────────────────────────────────────────────────────
    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train_sc    = scaler.transform(X_train)
    X_test_sc     = scaler.transform(X_test)
    X_train_qg_sc = scaler.transform(X_train_qg) if len(X_train_qg) > 0 else X_train_qg
    X_test_qg_sc  = scaler.transform(X_test_qg)  if len(X_test_qg)  > 0 else X_test_qg

    # ── Fit GPR ───────────────────────────────────────────────────────────────
    # ARD Matérn-5/2 on GPR_FEATURE_COLS (11 features; 3 ARD-boundary features
    # excluded).  Target is cycle_soh directly; ekf_soh is a feature.
    # Subsampling is vehicle-stratified so all 66 vehicles are represented.
    MAX_FIT_ROWS = 4000   # covers all ~3,786 QG training rows without truncation
    n_qg = len(X_train_qg_sc)
    if n_qg > MAX_FIT_ROWS:
        # Stratified by vehicle: ~MAX_FIT_ROWS/n_vehicles rows per vehicle
        n_vehicles  = len(np.unique(regs_train_qg))
        n_per_veh   = max(5, MAX_FIT_ROWS // n_vehicles)
        rng         = np.random.default_rng(SEED)
        idx_sub     = []
        for reg in np.unique(regs_train_qg):
            veh_idx = np.where(regs_train_qg == reg)[0]
            chosen  = rng.choice(veh_idx, size=min(len(veh_idx), n_per_veh), replace=False)
            idx_sub.extend(chosen.tolist())
        idx_sub = np.array(idx_sub)
        if len(idx_sub) > MAX_FIT_ROWS:   # trim to cap
            idx_sub = rng.choice(idx_sub, size=MAX_FIT_ROWS, replace=False)
        print(f"  Subsampling training set to {len(idx_sub)} quality-gated rows "
              f"(stratified by vehicle) for GPR fit ...")
        X_fit     = X_train_qg_sc[idx_sub][:, GPR_FEAT_IDX]
        y_fit_res = y_train_qg[idx_sub]
    else:
        X_fit     = X_train_qg_sc[:, GPR_FEAT_IDX]
        y_fit_res = y_train_qg

    n_feats = len(GPR_FEATURE_COLS)
    kernel_base = Matern(
        length_scale=np.ones(n_feats),
        length_scale_bounds=[(0.05, 100.0)] * n_feats,
        nu=2.5,
    )

    print(f"  Fitting GPR on {len(X_fit):,} quality-gated training samples ...")
    gpr = GaussianProcessRegressor(
        kernel=kernel_base,
        n_restarts_optimizer=2,
        normalize_y=True,
        random_state=SEED,
        alpha=0.01,
    )
    gpr.fit(X_fit, y_fit_res)
    print(f"  Log marginal likelihood: {gpr.log_marginal_likelihood_value_:.3f}")
    print(f"  Optimised kernel: {gpr.kernel_}")

    # Save kernel params
    try:
        from sklearn.gaussian_process.kernels import Product, Sum
        def _find_matern(k):
            if isinstance(k, Matern):
                return k
            if isinstance(k, (Product, Sum)):
                for sub in [k.k1, k.k2]:
                    f = _find_matern(sub)
                    if f is not None:
                        return f
            return None

        mk = _find_matern(gpr.kernel_)
        ls_dict = {}
        if mk is not None:
            ls = np.atleast_1d(mk.length_scale)
            if len(ls) == len(GPR_FEATURE_COLS):
                ls_dict = {GPR_FEATURE_COLS[i]: float(ls[i]) for i in range(len(GPR_FEATURE_COLS))}

        kernel_info = {
            "kernel_str":              str(gpr.kernel_),
            "log_marginal_likelihood": float(gpr.log_marginal_likelihood_value_),
            "length_scales":           ls_dict,
            "importance_1_over_ls":    {k: round(1.0/v, 6) for k, v in ls_dict.items() if v > 0},
        }
        with open(KERNEL_JSON, "w") as f:
            json.dump(kernel_info, f, indent=2)
        print(f"  Saved kernel params → {KERNEL_JSON}")
        if ls_dict:
            print("  ARD length scales (shorter = more important):")
            for feat, ls_val in sorted(ls_dict.items(), key=lambda x: x[1]):
                print(f"    {FEATURE_LABELS.get(feat, feat):45s} {ls_val:.4f}")
    except Exception as e:
        print(f"  [WARN] Could not save kernel params: {e}")

    # ── Kernel sensitivity ────────────────────────────────────────────────────
    # Fits on residuals (cycle_soh - ekf_soh) using GPR_FEATURE_COLS.
    # RBF removed — Matern-5/2 dominates for battery SoH and RBF hangs on ≥11 dims.
    print("\n  Kernel sensitivity comparison ...")
    X_test_qg_sc_gpr = X_test_qg_sc[:, GPR_FEAT_IDX]
    kernels_to_compare = {
        "Matern-5/2 ARD":        kernel_base,
        "Matern-5/2 ARD + White": kernel_base + WhiteKernel(noise_level=0.1),
    }
    print(f"  {'Kernel':<30} {'test_RMSE':>10} {'PICP90':>8} {'log_ML':>10}")
    for kname, kern in kernels_to_compare.items():
        try:
            gp_tmp = GaussianProcessRegressor(
                kernel=kern, n_restarts_optimizer=0,
                normalize_y=True, random_state=SEED, alpha=0.1,
            )
            gp_tmp.fit(X_fit, y_fit_res)
            pm, ps = gp_tmp.predict(X_test_qg_sc_gpr, return_std=True)
            rmse_k = np.sqrt(mean_squared_error(y_test_qg, pm)) if len(y_test_qg) > 0 else np.nan
            picp_k = float(np.mean((y_test_qg >= pm - 1.645*ps) & (y_test_qg <= pm + 1.645*ps))) \
                     if len(y_test_qg) > 0 else np.nan
            logml_k = gp_tmp.log_marginal_likelihood_value_
            print(f"  {kname:<30} {rmse_k:>10.4f} {picp_k:>8.3f} {logml_k:>10.3f}")
        except Exception as e:
            print(f"  {kname:<30} {'ERROR':>10}  ({e})")

    # ── Predict on all sessions ────────────────────────────────────────────────
    print("\nGenerating predictions on all sessions ...")
    X_all_sc = scaler.transform(X_all)
    y_pred_all, y_std_all = gpr.predict(X_all_sc[:, GPR_FEAT_IDX], return_std=True)
    cycles = cycles.copy()
    cycles["gpr_soh_pred"] = np.round(y_pred_all, 4)
    cycles["gpr_soh_std"]  = np.round(y_std_all,  4)
    cycles["gpr_residual"] = np.round(cycles["cycle_soh"] - cycles["gpr_soh_pred"], 4)

    # ── Metrics ───────────────────────────────────────────────────────────────
    print("\nComputing metrics ...")
    metrics_rows = []
    for split_name, split_mask in [("train", train_mask), ("test", test_mask)]:
        for subset_name, subset_mask in [("all", np.ones(len(cycles), dtype=bool)),
                                          ("quality_gated", qg_mask)]:
            mask = (split_mask & subset_mask
                    & cycles["gpr_soh_pred"].notna().values
                    & cycles["cycle_soh"].notna().values)
            if mask.sum() == 0:
                continue
            yt = cycles.loc[mask, "cycle_soh"].values
            yp = cycles.loc[mask, "gpr_soh_pred"].values
            ys = cycles.loc[mask, "gpr_soh_std"].values
            m  = compute_metrics(yt, yp, ys)
            m["split"]       = split_name
            m["eval_subset"] = subset_name
            metrics_rows.append(m)
            print(f"  [{split_name}/{subset_name}] n={m['n_obs']:,}  "
                  f"MAE={m.get('mae','NA'):.4f}  RMSE={m.get('rmse','NA'):.4f}  "
                  f"R²={m.get('r2','NA'):.4f}  "
                  f"PICP90={m.get('picp90','NA')}")

    metrics_df = pd.DataFrame(metrics_rows)
    metrics_df.to_csv(METRICS_CSV, index=False)
    print(f"  Saved {METRICS_CSV}")

    # Per-vehicle metrics
    veh_rows = []
    for reg, vdf in cycles.groupby("registration_number"):
        tr = vdf[vdf["split"] == "train"]
        te = vdf[(vdf["split"] == "test") & vdf["is_quality_gated"] & vdf["gpr_soh_pred"].notna()]
        if len(te) < 2:
            continue
        mae_v  = mean_absolute_error(te["cycle_soh"], te["gpr_soh_pred"])
        rmse_v = np.sqrt(mean_squared_error(te["cycle_soh"], te["gpr_soh_pred"]))
        veh_rows.append({
            "registration_number": reg,
            "n_train": len(tr),
            "n_test_quality_gated": len(te),
            "mae_test": round(mae_v, 4),
            "rmse_test": round(rmse_v, 4),
            "mean_std_test": round(te["gpr_soh_std"].mean(), 4),
        })
    pd.DataFrame(veh_rows).sort_values("mae_test", ascending=False).to_csv(VEH_METRICS_CSV, index=False)
    print(f"  Saved {VEH_METRICS_CSV}")

    # ── 60/90-day forecast ────────────────────────────────────────────────────
    print("\nBuilding 60/90-day forecasts ...")
    forecast_df = build_forecast_features(cycles, gpr, scaler, horizons=[60, 90])
    forecast_df.to_csv(FORECAST_CSV, index=False)
    print(f"  Saved {FORECAST_CSV}")
    print(f"\n  Risk summary:")
    for rf in ["red", "amber", "green"]:
        n = (forecast_df["risk_flag"] == rf).sum()
        print(f"    {rf.upper():<8} {n:>3} vehicles")

    # Add 60/90d columns to predictions CSV
    cycles = cycles.merge(
        forecast_df[["registration_number", "soh_pred_60d", "soh_pred_60d_lo", "soh_pred_60d_hi",
                     "soh_pred_90d", "soh_pred_90d_lo", "soh_pred_90d_hi"]],
        on="registration_number", how="left",
    )

    # Save predictions CSV
    pred_cols = [
        "registration_number", "session_id", "start_time",
        "cum_efc", "days_since_first", "cycle_soh", "is_quality_gated",
        "gpr_soh_pred", "gpr_soh_std", "split", "gpr_residual",
        "soh_pred_60d", "soh_pred_60d_lo", "soh_pred_60d_hi",
        "soh_pred_90d", "soh_pred_90d_lo", "soh_pred_90d_hi",
    ]
    pred_cols = [c for c in pred_cols if c in cycles.columns]
    cycles[pred_cols].to_csv(PRED_CSV, index=False)
    print(f"  Saved {PRED_CSV}")

    # ── Plots ─────────────────────────────────────────────────────────────────
    print("\nGenerating plots ...")
    plot_fit_surface(cycles, gpr, scaler)
    plot_residuals(cycles)
    plot_calibration(cycles)
    plot_per_vehicle(cycles)

    # Learning curve (uses quality-gated test as fixed holdout)
    plot_learning_curve(
        cycles, scaler, kernel_base,
        X_test_qg_sc, y_test_qg,
        cycles[test_mask & qg_mask]["gpr_soh_std"].values,
    )

    plot_feature_importance(gpr)
    plot_fleet_forecast(forecast_df, cycles, gpr, scaler)

    # ── Final summary ─────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("GPR SOH SUMMARY")
    print("=" * 70)
    best_test = metrics_df[(metrics_df["split"] == "test") &
                            (metrics_df["eval_subset"] == "quality_gated")]
    if len(best_test) > 0:
        row = best_test.iloc[0]
        print(f"  Quality-gated test MAE  : {row.get('mae', 'NA')}")
        print(f"  Quality-gated test RMSE : {row.get('rmse', 'NA')}")
        print(f"  Quality-gated test R²   : {row.get('r2', 'NA')}")
        print(f"  PICP90                  : {row.get('picp90', 'NA')}")
        print(f"  Sharpness (mean σ)      : {row.get('sharpness', 'NA')}")
    print(f"\n  Outputs in: {ARTIFACTS_DIR}")
    print(f"  Plots  in : {PLOTS_DIR}")
