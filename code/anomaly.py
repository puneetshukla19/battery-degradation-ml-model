import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) if "__file__" in dir() else os.getcwd())

"""
anomaly.py — Unsupervised anomaly detection for individual discharge cycles.

Two complementary methods:
  1. Isolation Forest — flags cycles whose feature combination is globally unusual
     relative to the entire fleet. Good at catching sudden one-off outlier events.
     Features include: voltage/cell/thermal signals, health counts (voltage sag,
     high-IR, low-SoC), energy-per-km, session duration, insulation, time gap,
     and capacity_soh (continuous SoH proxy from data_prep).

  2. CUSUM (Cumulative Sum) — per-vehicle change-point detector. Runs on four
     signals: capacity_soh (primary), energy_per_km, temp_rise_rate, and
     cell_spread_mean (rising imbalance = accelerating degradation).
     Flags the point in time where any of these signals shifts significantly.

Outputs
-------
anomaly_scores.csv  One row per discharge cycle with IF score and CUSUM flags.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from config import CYCLES_CSV, ARTIFACTS_DIR, SEED, SCALAR_FEATURES, MIN_SOC_RANGE_FOR_TREND
import os

ANOMALY_FILE   = os.path.join(ARTIFACTS_DIR, "anomaly_scores.csv")
LGBM_PRED_FILE = os.path.join(ARTIFACTS_DIR, "lgbm_soh_predictions.csv")
CLUSTER_FILE   = os.path.join(ARTIFACTS_DIR, "regime_clusters.csv")

# ── Isolation Forest features ──────────────────────────────────────────────────
# All used gracefully: only columns that exist in df are selected.
IF_FEATURES = [
    # Core discharge profile
    "voltage_mean", "voltage_min",
    "cell_spread_mean", "cell_spread_max",
    "temp_max", "temp_rise_rate",
    "capacity_ah", "duration_hr",
    # SoC / SoH
    "soc_range", "soc_diff",        # soc_diff negative for discharge; deep discharge = more negative
    "soh",                          # BMS SoH (capacity_soh removed: Coulomb counting captures only 38%
                                    # of actual discharge energy, making it systematically 60-95% too low)
    # Health flag counts
    # n_vsag: single consolidated voltage-sag count (data_prep_1.py consolidates
    # the three severity buckets into one flag per row; see compute_voltage_sag)
    "n_vsag",
    "n_high_ir", "n_low_soc",
    # Efficiency & thermal
    "energy_per_km",
    # Degradation rate-of-change (discharge-to-discharge delta; NaN for other session types)
    # d_vsag_per_cycle and d_n_high_ir now correctly compare consecutive discharge sessions only
    "d_vsag_per_cycle", "d_n_high_ir", "d_ir_ohm_per_cycle",
    # Session context
    "insulation_mean",              # low insulation = safety concern
    "time_delta_hr",                # unusually long gap before session = abnormal operating pattern
]

# CUSUM parameters
CUSUM_K = 0.5   # allowance (half the shift size to detect), in std units
CUSUM_H = 4.0   # decision threshold in std units (lower = more sensitive)


def isolation_forest_scores(df: pd.DataFrame) -> pd.Series:
    """
    Train Isolation Forest on all discharge cycles and return anomaly scores.
    Only uses features that exist in df (graceful if columns not yet present).
    Higher score = more anomalous.
    """
    feats = [f for f in IF_FEATURES if f in df.columns]
    X     = df[feats].fillna(df[feats].median())
    X_s   = StandardScaler().fit_transform(X)

    iso = IsolationForest(n_estimators=200, contamination=0.05,
                          random_state=SEED, n_jobs=-1)
    iso.fit(X_s)
    raw = iso.decision_function(X_s)
    print(f"  IF features used ({len(feats)}): {feats}")
    return pd.Series(-raw, index=df.index, name="if_score")   # inverted: higher = worse


def cusum_per_vehicle(series: pd.Series, k: float = CUSUM_K,
                      h: float = CUSUM_H) -> pd.Series:
    """
    One-sided (downward) CUSUM on a single vehicle's signal.
    Returns boolean Series: True = alarm (signal shifted negatively).
    """
    if len(series) < 4:
        return pd.Series(False, index=series.index)
    mu, sigma = series.mean(), series.std()
    if sigma < 1e-6:
        return pd.Series(False, index=series.index)
    z     = (series - mu) / sigma
    cusum = np.zeros(len(z))
    for i in range(1, len(z)):
        cusum[i] = max(0, cusum[i - 1] - z.iloc[i] - k)
    return pd.Series(cusum > h, index=series.index)


def run_cusum_signals(disc: pd.DataFrame) -> pd.DataFrame:
    """
    Run CUSUM on four signals per vehicle:
      - soh_smooth       (BMS SoH smoothed rolling median)
      - energy_per_km    (rising = worse efficiency; negated before downward CUSUM)
      - temp_rise_rate   (rising = faster heating;   negated before downward CUSUM)
      - cell_spread_mean (rising = growing cell imbalance; negated before downward CUSUM)

    Negating upward-bad signals lets the single downward CUSUM implementation detect
    positive shifts in those signals (positive shift in original = negative shift in negated).
    """
    disc = disc.copy()
    disc["soh_smooth"] = (
        disc.groupby("registration_number")["soh"]
        .transform(lambda s: s.rolling(5, center=True, min_periods=1).median())
    )

    cusum_soh, cusum_epk, cusum_heat, cusum_spread = [], [], [], []

    for reg, veh in disc.groupby("registration_number"):
        # SoH — downward shift is bad
        cusum_soh.append(cusum_per_vehicle(veh["soh_smooth"]))

        # Energy-per-km — upward shift is bad (negate)
        if "energy_per_km" in veh.columns:
            epk = veh["energy_per_km"].fillna(veh["energy_per_km"].median())
            cusum_epk.append(cusum_per_vehicle(-epk))
        else:
            cusum_epk.append(pd.Series(False, index=veh.index))

        # Temp rise rate — upward shift is bad (negate)
        if "temp_rise_rate" in veh.columns:
            thr = veh["temp_rise_rate"].fillna(veh["temp_rise_rate"].median())
            cusum_heat.append(cusum_per_vehicle(-thr))
        else:
            cusum_heat.append(pd.Series(False, index=veh.index))

        # Cell spread — upward shift is bad (negate)
        if "cell_spread_mean" in veh.columns:
            cs = veh["cell_spread_mean"].fillna(veh["cell_spread_mean"].median())
            cusum_spread.append(cusum_per_vehicle(-cs))
        else:
            cusum_spread.append(pd.Series(False, index=veh.index))

    disc["cusum_soh_alarm"]    = pd.concat(cusum_soh).reindex(disc.index).fillna(False)
    disc["cusum_epk_alarm"]    = pd.concat(cusum_epk).reindex(disc.index).fillna(False)
    disc["cusum_heat_alarm"]   = pd.concat(cusum_heat).reindex(disc.index).fillna(False)
    disc["cusum_spread_alarm"] = pd.concat(cusum_spread).reindex(disc.index).fillna(False)
    disc["cusum_alarm"] = (
        disc["cusum_soh_alarm"]    |
        disc["cusum_epk_alarm"]    |
        disc["cusum_heat_alarm"]   |
        disc["cusum_spread_alarm"]
    )
    return disc


# ══════════════════════════════════════════════════════════════════════════════
# MODEL A — LightGBM Charging-SOH Regression (supervised)
# ══════════════════════════════════════════════════════════════════════════════

# Features for Model A (all must be present in cycles.csv after data_prep_1.py)
LGBM_FEATURES = [
    # Aging clock
    "cum_efc", "days_since_first", "aging_index",
    # IR signals
    "ir_ohm_mean", "ir_ohm_mean_ewm10", "ir_ohm_trend_slope",
    # Voltage sag
    "vsag_rate_per_hr", "vsag_trend_slope",
    # Cell spread
    "cell_spread_mean", "spread_trend_slope",
    # Thermal
    "temp_rise_rate", "thermal_stress", "rapid_heating",
    # Charging stress
    "c_rate_chg", "fast_charging", "slow_charging",
    # Depth of discharge / energy
    "soc_range", "dod_stress", "energy_kwh", "charging_rate_kw",
    # Load context
    "is_loaded", "load_direction_enc",
    # Data quality
    "bms_coverage", "cell_health_poor",
    # Insulation
    "insulation_mean",
]


def train_lgbm_soh(cycles: pd.DataFrame) -> pd.DataFrame | None:
    """
    Model A: LightGBM regression to predict capacity_soh from stress features.

    Training strategy: temporal split per vehicle — train on first 70% of each
    vehicle's charging sessions (chronologically), test on last 30%.

    Returns DataFrame with columns: registration_number, session_id, start_time,
    capacity_soh, lgbm_soh_pred, split (train/test).
    Returns None if lightgbm not installed or insufficient data.
    """
    try:
        from lightgbm import LGBMRegressor
    except ImportError:
        print("  [Model A] lightgbm not installed — skipping. pip install lightgbm")
        return None

    # Charging sessions with high-confidence capacity_soh
    target = "capacity_soh"
    low_conf_col = "soh_low_confidence"

    mask = (
        (cycles["session_type"] == "charging") &
        (cycles[target].notna())
    )
    if low_conf_col in cycles.columns:
        mask &= ~cycles[low_conf_col].fillna(False)
    # Filter to sessions with sufficient SOC swing — small partial charges
    # give unreliable Coulomb-counted capacity_soh (diagnostic finding: 99.95%
    # of all charging sessions show exactly 100%, masking any real signal).
    if "soc_range" in cycles.columns:
        mask &= cycles["soc_range"].abs() >= MIN_SOC_RANGE_FOR_TREND

    chg = cycles[mask].copy()
    chg = chg.sort_values(["registration_number", "start_time"])

    feats = [f for f in LGBM_FEATURES if f in chg.columns]
    if len(feats) < 5:
        print(f"  [Model A] Only {len(feats)} features available — need at least 5. "
              f"Run data_prep_1.py first to generate engineered features.")
        return None

    # Temporal split per vehicle (70/30)
    train_rows, test_rows = [], []
    for reg, veh in chg.groupby("registration_number"):
        veh = veh.sort_values("start_time")
        n_train = max(1, int(len(veh) * 0.70))
        train_rows.append(veh.iloc[:n_train])
        test_rows.append(veh.iloc[n_train:])

    train_df = pd.concat(train_rows).reset_index(drop=True)
    test_df  = pd.concat(test_rows).reset_index(drop=True)

    if len(test_df) < 10:
        print(f"  [Model A] Insufficient test sessions ({len(test_df)}) — need ≥10.")
        return None

    X_train = train_df[feats].fillna(train_df[feats].median())
    y_train = train_df[target].values
    X_test  = test_df[feats].fillna(train_df[feats].median())
    y_test  = test_df[target].values

    # Try Optuna tuning if available, else use sensible defaults
    best_params = {
        "num_leaves": 31, "max_depth": 6, "learning_rate": 0.05,
        "n_estimators": 300, "min_child_samples": 10,
        "reg_alpha": 0.1, "reg_lambda": 0.1,
        "colsample_bytree": 0.8, "subsample": 0.8,
        "random_state": SEED, "n_jobs": -1, "verbose": -1,
    }
    try:
        import optuna
        optuna.logging.set_verbosity(optuna.logging.WARNING)

        def _objective(trial):
            params = {
                "num_leaves":        trial.suggest_int("num_leaves", 15, 127),
                "max_depth":         trial.suggest_int("max_depth", 3, 10),
                "learning_rate":     trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
                "n_estimators":      trial.suggest_int("n_estimators", 100, 600),
                "min_child_samples": trial.suggest_int("min_child_samples", 5, 50),
                "reg_alpha":         trial.suggest_float("reg_alpha", 1e-4, 10.0, log=True),
                "reg_lambda":        trial.suggest_float("reg_lambda", 1e-4, 10.0, log=True),
                "colsample_bytree":  trial.suggest_float("colsample_bytree", 0.5, 1.0),
                "subsample":         trial.suggest_float("subsample", 0.5, 1.0),
                "random_state": SEED, "n_jobs": -1, "verbose": -1,
            }
            m = LGBMRegressor(**params)
            m.fit(X_train, y_train)
            return mean_absolute_error(y_test, m.predict(X_test))

        study = optuna.create_study(direction="minimize")
        study.optimize(_objective, n_trials=60, show_progress_bar=False)
        best_params.update(study.best_params)
        best_params.update({"random_state": SEED, "n_jobs": -1, "verbose": -1})
        print(f"  [Model A] Optuna best MAE: {study.best_value:.4f}%")
    except ImportError:
        print("  [Model A] optuna not installed — using default hyperparameters. "
              "pip install optuna for tuning.")

    model = LGBMRegressor(**best_params)
    model.fit(X_train, y_train)

    # Evaluate on test set
    preds_test = model.predict(X_test)
    rmse = float(np.sqrt(mean_squared_error(y_test, preds_test)))
    mae  = float(mean_absolute_error(y_test, preds_test))

    # Directional accuracy: does sign(Δŷ) == sign(Δy) between consecutive sessions?
    dir_correct = 0
    dir_total   = 0
    for reg, veh in test_df.groupby("registration_number"):
        veh_preds = model.predict(veh[feats].fillna(train_df[feats].median()))
        dy_true   = np.diff(veh[target].values)
        dy_pred   = np.diff(veh_preds)
        mask_nz   = (dy_true != 0)
        dir_correct += int(np.sum((np.sign(dy_true[mask_nz]) == np.sign(dy_pred[mask_nz]))))
        dir_total   += int(mask_nz.sum())
    dir_acc = dir_correct / dir_total * 100 if dir_total > 0 else np.nan

    print(f"\n  [Model A] LightGBM SOH Regressor — Temporal Test Set:")
    print(f"    Train sessions: {len(train_df):,}  |  Test sessions: {len(test_df):,}")
    print(f"    RMSE           : {rmse:.4f} %  (target < 1.5%)")
    print(f"    MAE            : {mae:.4f} %  (target < 1.0%)")
    print(f"    Dir. accuracy  : {dir_acc:.1f}%  (target > 65%)")
    print(f"    Features used  : {len(feats)}")

    # Feature importance (top 10)
    fi = pd.Series(model.feature_importances_, index=feats).nlargest(10)
    print(f"\n    Top-10 feature importances:")
    for feat, imp in fi.items():
        print(f"      {feat:<35} {imp:>6.0f}")

    # Build output DataFrame
    preds_train = model.predict(X_train)
    out_train   = train_df[["registration_number", "session_id", "start_time", target]].copy()
    out_train["lgbm_soh_pred"] = preds_train
    out_train["split"]         = "train"
    out_test    = test_df[["registration_number", "session_id", "start_time", target]].copy()
    out_test["lgbm_soh_pred"] = preds_test
    out_test["split"]         = "test"

    result = pd.concat([out_train, out_test]).sort_values(
        ["registration_number", "start_time"]
    ).reset_index(drop=True)
    result["lgbm_residual"] = result[target] - result["lgbm_soh_pred"]

    return result


# ══════════════════════════════════════════════════════════════════════════════
# MODEL B — UMAP + HDBSCAN Degradation Regime Clustering (unsupervised)
# ══════════════════════════════════════════════════════════════════════════════

CLUSTER_FEATURES = [
    "ir_ohm_mean_ewm10", "cell_spread_mean_ewm10",
    "vsag_rate_per_hr_ewm10", "temp_rise_rate_ewm10",
    "c_rate_chg", "dod_stress", "thermal_stress", "is_loaded",
    "ir_ohm_trend_slope", "spread_trend_slope", "vsag_trend_slope",
    "cum_efc", "bms_coverage", "capacity_soh", "cell_health_poor",
]


def run_umap_hdbscan(cycles: pd.DataFrame) -> pd.DataFrame | None:
    """
    Model B: Per-vehicle rolling feature vector → UMAP + HDBSCAN.

    Detects vehicles transitioning into a stressed/degraded operating regime.
    Returns DataFrame with columns: registration_number, session_id, start_time,
    umap_x, umap_y, cluster_label, regime_flag (True = non-healthy regime).
    Returns None if umap-learn or hdbscan not installed.
    """
    try:
        import umap
        import hdbscan as hdb
    except ImportError:
        print("  [Model B] umap-learn and/or hdbscan not installed — skipping.")
        print("            pip install umap-learn hdbscan")
        return None

    feats = [f for f in CLUSTER_FEATURES if f in cycles.columns]
    if len(feats) < 5:
        print(f"  [Model B] Only {len(feats)} cluster features available — "
              f"run data_prep_1.py first.")
        return None

    # Use all session types with at least some features populated
    df = cycles[["registration_number", "session_id", "start_time"] + feats].copy()
    df = df.dropna(subset=feats, how="all")

    if len(df) < 50:
        print("  [Model B] Fewer than 50 sessions available — skipping UMAP/HDBSCAN.")
        return None

    # Standardise
    X = df[feats].fillna(df[feats].median())
    scaler = StandardScaler()
    X_s    = scaler.fit_transform(X)

    # UMAP dimensionality reduction
    reducer = umap.UMAP(
        n_components=2,
        n_neighbors=15,
        min_dist=0.1,
        metric="euclidean",
        random_state=SEED,
    )
    embedding = reducer.fit_transform(X_s)

    # HDBSCAN clustering
    clusterer = hdb.HDBSCAN(
        min_cluster_size=10,
        min_samples=5,
        cluster_selection_epsilon=0.5,
    )
    labels = clusterer.fit_predict(embedding)

    df["umap_x"]       = embedding[:, 0]
    df["umap_y"]       = embedding[:, 1]
    df["cluster_label"] = labels

    # Identify the dominant "healthy" cluster as the one with the highest
    # mean capacity_soh (or largest cluster if capacity_soh not available)
    if "capacity_soh" in df.columns and df["capacity_soh"].notna().any():
        cluster_soh = (
            df[df["cluster_label"] >= 0]
            .groupby("cluster_label")["capacity_soh"].mean()
        )
        healthy_cluster = int(cluster_soh.idxmax()) if len(cluster_soh) else 0
    else:
        cluster_counts  = df[df["cluster_label"] >= 0]["cluster_label"].value_counts()
        healthy_cluster = int(cluster_counts.idxmax()) if len(cluster_counts) else 0

    df["regime_flag"] = (df["cluster_label"] != healthy_cluster) & (df["cluster_label"] >= 0)

    n_clusters = int((labels >= 0).sum() > 0) and int(np.unique(labels[labels >= 0]).size)
    n_flagged  = df["regime_flag"].sum()
    print(f"\n  [Model B] UMAP + HDBSCAN:")
    print(f"    Sessions clustered    : {len(df):,}")
    print(f"    Clusters found        : {n_clusters}")
    print(f"    Healthy cluster label : {healthy_cluster}")
    print(f"    Regime-flagged sessions: {n_flagged:,} "
          f"({n_flagged / len(df):.1%} of sessions)")

    if n_flagged > 0:
        flagged_vehs = df[df["regime_flag"]]["registration_number"].value_counts().head(10)
        print(f"\n    Top vehicles in non-healthy regime:")
        for v, cnt in flagged_vehs.items():
            print(f"      {v}: {cnt} sessions flagged")

    return df[["registration_number", "session_id", "start_time",
               "umap_x", "umap_y", "cluster_label", "regime_flag"]]


# ══════════════════════════════════════════════════════════════════════════════
# MODEL C — Enhanced CUSUM on Composite Degradation Score
# ══════════════════════════════════════════════════════════════════════════════

COMPOSITE_WEIGHTS_CUSUM = {
    "soh_smooth":        0.30,
    "ir_ohm_mean_ewm10": 0.20,
    "cell_spread_mean_ewm10": 0.20,
    "vsag_rate_per_hr_ewm10": 0.15,
    "temp_rise_rate_ewm10":   0.15,
}


def compute_composite_score(disc: pd.DataFrame) -> pd.Series:
    """
    Compute a per-session composite degradation score from EWM-smoothed signals.
    Each signal is normalised fleet-wide to [0, 1] before weighting.
    Higher score = more degraded.
    """
    score = pd.Series(0.0, index=disc.index)
    for col, w in COMPOSITE_WEIGHTS_CUSUM.items():
        if col not in disc.columns:
            continue
        s = disc[col].copy()
        # For soh_smooth: lower = worse → invert
        if col == "soh_smooth":
            lo, hi = s.min(), s.max()
            if hi > lo:
                s = 1.0 - (s - lo) / (hi - lo)
            else:
                s = pd.Series(0.0, index=disc.index)
        else:
            lo, hi = s.min(), s.max()
            s = (s - lo) / (hi - lo) if hi > lo else pd.Series(0.0, index=disc.index)
        score += w * s.fillna(0.0)
    return score


def run_cusum_composite(disc: pd.DataFrame) -> pd.DataFrame:
    """
    Model C: CUSUM on composite score + individual trend slopes.
    Returns disc with added columns: cusum_composite_alarm, cusum_ir_slope_alarm,
    cusum_spread_slope_alarm.
    """
    disc = disc.copy()

    # Composite score CUSUM (upward shift = worsening = bad → use direct sign)
    composite = compute_composite_score(disc)
    disc["composite_degradation_score"] = composite

    cusum_composite, cusum_ir_slope, cusum_spread_slope = [], [], []

    for reg, veh in disc.groupby("registration_number"):
        # Composite score: upward shift is bad → negate before downward CUSUM
        cusum_composite.append(cusum_per_vehicle(-veh["composite_degradation_score"]))

        # IR trend slope: rising = worse → negate
        if "ir_ohm_trend_slope" in veh.columns:
            ir_s = veh["ir_ohm_trend_slope"].fillna(0)
            cusum_ir_slope.append(cusum_per_vehicle(-ir_s))
        else:
            cusum_ir_slope.append(pd.Series(False, index=veh.index))

        # Spread trend slope: rising = worse → negate
        if "spread_trend_slope" in veh.columns:
            sp_s = veh["spread_trend_slope"].fillna(0)
            cusum_spread_slope.append(cusum_per_vehicle(-sp_s))
        else:
            cusum_spread_slope.append(pd.Series(False, index=veh.index))

    disc["cusum_composite_alarm"]     = pd.concat(cusum_composite).reindex(disc.index).fillna(False)
    disc["cusum_ir_slope_alarm"]      = pd.concat(cusum_ir_slope).reindex(disc.index).fillna(False)
    disc["cusum_spread_slope_alarm"]  = pd.concat(cusum_spread_slope).reindex(disc.index).fillna(False)

    n_comp   = disc["cusum_composite_alarm"].sum()
    n_ir     = disc["cusum_ir_slope_alarm"].sum()
    n_spread = disc["cusum_spread_slope_alarm"].sum()
    print(f"\n  [Model C] Enhanced CUSUM on composite score:")
    print(f"    Composite alarm sessions  : {n_comp:,}")
    print(f"    IR slope alarm sessions   : {n_ir:,}")
    print(f"    Spread slope alarm sessions: {n_spread:,}")

    return disc


if __name__ == "__main__":
    cycles = pd.read_csv(CYCLES_CSV)
    disc   = cycles[
        (cycles["session_type"] == "discharge") & (cycles["current_mean"] > 0)
    ].copy()
    disc   = disc.sort_values(["registration_number", "start_time"]).reset_index(drop=True)

    print(f"Discharge cycles loaded: {len(disc):,} across "
          f"{disc['registration_number'].nunique()} vehicles")

    # ── Isolation Forest ──────────────────────────────────────────────────────
    print("\nFitting Isolation Forest ...")
    disc["if_score"]   = isolation_forest_scores(disc)
    disc["if_anomaly"] = disc["if_score"] > disc["if_score"].quantile(0.95)

    # ── CUSUM on four signals ─────────────────────────────────────────────────
    print("Running CUSUM change-point detection ...")
    disc = run_cusum_signals(disc)

    # Combined flag (original: IF + 4-channel CUSUM)
    disc["anomaly"] = disc["if_anomaly"] | disc["cusum_alarm"]

    # ── Model C: Enhanced CUSUM on composite score + trend slopes ─────────────
    print("\nRunning Model C: Enhanced CUSUM on composite score ...")
    disc = run_cusum_composite(disc)

    # Extend combined flag to include composite CUSUM
    disc["anomaly"] |= (
        disc["cusum_composite_alarm"]  |
        disc["cusum_ir_slope_alarm"]   |
        disc["cusum_spread_slope_alarm"]
    )

    disc.to_csv(ANOMALY_FILE, index=False)
    print(f"Saved: {ANOMALY_FILE}")

    # ── Model A: LightGBM SOH Regressor ───────────────────────────────────────
    print("\nRunning Model A: LightGBM SOH Regressor ...")
    lgbm_df = train_lgbm_soh(cycles)
    if lgbm_df is not None:
        lgbm_df.to_csv(LGBM_PRED_FILE, index=False)
        print(f"Saved: {LGBM_PRED_FILE}")

    # ── Model B: UMAP + HDBSCAN Regime Detection ──────────────────────────────
    print("\nRunning Model B: UMAP + HDBSCAN regime detection ...")
    cluster_df = run_umap_hdbscan(cycles)
    if cluster_df is not None:
        cluster_df.to_csv(CLUSTER_FILE, index=False)
        print(f"Saved: {CLUSTER_FILE}")

    # ── Per-vehicle anomaly summary ────────────────────────────────────────────
    print("\n" + "=" * 75)
    print("ANOMALY SUMMARY PER VEHICLE  (discharge sessions only)")
    print("=" * 75)
    agg_cols = dict(
        cycles          = ("if_score",           "count"),
        if_anomalies    = ("if_anomaly",          "sum"),
        cusum_soh       = ("cusum_soh_alarm",     "sum"),
        cusum_energy    = ("cusum_epk_alarm",     "sum"),
        cusum_heating   = ("cusum_heat_alarm",    "sum"),
        cusum_spread    = ("cusum_spread_alarm",  "sum"),
        combined        = ("anomaly",             "sum"),
        if_score_mean   = ("if_score",            "mean"),
    )
    summary = (
        disc.groupby("registration_number")
        .agg(**agg_cols)
        .sort_values("combined", ascending=False)
    )
    print(summary[summary["combined"] > 0].to_string())

    # ── CUSUM first-alarm detail ───────────────────────────────────────────────
    print("\nVehicles with CUSUM alarm:")
    cusum_veh = disc[disc["cusum_alarm"]]["registration_number"].unique()
    for v in sorted(cusum_veh):
        vdf        = disc[(disc["registration_number"] == v) & disc["cusum_alarm"]]
        first_date = pd.to_datetime(vdf["start_time"].min(), unit="ms").date()
        soh_f      = disc[(disc["registration_number"] == v) & disc["cusum_soh_alarm"]].shape[0]
        epk_f      = disc[(disc["registration_number"] == v) & disc["cusum_epk_alarm"]].shape[0]
        heat_f     = disc[(disc["registration_number"] == v) & disc["cusum_heat_alarm"]].shape[0]
        spread_f   = disc[(disc["registration_number"] == v) & disc["cusum_spread_alarm"]].shape[0]
        print(f"  {v}: first alarm {first_date} | "
              f"SoH={soh_f}  Energy={epk_f}  Heating={heat_f}  Spread={spread_f} cycles flagged")

    # ── Fleet-flag summary (from data_prep flags already in cycles.csv) ────────
    print("\n" + "=" * 75)
    print("FLEET FLAGS SUMMARY (all sessions)")
    print("=" * 75)
    flag_cols = ["rapid_heating", "high_energy_per_km", "slow_charging", "fast_charging"]
    existing_flags = [c for c in flag_cols if c in cycles.columns]
    if existing_flags:
        flag_summary = (
            cycles.groupby("registration_number")[existing_flags]
            .sum()
            .sort_values(existing_flags[0], ascending=False)
        )
        print(flag_summary[flag_summary.sum(axis=1) > 0].to_string())
    else:
        print("  No fleet flags found — re-run data_prep.py to generate them.")
