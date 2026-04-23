import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) if "__file__" in dir() else os.getcwd())

"""
lstm_soh.py — LSTM with self-attention for SoH trend forecasting.

Model
-----
30-session rolling window → predict next session's cycle_soh.
Each session in the window is a 12-feature vector from cycles.csv.
The LSTM operates at session level — NOT on raw
10-second BMS timesteps — because sessions have variable lengths (50-5000 rows)
making raw-timestep batching impractical for 66 vehicles.

Architecture: LSTM(64) → Self-Attention(1 head, 64 dim) → Dense(32, relu) → Dense(1)
Loss weighting: quality-gated sessions (anchor points) weighted 3×; interpolated rows 1×.
MC-Dropout (p=0.3, 5 forward passes) gives prediction intervals at inference time.

Business outputs
----------------
  • 60-day and 90-day SoH forecast per vehicle (autoregressive, MC-dropout)
  • Maintenance priority ranking (90-day degradation Δ per vehicle)

Outputs
-------
  artifacts/lstm_predictions.csv
  artifacts/lstm_metrics.csv
  artifacts/lstm_per_vehicle_metrics.csv
  artifacts/lstm_training_history.csv
  artifacts/lstm_latent_states.npy
  artifacts/fleet_forecast_lstm.csv
  plots/lstm_training_curves.png
  plots/lstm_soh_trajectory.png
  plots/lstm_residuals.png
  plots/lstm_latent_space.png
  plots/lstm_window_attribution.png
  plots/lstm_fleet_forecast_60_90.png   ← stakeholder-facing
  plots/lstm_maintenance_priority.png   ← stakeholder-facing
"""

import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from scipy import stats
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.cluster import KMeans
from sklearn.metrics import davies_bouldin_score

from config import (
    CYCLES_CSV, ARTIFACTS_DIR, PLOTS_DIR,
    EOL_SOH, CYCLE_SOH_OBS_CAP, CYCLE_SOH_MIN_BLOCK_DOD, SEED,
)

warnings.filterwarnings("ignore")
np.random.seed(SEED)

# ── Paths ──────────────────────────────────────────────────────────────────────
PRED_CSV         = os.path.join(ARTIFACTS_DIR, "lstm_predictions.csv")
METRICS_CSV      = os.path.join(ARTIFACTS_DIR, "lstm_metrics.csv")
VEH_METRICS_CSV  = os.path.join(ARTIFACTS_DIR, "lstm_per_vehicle_metrics.csv")
HISTORY_CSV      = os.path.join(ARTIFACTS_DIR, "lstm_training_history.csv")
LATENT_NPY       = os.path.join(ARTIFACTS_DIR, "lstm_latent_states.npy")
FORECAST_CSV     = os.path.join(ARTIFACTS_DIR, "fleet_forecast_lstm.csv")
MODEL_PATH       = os.path.join(ARTIFACTS_DIR, "lstm_soh_model.keras")

# ── Sequence & model config ────────────────────────────────────────────────────
WINDOW         = 30          # sessions per input window
HIDDEN_DIM     = 64
DROPOUT        = 0.30
L2_REG         = 1e-3
EPOCHS         = 120
BATCH_SIZE     = 64
PATIENCE       = 15
TRAIN_FRAC     = 0.80
MC_SAMPLES     = 5           # MC-dropout forward passes for uncertainty
QUALITY_WEIGHT = 3.0         # loss weight for quality-gated target sessions
BASE_WEIGHT    = 1.0
SOH_Y_LIM      = (85, 102)

# 12 sequence features (from cycles.csv)
SEQ_FEATURE_COLS = [
    "cycle_soh", "ir_ohm_mean", "cell_spread_mean", "temp_rise_rate",
    "cum_efc", "days_since_first", "soc_range", "n_vsag",
    "energy_per_km", "dod_stress", "capacity_ah_discharge", "soc_diff",
]

SEQ_FEATURE_LABELS = {
    "cycle_soh":            "Battery Health (%)",
    "ir_ohm_mean":          "Internal Resistance (avg)",
    "cell_spread_mean":     "Cell Imbalance (avg)",
    "temp_rise_rate":       "Temp Rise Rate",
    "cum_efc":              "Charge Cycles",
    "days_since_first":     "Days in Service",
    "soc_range":            "SoC Swing",
    "n_vsag":               "Voltage Sag Events",
    "energy_per_km":        "Energy/km",
    "dod_stress":           "DoD Stress",
    "capacity_ah_discharge":"Discharged Capacity (Ah)",
    "soc_diff":             "SoC Difference",
}


# ── Data loading ───────────────────────────────────────────────────────────────

def load_data() -> pd.DataFrame:
    print(f"Loading cycles.csv ...")
    cycles = pd.read_csv(CYCLES_CSV, low_memory=False)
    print(f"  {len(cycles):,} sessions, {cycles['registration_number'].nunique()} vehicles")


    # Ensure all feature cols exist
    for col in SEQ_FEATURE_COLS:
        if col not in cycles.columns:
            cycles[col] = np.nan

    # Quality gate
    block_dod = cycles.get(
        "block_soc_diff", cycles.get("soc_range", pd.Series(0.0, index=cycles.index))
    ).abs()
    cycles["is_quality_gated"] = (
        cycles["cycle_soh"].notna() &
        (cycles["cycle_soh"] < CYCLE_SOH_OBS_CAP) &
        (block_dod >= CYCLE_SOH_MIN_BLOCK_DOD)
    )

    try:
        cycles["start_dt"] = pd.to_datetime(
            cycles["start_time"], unit="ms", utc=True
        ).dt.tz_convert("Asia/Kolkata")
    except Exception:
        try:
            cycles["start_dt"] = pd.to_datetime(cycles["start_time"])
        except Exception:
            cycles["start_dt"] = pd.NaT

    return cycles


def make_train_test_split(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(["registration_number", "start_time"]).copy()
    df["split"] = "train"
    for _, grp in df.groupby("registration_number"):
        n = len(grp)
        cutoff = int(n * TRAIN_FRAC)
        df.loc[grp.index[cutoff:], "split"] = "test"
    return df


# ── Window builder ─────────────────────────────────────────────────────────────

def build_windows(cycles: pd.DataFrame):
    """
    For each vehicle, build (X_window, y_label, meta) tuples.
    X_window : (WINDOW, n_features)  — past WINDOW sessions
    y_label  : scalar cycle_soh at session i+WINDOW
    meta     : dict with session_id, split, is_quality_gated, registration_number
    """
    # Per-feature column-wise median for imputation (computed fleet-wide)
    feat_medians = cycles[SEQ_FEATURE_COLS].median().values

    windows, labels, weights, metas = [], [], [], []

    for reg, grp in cycles.groupby("registration_number"):
        grp = grp.sort_values("start_time").reset_index(drop=True)

        # Feature matrix for this vehicle — impute NaN with fleet medians
        feat_mat = grp[SEQ_FEATURE_COLS].values.copy().astype(float)
        for ci in range(feat_mat.shape[1]):
            nan_mask = np.isnan(feat_mat[:, ci])
            feat_mat[nan_mask, ci] = feat_medians[ci]

        # Normalise cycle_soh to [0, 1] range within the window? No — keep in % for interpretability.
        n = len(grp)
        if n <= WINDOW:
            continue

        for i in range(n - WINDOW):
            win  = feat_mat[i: i + WINDOW]          # (WINDOW, n_feats)
            tgt_row = grp.iloc[i + WINDOW]
            y    = float(tgt_row.get("cycle_soh", np.nan) or np.nan)

            if not np.isfinite(y):
                continue

            split_val = tgt_row.get("split", "train")
            is_qg     = bool(tgt_row.get("is_quality_gated", False))
            w         = QUALITY_WEIGHT if is_qg else BASE_WEIGHT

            windows.append(win)
            labels.append(y)
            weights.append(w)
            metas.append({
                "registration_number": reg,
                "session_id":          tgt_row.get("session_id"),
                "start_time":          tgt_row.get("start_time"),
                "start_dt":            tgt_row.get("start_dt"),
                "cum_efc":             float(tgt_row.get("cum_efc", np.nan) or np.nan),
                "is_quality_gated":    is_qg,
                "split":               split_val,
                "cycle_soh_label":     y,
            })

    X = np.array(windows, dtype=np.float32)          # (N, WINDOW, n_feats)
    y = np.array(labels,  dtype=np.float32)
    w = np.array(weights, dtype=np.float32)
    meta_df = pd.DataFrame(metas)
    return X, y, w, meta_df


# ── Model ─────────────────────────────────────────────────────────────────────

def build_model(n_features: int):
    """LSTM + dot-product self-attention + Dense head."""
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers, regularizers

    reg = regularizers.l2(L2_REG)

    inp  = keras.Input(shape=(WINDOW, n_features), name="seq_input")

    # LSTM — return sequences for attention
    x = layers.LSTM(HIDDEN_DIM, return_sequences=True, dropout=DROPOUT,
                    recurrent_dropout=0.0,
                    kernel_regularizer=reg, name="lstm")(inp, training=True)

    # Dot-product self-attention (single head)
    attn_out = layers.Attention(use_scale=True, name="self_attn")([x, x])
    x = layers.Add(name="residual")([x, attn_out])
    x = layers.LayerNormalization(name="ln")(x)

    # Take last timestep
    x = layers.Lambda(lambda t: t[:, -1, :], name="last_step")(x)

    x = layers.Dense(32, activation="relu", kernel_regularizer=reg, name="dense1")(x)
    x = layers.Dropout(DROPOUT, name="dropout_head")(x, training=True)  # training=True → always on (MC-dropout)
    out = layers.Dense(1, name="output")(x)

    model = keras.Model(inp, out, name="LSTM_SoH")
    model.compile(
        optimizer=keras.optimizers.AdamW(learning_rate=1e-3, weight_decay=L2_REG),
        loss="mse",
    )
    return model


def get_latent_model(model):
    """Return sub-model that outputs the last-step LSTM hidden state (before dense head)."""
    import tensorflow as tf
    from tensorflow import keras
    try:
        last_step_layer = model.get_layer("last_step")
        return keras.Model(model.input, last_step_layer.output)
    except Exception:
        return None


# ── Metrics ───────────────────────────────────────────────────────────────────

def compute_metrics(y_true, y_pred, label="") -> dict:
    if len(y_true) < 2:
        return {}
    mae  = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2   = r2_score(y_true, y_pred) if len(np.unique(y_true)) > 1 else np.nan
    mbe  = float(np.mean(y_pred - y_true))
    rng  = float(y_true.max() - y_true.min())
    rel_rmse = rmse / rng if rng > 0 else np.nan

    # Directional accuracy (consecutive pairs)
    dir_acc = np.nan
    if len(y_true) > 1:
        delta_true = np.diff(y_true)
        delta_pred = np.diff(y_pred)
        nonzero    = delta_true != 0
        if nonzero.sum() > 0:
            dir_acc = float(np.mean(np.sign(delta_pred[nonzero]) == np.sign(delta_true[nonzero])))

    return dict(
        label=label,
        n_obs=len(y_true),
        mae=round(mae, 4),
        rmse=round(rmse, 4),
        r2=round(float(r2), 4),
        mbe=round(mbe, 4),
        within_1pct=round(float(np.mean(np.abs(y_pred - y_true) <= 1.0)), 4),
        within_2pct=round(float(np.mean(np.abs(y_pred - y_true) <= 2.0)), 4),
        relative_rmse=round(float(rel_rmse), 4) if np.isfinite(rel_rmse) else None,
        directional_accuracy=round(dir_acc, 4) if np.isfinite(dir_acc) else None,
    )


# ── Forecast helpers ───────────────────────────────────────────────────────────

def mc_predict(model, X_batch: np.ndarray, n_samples: int = MC_SAMPLES) -> tuple:
    """Run MC-dropout forward passes; return (mean, std) arrays."""
    import tensorflow as tf
    preds = np.stack([
        model(X_batch, training=True).numpy().flatten()
        for _ in range(n_samples)
    ], axis=0)                             # (n_samples, N)
    return preds.mean(axis=0), preds.std(axis=0)


def build_forecast(cycles: pd.DataFrame, model, feat_medians: np.ndarray,
                   horizons: list = [60, 90]) -> pd.DataFrame:
    """
    Autoregressive forecast for each vehicle.
    Start with last WINDOW real sessions; iteratively predict one step ahead
    until we have covered the horizon in days.
    MC-dropout over MC_SAMPLES passes gives p05/p95.
    """
    rows = []
    for reg, grp in cycles.groupby("registration_number"):
        grp = grp.sort_values("start_time").reset_index(drop=True)

        feat_mat = grp[SEQ_FEATURE_COLS].values.copy().astype(float)
        for ci in range(feat_mat.shape[1]):
            nan_mask = np.isnan(feat_mat[:, ci])
            feat_mat[nan_mask, ci] = feat_medians[ci]

        if len(grp) < WINDOW:
            continue

        last = grp.iloc[-1]
        current_soh = float(last.get("cycle_soh") or last.get("ekf_soh", np.nan))

        # Estimate avg session gap in days
        days_span   = float(last.get("days_since_first", 0) or 0)
        efc_total   = float(last.get("cum_efc", 0) or 0)
        n_sessions  = len(grp)
        avg_day_gap = (days_span / n_sessions) if n_sessions > 0 and days_span > 0 else 0.5
        efc_per_day = (efc_total / days_span) if days_span > 1 else 0.5

        # Seed window: last WINDOW sessions
        window = feat_mat[-WINDOW:].copy()

        row = {
            "registration_number": reg,
            "current_soh": round(current_soh, 3) if np.isfinite(current_soh) else np.nan,
        }

        for h in horizons:
            n_steps = max(1, int(round(h / avg_day_gap)))
            win_h   = window.copy()

            # Track MC predictions at horizon step
            soh_idx  = SEQ_FEATURE_COLS.index("cycle_soh")
            efc_idx  = SEQ_FEATURE_COLS.index("cum_efc")
            days_idx = SEQ_FEATURE_COLS.index("days_since_first")

            current_efc  = float(last.get("cum_efc", 0) or 0)
            current_days = float(last.get("days_since_first", 0) or 0)

            step_preds = []
            try:
                import tensorflow as tf
                for step in range(n_steps):
                    X_in = win_h[np.newaxis].astype(np.float32)   # (1, WINDOW, n_feats)
                    p_mc = np.array([
                        model(X_in, training=True).numpy()[0, 0]
                        for _ in range(MC_SAMPLES)
                    ])
                    step_pred = float(p_mc.mean())
                    step_preds.append(p_mc)

                    # Shift window: append new synthetic row
                    new_row = win_h[-1].copy()
                    new_row[soh_idx]  = step_pred
                    new_row[efc_idx]  = current_efc + efc_per_day * (step + 1) * avg_day_gap
                    new_row[days_idx] = current_days + (step + 1) * avg_day_gap
                    win_h = np.roll(win_h, -1, axis=0)
                    win_h[-1] = new_row

                # Final horizon: stats over MC samples at last step
                final_mc = step_preds[-1]
                p50 = float(np.mean(final_mc))
                p05 = float(np.percentile(final_mc, 5))
                p95 = float(np.percentile(final_mc, 95))
            except Exception as e:
                p50, p05, p95 = np.nan, np.nan, np.nan

            row[f"soh_pred_{h}d"]    = round(p50, 3) if np.isfinite(p50) else np.nan
            row[f"soh_pred_{h}d_lo"] = round(p05, 3) if np.isfinite(p05) else np.nan
            row[f"soh_pred_{h}d_hi"] = round(p95, 3) if np.isfinite(p95) else np.nan

        # Risk flag: 90d delta
        soh_90  = row.get("soh_pred_90d", np.nan)
        delta   = (current_soh - soh_90) if (np.isfinite(current_soh) and np.isfinite(soh_90)) else np.nan
        row["risk_flag"]        = ("red" if delta > 3.0 else "amber" if delta > 1.0 else "green") \
                                  if np.isfinite(delta) else "unknown"
        row["soh_delta_90d"]    = round(delta, 3) if np.isfinite(delta) else np.nan
        rows.append(row)

    return pd.DataFrame(rows)


# ── Plots ──────────────────────────────────────────────────────────────────────

def _save(fig, name: str):
    path = os.path.join(PLOTS_DIR, name)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {name}")


def plot_training_curves(history_df: pd.DataFrame, best_epoch: int):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle("LSTM Training Curves", fontsize=12, fontweight="bold")

    ax = axes[0]
    ax.plot(history_df["epoch"], history_df["train_loss"], label="Train loss", c="#4E79A7", lw=1.5)
    ax.plot(history_df["epoch"], history_df["val_loss"],   label="Val loss",   c="#E15759", lw=1.5)
    ax.axvline(best_epoch, c="grey", lw=1, linestyle="--", label=f"Best epoch {best_epoch}")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE Loss")
    ax.set_title("Loss Curves")
    ax.legend()
    ax.grid(True, alpha=0.3)

    final_train = history_df["train_loss"].iloc[-1]
    final_val   = history_df["val_loss"].iloc[-1]
    overfit_ratio = final_val / final_train if final_train > 0 else np.nan
    ax.text(0.02, 0.97,
            f"Final train={final_train:.4f}  val={final_val:.4f}  ratio={overfit_ratio:.2f}",
            transform=ax.transAxes, va="top", fontsize=8,
            color="red" if overfit_ratio > 2.0 else "green")

    ax = axes[1]
    if "grad_norm" in history_df.columns and history_df["grad_norm"].notna().any():
        ax.plot(history_df["epoch"], history_df["grad_norm"], c="#59A14F", lw=1.2)
        ax.axhline(1e-6, c="red",  lw=0.8, linestyle=":", label="Vanishing threshold 1e-6")
        ax.axhline(100,  c="orange", lw=0.8, linestyle=":", label="Exploding threshold 100")
        ax.set_yscale("log")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Gradient Norm (log scale)")
        ax.set_title("Gradient Norm per Epoch")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    else:
        ax.text(0.5, 0.5, "Gradient norm not recorded", ha="center", transform=ax.transAxes)
        ax.set_title("Gradient Norm")

    fig.tight_layout()
    _save(fig, "lstm_training_curves.png")


def plot_soh_trajectories(cycles: pd.DataFrame, meta_df: pd.DataFrame,
                           y_pred_all: np.ndarray):
    vehicles = sorted(cycles["registration_number"].unique())
    ncols = 6
    nrows = int(np.ceil(len(vehicles) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 3.5, nrows * 2.8))
    axes = np.array(axes).flatten()
    fig.suptitle("LSTM SoH Predictions vs Actual (per vehicle)", fontsize=11, fontweight="bold")

    for i, reg in enumerate(vehicles):
        ax   = axes[i]
        vdf  = cycles[cycles["registration_number"] == reg].sort_values("cum_efc")
        vmeta = meta_df[meta_df["registration_number"] == reg].copy()

        # Actual quality-gated observations
        qg = vdf[vdf["is_quality_gated"]]
        ax.scatter(qg["cum_efc"], qg["cycle_soh"],
                   c="grey", s=10, alpha=0.5, zorder=2, label="Obs")

        # LSTM predictions (test only)
        test_meta = vmeta[vmeta["split"] == "test"]
        if len(test_meta) > 0 and "lstm_soh_pred" in test_meta.columns:
            ax.plot(test_meta["cum_efc"], test_meta["lstm_soh_pred"],
                    c="#E15759", lw=1.2, zorder=3, label="LSTM test")

        # Train boundary
        train_meta = vmeta[vmeta["split"] == "train"]
        if len(train_meta) > 0:
            cutoff_efc = vmeta[vmeta["split"] == "test"]["cum_efc"].min() if len(test_meta) > 0 else None
            if cutoff_efc is not None and np.isfinite(cutoff_efc):
                ax.axvline(cutoff_efc, c="grey", lw=0.8, linestyle="--", alpha=0.5)

        # MAE annotation
        tq = test_meta[(test_meta["is_quality_gated"]) & test_meta["lstm_soh_pred"].notna()] \
             if "lstm_soh_pred" in test_meta.columns else pd.DataFrame()
        title = reg
        if len(tq) >= 2:
            mae_v = mean_absolute_error(tq["cycle_soh_label"], tq["lstm_soh_pred"])
            title = f"{reg}\nMAE={mae_v:.2f}%"

        ax.set_title(title, fontsize=7, pad=2)
        ax.set_ylim(*SOH_Y_LIM)
        ax.axhline(EOL_SOH, c="red", lw=0.7, linestyle=":", alpha=0.6)
        ax.set_xlabel("EFC", fontsize=6)
        ax.set_ylabel("SoH %", fontsize=6)
        ax.tick_params(labelsize=5)
        ax.grid(True, alpha=0.25)

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    fig.tight_layout()
    _save(fig, "lstm_soh_trajectory.png")


def plot_residuals(meta_df: pd.DataFrame):
    test_qg = meta_df[
        (meta_df["split"] == "test") &
        meta_df["is_quality_gated"] &
        meta_df["lstm_soh_pred"].notna()
    ]
    if len(test_qg) < 5:
        print("  [SKIP] Not enough quality-gated test rows for residual plot.")
        return

    resid = (test_qg["cycle_soh_label"] - test_qg["lstm_soh_pred"]).values
    pred  = test_qg["lstm_soh_pred"].values
    efc   = test_qg["cum_efc"].values

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    fig.suptitle("LSTM Residual Diagnostics (quality-gated test)", fontsize=12, fontweight="bold")

    ax = axes[0, 0]
    ax.scatter(efc, resid, alpha=0.4, s=10, c="#4E79A7")
    ax.axhline(0, c="red", lw=1)
    ax.set_xlabel("Charge Cycles (EFC)")
    ax.set_ylabel("Residual %")
    ax.set_title("Residuals vs Charge Cycles")
    ax.grid(True, alpha=0.3)
    if len(resid) >= 8:
        try:
            sw_stat, sw_p = stats.shapiro(resid[:5000])
            ax.text(0.02, 0.97, f"Shapiro-Wilk p={sw_p:.3f}",
                    transform=ax.transAxes, va="top", fontsize=8,
                    color="green" if sw_p > 0.05 else "red")
        except Exception:
            pass

    ax = axes[0, 1]
    ax.scatter(pred, resid, alpha=0.4, s=10, c="#F28E2B")
    ax.axhline(0, c="red", lw=1)
    ax.set_xlabel("Predicted SoH (%)")
    ax.set_ylabel("Residual %")
    ax.set_title("Residuals vs Fitted")
    ax.grid(True, alpha=0.3)

    ax = axes[1, 0]
    try:
        (osm, osr), (slope, intercept, _) = stats.probplot(resid)
        ax.plot(osm, osr, "o", alpha=0.4, ms=3, c="#59A14F")
        ax.plot(osm, slope * np.array(osm) + intercept, "r-", lw=1.5)
    except Exception:
        pass
    ax.set_title("Q-Q Plot")
    ax.set_xlabel("Theoretical Quantiles")
    ax.set_ylabel("Residual Quantiles")
    ax.grid(True, alpha=0.3)

    ax = axes[1, 1]
    ax.hist(resid, bins=40, color="#B07AA1", alpha=0.7, density=True)
    x_p = np.linspace(resid.min(), resid.max(), 200)
    ax.plot(x_p, stats.norm.pdf(x_p, resid.mean(), resid.std()), "k-", lw=1.5)
    ax.set_xlabel("Residual %")
    ax.set_ylabel("Density")
    ax.set_title(f"Residual Histogram  skew={stats.skew(resid):.3f}")
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    _save(fig, "lstm_residuals.png")


def plot_latent_space(latent_states: np.ndarray, meta_df: pd.DataFrame):
    if latent_states is None or len(latent_states) < 10:
        return
    try:
        from umap import UMAP
        reducer = UMAP(n_components=2, random_state=SEED, n_neighbors=15, min_dist=0.1)
        emb = reducer.fit_transform(latent_states)
        method = "UMAP"
    except ImportError:
        from sklearn.manifold import TSNE
        n_s = min(len(latent_states), 3000)
        idx = np.random.choice(len(latent_states), n_s, replace=False)
        latent_states = latent_states[idx]
        meta_df = meta_df.iloc[idx].reset_index(drop=True)
        reducer = TSNE(n_components=2, random_state=SEED, perplexity=30)
        emb = reducer.fit_transform(latent_states)
        method = "t-SNE"

    efc_vals = meta_df["cum_efc"].values
    reg_vals = pd.Categorical(meta_df["registration_number"]).codes

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle(f"LSTM Latent Space ({method}) — does the encoder learn aging structure?",
                 fontsize=11, fontweight="bold")

    sc = axes[0].scatter(emb[:, 0], emb[:, 1], c=efc_vals, cmap="RdYlGn_r",
                         alpha=0.5, s=8)
    plt.colorbar(sc, ax=axes[0], label="Charge Cycles (EFC)")
    axes[0].set_title("Coloured by Charge Cycles\n(clustered = encoder learned aging)")
    axes[0].set_xlabel(f"{method} 1")
    axes[0].set_ylabel(f"{method} 2")
    axes[0].grid(True, alpha=0.2)

    sc2 = axes[1].scatter(emb[:, 0], emb[:, 1], c=reg_vals, cmap="tab20",
                          alpha=0.5, s=8)
    axes[1].set_title("Coloured by Vehicle ID\n(clustered by vehicle = overfitting risk)")
    axes[1].set_xlabel(f"{method} 1")
    axes[1].set_ylabel(f"{method} 2")
    axes[1].grid(True, alpha=0.2)

    fig.tight_layout()
    _save(fig, "lstm_latent_space.png")


def plot_window_attribution(model, X_sample: np.ndarray, n_feats: int):
    """Approximate feature × timestep importance via gradient × input."""
    if X_sample is None or len(X_sample) == 0:
        return
    try:
        import tensorflow as tf
        X_t = tf.Variable(X_sample[:50].astype(np.float32))
        with tf.GradientTape() as tape:
            preds = model(X_t, training=False)
        grads = tape.gradient(preds, X_t).numpy()          # (batch, WINDOW, n_feats)
        attr  = np.abs(grads * X_sample[:50]).mean(axis=0) # (WINDOW, n_feats)
        attr  = attr / (attr.max() + 1e-9)

        feat_labels = [SEQ_FEATURE_LABELS.get(c, c) for c in SEQ_FEATURE_COLS]

        fig, ax = plt.subplots(figsize=(14, 5))
        im = ax.imshow(attr.T, aspect="auto", cmap="YlOrRd", interpolation="nearest")
        ax.set_xticks(range(WINDOW))
        ax.set_xticklabels([f"{WINDOW - j}" for j in range(WINDOW)],
                           fontsize=7, rotation=45, ha="right")
        ax.set_yticks(range(n_feats))
        ax.set_yticklabels(feat_labels, fontsize=8)
        ax.set_xlabel("Window position (1 = most recent session, left = oldest)")
        ax.set_title("Feature × Timestep Attribution (gradient × input)\n"
                     "Brighter = model relies more on that feature at that time step",
                     fontsize=10, fontweight="bold")
        plt.colorbar(im, ax=ax, label="Normalised attribution")
        fig.tight_layout()
        _save(fig, "lstm_window_attribution.png")
    except Exception as e:
        print(f"  [WARN] Attribution plot failed: {e}")


def plot_fleet_forecast(forecast_df: pd.DataFrame, cycles: pd.DataFrame):
    vehicles = sorted(forecast_df["registration_number"].unique())
    n = len(vehicles)
    ncols = min(6, n)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 3.8, nrows * 3.0))
    axes = np.array(axes).flatten()
    fig.suptitle("LSTM Battery SoH Forecast — Next 60 & 90 Days\n"
                 "Solid: historical  |  Dashed: forecast  |  Red line: end-of-life (80%)",
                 fontsize=11, fontweight="bold")

    risk_color = {"red": "#E15759", "amber": "#F28E2B", "green": "#59A14F", "unknown": "#888888"}

    for i, reg in enumerate(vehicles):
        ax   = axes[i]
        vdf  = cycles[cycles["registration_number"] == reg].sort_values("start_dt")
        frow = forecast_df[forecast_df["registration_number"] == reg].iloc[0]
        rflag = frow.get("risk_flag", "unknown")

        if vdf["start_dt"].notna().any():
            ax.plot(vdf["start_dt"], vdf["cycle_soh"], c="#4E79A7", lw=1.2, alpha=0.7)
            last_dt     = vdf["start_dt"].max()
            current_soh = frow.get("current_soh", np.nan)

            for h, col_fc in [(60, "#4E79A7"), (90, "#E15759")]:
                soh_fc = frow.get(f"soh_pred_{h}d", np.nan)
                lo_fc  = frow.get(f"soh_pred_{h}d_lo", np.nan)
                hi_fc  = frow.get(f"soh_pred_{h}d_hi", np.nan)
                if np.isfinite(soh_fc):
                    dt_fc = last_dt + pd.Timedelta(days=h)
                    ax.plot([last_dt, dt_fc], [current_soh, soh_fc],
                            "--", c=col_fc, lw=1.5)
                    if np.isfinite(lo_fc):
                        ax.fill_between([last_dt, dt_fc],
                                        [current_soh, lo_fc],
                                        [current_soh, hi_fc],
                                        alpha=0.15, color=col_fc)
                    ax.plot(dt_fc, soh_fc, "o", c=col_fc, ms=5)
                    ax.text(dt_fc, soh_fc - 0.8, f"+{h}d\n{soh_fc:.1f}%",
                            fontsize=5.5, ha="center", color=col_fc)

        ax.axhline(EOL_SOH, c="red", lw=0.8, linestyle=":", alpha=0.8)
        ax.set_ylim(*SOH_Y_LIM)
        ax.set_title(reg, fontsize=7,
                     color=risk_color.get(rflag, "black"), fontweight="bold", pad=2)
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b%y"))
        ax.tick_params(labelsize=5, axis="x", rotation=30)
        ax.tick_params(labelsize=5, axis="y")
        ax.set_ylabel("Battery Health (%)", fontsize=6)
        ax.grid(True, alpha=0.25)

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    fig.tight_layout()
    _save(fig, "lstm_fleet_forecast_60_90.png")


def plot_maintenance_priority(forecast_df: pd.DataFrame):
    df = forecast_df.dropna(subset=["soh_delta_90d"]).sort_values(
        "soh_delta_90d", ascending=False
    ).reset_index(drop=True)
    if len(df) == 0:
        return

    risk_color = {"red": "#E15759", "amber": "#F28E2B", "green": "#59A14F", "unknown": "#AAAAAA"}
    colors = [risk_color.get(r, "#AAAAAA") for r in df["risk_flag"]]

    fig, ax = plt.subplots(figsize=(10, max(5, len(df) * 0.35)))
    bars = ax.barh(range(len(df)), df["soh_delta_90d"], color=colors,
                   height=0.65, alpha=0.85)
    ax.set_yticks(range(len(df)))
    ax.set_yticklabels(df["registration_number"], fontsize=8)
    ax.set_xlabel("Projected SoH Drop over 90 Days (%)", fontsize=10)
    ax.set_title("90-Day Degradation Risk — Maintenance Priority\n"
                 "Sorted by projected SoH decline  |  Red = high risk  |  Green = low risk",
                 fontsize=11, fontweight="bold")
    ax.axvline(3.0, c="#E15759", lw=1, linestyle="--", alpha=0.6, label="High-risk threshold (3%)")
    ax.axvline(1.0, c="#F28E2B", lw=1, linestyle="--", alpha=0.6, label="Medium-risk threshold (1%)")

    for bar, val in zip(bars, df["soh_delta_90d"]):
        ax.text(val + 0.05, bar.get_y() + bar.get_height() / 2,
                f"{val:.2f}%", va="center", fontsize=7.5)

    from matplotlib.patches import Patch
    ax.legend(handles=[
        Patch(facecolor="#E15759", label="High risk (Δ > 3%)"),
        Patch(facecolor="#F28E2B", label="Medium risk (1-3%)"),
        Patch(facecolor="#59A14F", label="Low risk (< 1%)"),
    ], loc="lower right", fontsize=9)
    ax.grid(True, axis="x", alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.invert_yaxis()
    fig.tight_layout()
    _save(fig, "lstm_maintenance_priority.png")


# ── Main ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 70)
    print("lstm_soh.py — LSTM + Self-Attention for SoH Trend Forecasting")
    print("=" * 70)

    import tensorflow as tf
    tf.random.set_seed(SEED)
    print(f"  TensorFlow version: {tf.__version__}")

    # ── Load ──────────────────────────────────────────────────────────────────
    cycles = load_data()
    cycles = make_train_test_split(cycles)

    max_days = cycles["days_since_first"].max() if "days_since_first" in cycles.columns else 0
    max_efc  = cycles["cum_efc"].max() if "cum_efc" in cycles.columns else 0
    if max_days < 180:
        print(f"\n  YOUNG FLEET WARNING: {max_days:.0f} days, {max_efc:.0f} EFC. "
              "Predictions dominated by recent session patterns, not long-term degradation.\n")

    # ── Build windows ─────────────────────────────────────────────────────────
    print("Building 30-session rolling windows ...")
    feat_medians = cycles[SEQ_FEATURE_COLS].median().values
    X, y, w, meta_df = build_windows(cycles)

    n_feats = X.shape[2]
    print(f"  Total windows: {len(X):,}")
    print(f"  Quality-gated targets: {(w == QUALITY_WEIGHT).sum():,} "
          f"({100*(w == QUALITY_WEIGHT).mean():.1f}%)")
    print(f"  Train windows: {(meta_df['split'] == 'train').sum():,} | "
          f"Test windows: {(meta_df['split'] == 'test').sum():,}")

    if len(X) < 50:
        print("[WARN] Very few windows — LSTM may not converge. Consider reducing WINDOW size.")

    train_mask = meta_df["split"].values == "train"
    test_mask  = ~train_mask
    qg_mask    = meta_df["is_quality_gated"].values

    X_train, y_train, w_train = X[train_mask], y[train_mask], w[train_mask]
    X_test,  y_test            = X[test_mask],  y[test_mask]

    # ── KS covariate shift check ──────────────────────────────────────────────
    print("\nCovariate shift check (train vs test feature distributions) ...")
    print(f"  {'Feature':<35} {'KS stat':>8} {'p-value':>10} Status")
    for fi, col in enumerate(SEQ_FEATURE_COLS):
        try:
            ks_stat, ks_p = stats.ks_2samp(X_train[:, -1, fi], X_test[:, -1, fi])
            flag = "[EXPECTED - monotonic]" if col in ("cum_efc", "days_since_first") \
                   else ("[SHIFT]" if ks_p < 0.05 else "[OK]")
            print(f"  {col:<35} {ks_stat:>8.4f} {ks_p:>10.4f} {flag}")
        except Exception:
            pass

    # ── Build & train model ───────────────────────────────────────────────────
    print("\nBuilding LSTM model ...")
    model = build_model(n_feats)
    model.summary(print_fn=print)

    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

    callbacks = [
        EarlyStopping(monitor="val_loss", patience=PATIENCE, restore_best_weights=True,
                      verbose=1),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=7, min_lr=1e-5,
                          verbose=1),
        ModelCheckpoint(MODEL_PATH, monitor="val_loss", save_best_only=True, verbose=0),
    ]

    # Gradient norm callback
    class GradNormCallback(tf.keras.callbacks.Callback):
        def __init__(self):
            super().__init__()
            self.grad_norms = []
        def on_epoch_end(self, epoch, logs=None):
            if epoch % 10 == 0:
                with tf.GradientTape() as tape:
                    preds = self.model(X_train[:64], training=True)
                    loss  = tf.reduce_mean((preds[:, 0] - y_train[:64]) ** 2)
                grads = tape.gradient(loss, self.model.trainable_variables)
                gnorm = float(tf.linalg.global_norm(
                    [g for g in grads if g is not None]
                ).numpy())
                self.grad_norms.append((epoch, gnorm))
                if gnorm < 1e-6:
                    print(f"  [WARN epoch {epoch}] Gradient vanishing: norm={gnorm:.2e}")
                elif gnorm > 100:
                    print(f"  [WARN epoch {epoch}] Gradient exploding: norm={gnorm:.2e}")

    grad_cb = GradNormCallback()
    callbacks.append(grad_cb)

    print(f"\nTraining for up to {EPOCHS} epochs (patience={PATIENCE}) ...")
    history = model.fit(
        X_train, y_train,
        sample_weight=w_train,
        validation_split=0.15,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=callbacks,
        verbose=2,
    )

    best_epoch = int(np.argmin(history.history["val_loss"])) + 1

    # Save training history
    hist_rows = []
    for ep in range(len(history.history["loss"])):
        row = {
            "epoch":      ep + 1,
            "train_loss": history.history["loss"][ep],
            "val_loss":   history.history["val_loss"][ep],
            "grad_norm":  np.nan,
        }
        for ep_g, g_val in grad_cb.grad_norms:
            if ep_g == ep:
                row["grad_norm"] = g_val
        hist_rows.append(row)
    history_df = pd.DataFrame(hist_rows)
    history_df.to_csv(HISTORY_CSV, index=False)
    print(f"\n  Best epoch: {best_epoch}")
    print(f"  Final val_loss: {history.history['val_loss'][-1]:.6f}")
    overfit_ratio = history.history["val_loss"][-1] / (history.history["loss"][-1] + 1e-9)
    print(f"  Overfit ratio (val/train): {overfit_ratio:.2f}"
          f"  {'[WARN overfitting]' if overfit_ratio > 2.0 else '[OK]'}")

    # ── Predict ───────────────────────────────────────────────────────────────
    print("\nGenerating predictions ...")
    y_pred_train, _ = mc_predict(model, X_train, n_samples=MC_SAMPLES)
    y_pred_test,  _ = mc_predict(model, X_test,  n_samples=MC_SAMPLES)

    meta_df = meta_df.copy()
    meta_df["lstm_soh_pred"] = np.nan
    meta_df.loc[train_mask, "lstm_soh_pred"] = y_pred_train
    meta_df.loc[test_mask,  "lstm_soh_pred"] = y_pred_test
    meta_df["lstm_residual"] = meta_df["cycle_soh_label"] - meta_df["lstm_soh_pred"]

    # ── Metrics ───────────────────────────────────────────────────────────────
    print("\nComputing metrics ...")
    metrics_rows = []
    for split_name, s_mask in [("train", train_mask), ("test", test_mask)]:
        for subset_name, sub_mask in [("all", np.ones(len(meta_df), dtype=bool)),
                                       ("quality_gated", qg_mask)]:
            m_mask = s_mask & sub_mask & meta_df["lstm_soh_pred"].notna().values
            if m_mask.sum() < 2:
                continue
            yt = meta_df.loc[m_mask, "cycle_soh_label"].values
            yp = meta_df.loc[m_mask, "lstm_soh_pred"].values
            m  = compute_metrics(yt, yp, label=f"{split_name}/{subset_name}")
            m["split"]       = split_name
            m["eval_subset"] = subset_name
            m["val_loss_final"] = round(history.history["val_loss"][-1], 6)
            m["overfit_ratio"]  = round(overfit_ratio, 3)
            m["best_epoch"]     = best_epoch
            metrics_rows.append(m)
            print(f"  [{split_name}/{subset_name}] n={m['n_obs']:,}  "
                  f"MAE={m.get('mae','NA'):.4f}  RMSE={m.get('rmse','NA'):.4f}")

    pd.DataFrame(metrics_rows).to_csv(METRICS_CSV, index=False)

    # Per-vehicle metrics
    veh_rows = []
    for reg in meta_df["registration_number"].unique():
        tq = meta_df[
            (meta_df["registration_number"] == reg) &
            (meta_df["split"] == "test") &
            meta_df["is_quality_gated"] &
            meta_df["lstm_soh_pred"].notna()
        ]
        if len(tq) < 2:
            continue
        mae_v  = mean_absolute_error(tq["cycle_soh_label"], tq["lstm_soh_pred"])
        rmse_v = np.sqrt(mean_squared_error(tq["cycle_soh_label"], tq["lstm_soh_pred"]))
        veh_rows.append({
            "registration_number": reg,
            "n_test_qg": len(tq),
            "mae_test": round(mae_v, 4),
            "rmse_test": round(rmse_v, 4),
        })
    pd.DataFrame(veh_rows).sort_values("mae_test", ascending=False).to_csv(VEH_METRICS_CSV, index=False)

    # ── Latent states (encoder output) ────────────────────────────────────────
    print("\nExtracting latent states ...")
    latent_model = get_latent_model(model)
    latent_states = None
    if latent_model is not None:
        try:
            latent_states = latent_model.predict(X, batch_size=256, verbose=0)
            np.save(LATENT_NPY, latent_states)
            print(f"  Saved latent states: {latent_states.shape} → {LATENT_NPY}")

            # KMeans cluster quality
            if len(latent_states) >= 4:
                km = KMeans(n_clusters=4, random_state=SEED, n_init=10)
                cluster_labels = km.fit_predict(latent_states)
                db_idx = davies_bouldin_score(latent_states, cluster_labels)
                print(f"  KMeans(k=4) Davies-Bouldin index: {db_idx:.4f} (lower = better separation)")
                efc_vals = meta_df["cum_efc"].values
                for k in range(4):
                    mask_k = cluster_labels == k
                    print(f"    Cluster {k}: n={mask_k.sum():,}  mean EFC={efc_vals[mask_k].mean():.1f}")
        except Exception as e:
            print(f"  [WARN] Latent extraction failed: {e}")

    # ── 60/90-day forecast ────────────────────────────────────────────────────
    print("\nBuilding 60/90-day forecasts ...")
    forecast_df = build_forecast(cycles, model, feat_medians)
    forecast_df.to_csv(FORECAST_CSV, index=False)
    print(f"  Saved {FORECAST_CSV}")
    print("  Risk summary:")
    for rf in ["red", "amber", "green"]:
        n = (forecast_df["risk_flag"] == rf).sum()
        print(f"    {rf.upper():<8} {n:>3} vehicles")

    # ── Save predictions CSV ──────────────────────────────────────────────────
    meta_out = meta_df.merge(
        forecast_df[["registration_number",
                     "soh_pred_60d", "soh_pred_60d_lo", "soh_pred_60d_hi",
                     "soh_pred_90d", "soh_pred_90d_lo", "soh_pred_90d_hi"]],
        on="registration_number", how="left",
    )
    pred_cols = [
        "registration_number", "session_id", "start_time", "cum_efc",
        "cycle_soh_label", "is_quality_gated", "lstm_soh_pred", "split",
        "lstm_residual",
        "soh_pred_60d", "soh_pred_60d_lo", "soh_pred_60d_hi",
        "soh_pred_90d", "soh_pred_90d_lo", "soh_pred_90d_hi",
    ]
    meta_out[[c for c in pred_cols if c in meta_out.columns]].to_csv(PRED_CSV, index=False)
    print(f"  Saved {PRED_CSV}")

    # ── Plots ─────────────────────────────────────────────────────────────────
    print("\nGenerating plots ...")
    plot_training_curves(history_df, best_epoch)
    plot_soh_trajectories(cycles, meta_df, None)
    plot_residuals(meta_df)
    plot_latent_space(latent_states, meta_df)
    plot_window_attribution(model, X_test[:50] if len(X_test) > 0 else X_train[:50], n_feats)
    plot_fleet_forecast(forecast_df, cycles)
    plot_maintenance_priority(forecast_df)

    print("\n" + "=" * 70)
    print("LSTM SOH SUMMARY")
    print("=" * 70)
    best_test = [m for m in metrics_rows if m.get("split") == "test" and m.get("eval_subset") == "quality_gated"]
    if best_test:
        row = best_test[0]
        print(f"  Quality-gated test MAE  : {row.get('mae', 'NA')}")
        print(f"  Quality-gated test RMSE : {row.get('rmse', 'NA')}")
        print(f"  Directional accuracy    : {row.get('directional_accuracy', 'NA')}")
        print(f"  Overfit ratio           : {row.get('overfit_ratio', 'NA')}")
    print(f"\n  Outputs in: {ARTIFACTS_DIR}")
    print(f"  Plots  in : {PLOTS_DIR}")
