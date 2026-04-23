import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) if "__file__" in dir() else os.getcwd())

"""
model_eval_comparison.py — Cross-model evaluation and fleet scorecard.

Loads all model outputs (EKF, GPR, LSTM, ECM, PF) and:
  1. Computes a unified metric table on the quality-gated test set (~3,000 rows)
  2. Compares calibration curves (PICP at 50/68/80/90/95%)
  3. Computes pairwise residual correlations and model agreement
  4. Merges all fleet_forecast_*.csv into a consensus table
  5. Produces the fleet scorecard — primary business deliverable

Run this LAST, after all model scripts have completed.

Outputs
-------
artifacts/model_comparison_metrics.csv
artifacts/model_comparison_per_vehicle.csv
artifacts/model_agreement_sessions.csv
artifacts/fleet_scorecard.csv
artifacts/eval_summary_report.txt
plots/eval_model_metrics_bar.png
plots/eval_calibration_comparison.png
plots/eval_soh_fleet_ribbons.png
plots/eval_model_agreement_heatmap.png
plots/eval_rul_comparison.png
plots/eval_uncertainty_comparison.png
plots/eval_fleet_scorecard.png      ← stakeholder
plots/eval_forecast_consensus.png   ← stakeholder
"""

import warnings
import json
from datetime import datetime
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from scipy.stats import pearsonr, spearmanr, norm as sp_norm

from config import CYCLES_CSV, ARTIFACTS_DIR, PLOTS_DIR, EOL_SOH, SEED

np.random.seed(SEED)
warnings.filterwarnings("ignore", category=RuntimeWarning)

# ── File locations ─────────────────────────────────────────────────────────────
EKF_CSV  = os.path.join(ARTIFACTS_DIR, "ekf_soh.csv")
GPR_CSV  = os.path.join(ARTIFACTS_DIR, "gpr_predictions.csv")
LSTM_CSV = os.path.join(ARTIFACTS_DIR, "lstm_predictions.csv")
ECM_CSV  = os.path.join(ARTIFACTS_DIR, "ecm_soh.csv")
PF_CSV   = os.path.join(ARTIFACTS_DIR, "pf_soh.csv")

FLEET_CSVS = {
    "EKF":  os.path.join(ARTIFACTS_DIR, "ekf_soh.csv"),    # no separate fleet file; derive inline
    "GPR":  os.path.join(ARTIFACTS_DIR, "fleet_forecast_gpr.csv"),
    "LSTM": os.path.join(ARTIFACTS_DIR, "fleet_forecast_lstm.csv"),
    "ECM":  os.path.join(ARTIFACTS_DIR, "fleet_forecast_ecm.csv"),
    "PF":   os.path.join(ARTIFACTS_DIR, "fleet_forecast_pf.csv"),
}

OUT_METRICS  = os.path.join(ARTIFACTS_DIR, "model_comparison_metrics.csv")
OUT_PER_VEH  = os.path.join(ARTIFACTS_DIR, "model_comparison_per_vehicle.csv")
OUT_AGREE    = os.path.join(ARTIFACTS_DIR, "model_agreement_sessions.csv")
OUT_SCORECARD = os.path.join(ARTIFACTS_DIR, "fleet_scorecard.csv")
OUT_REPORT   = os.path.join(ARTIFACTS_DIR, "eval_summary_report.txt")

# Visual colours
RISK_RED   = "#c0392b"
RISK_AMBER = "#e67e22"
RISK_GREEN = "#27ae60"
FLEET_BLUE = "#2980b9"
FLEET_GREY = "#95a5a6"

MODEL_COLOURS = {
    "EKF":  "#16a085",
    "GPR":  "#8e44ad",
    "LSTM": "#e67e22",
    "ECM":  "#2980b9",
    "PF":   "#c0392b",
}


# ── Data loading ──────────────────────────────────────────────────────────────

def load_model_predictions() -> dict:
    """Load each model's prediction CSV. Returns dict of DataFrames."""
    loaders = {
        "EKF":  (EKF_CSV,  "ekf_soh",      "ekf_soh_std"),
        "GPR":  (GPR_CSV,  "gpr_soh_pred",  "gpr_soh_std"),
        "LSTM": (LSTM_CSV, "lstm_soh_pred", None),
        "ECM":  (ECM_CSV,  "ecm_soh",       "ecm_soh_std"),
        "PF":   (PF_CSV,   "pf_soh_mean",   "pf_soh_std"),
    }

    dfs = {}
    for name, (path, pred_col, std_col) in loaders.items():
        if not os.path.exists(path):
            print(f"  [SKIP] {name}: {path} not found")
            continue
        df = pd.read_csv(path)
        # Standardise column names
        df = df.rename(columns={pred_col: f"{name}_pred"})
        if std_col and std_col in df.columns:
            df = df.rename(columns={std_col: f"{name}_std"})

        # Ensure required columns
        for c in ["session_id", "registration_number", "cum_efc"]:
            if c not in df.columns:
                print(f"  [WARN] {name} missing column: {c}")

        dfs[name] = df
        print(f"  Loaded {name}: {len(df):,} rows")

    return dfs


def build_common_eval_set(dfs: dict) -> pd.DataFrame:
    """
    Join all models on session_id into a single evaluation frame.
    Only quality-gated test sessions included.
    """
    base_name = next((n for n in ["EKF", "ECM", "PF", "GPR", "LSTM"] if n in dfs), None)
    if base_name is None:
        return pd.DataFrame()

    base = dfs[base_name][["session_id", "registration_number", "cum_efc",
                             "start_time", "cycle_soh_obs", "split",
                             "is_quality_gated",
                             f"{base_name}_pred",
                             f"{base_name}_std"] if f"{base_name}_std" in dfs[base_name].columns
                            else ["session_id", "registration_number", "cum_efc",
                                  "start_time", "cycle_soh_obs", "split",
                                  "is_quality_gated",
                                  f"{base_name}_pred"]].copy()

    for name, df in dfs.items():
        if name == base_name:
            continue
        cols = ["session_id", f"{name}_pred"]
        if f"{name}_std" in df.columns:
            cols.append(f"{name}_std")
        # Include PF quantiles for calibration
        for q in ["p05", "p25", "p75", "p95"]:
            col = f"pf_soh_{q}"
            if col in df.columns:
                cols.append(col)
        base = base.merge(df[cols], on="session_id", how="left")

    # Apply quality-gate + test filter
    has_qg   = "is_quality_gated" in base.columns
    has_split = "split" in base.columns
    if has_qg and has_split:
        eval_df = base[(base["split"] == "test") & (base["is_quality_gated"])].copy()
    elif has_split:
        eval_df = base[base["split"] == "test"].copy()
    else:
        eval_df = base.copy()

    return eval_df.reset_index(drop=True)


# ── Metrics computation ───────────────────────────────────────────────────────

def model_metrics(eval_df: pd.DataFrame, model: str, label: str) -> dict:
    """Compute MAE, RMSE, R², MBE, within-1/2%, calibration metrics."""
    pred_col = f"{model}_pred"
    std_col  = f"{model}_std"

    valid = eval_df.dropna(subset=["cycle_soh_obs", pred_col])
    if len(valid) < 3:
        return {"model": model, "subset": label, "n": len(valid)}

    y_true = valid["cycle_soh_obs"].values
    y_pred = valid[pred_col].values
    resid  = y_pred - y_true

    mae   = float(np.mean(np.abs(resid)))
    rmse  = float(np.sqrt(np.mean(resid ** 2)))
    ss_r  = np.sum(resid ** 2)
    ss_t  = np.sum((y_true - y_true.mean()) ** 2)
    r2    = 1.0 - ss_r / ss_t if ss_t > 0 else np.nan
    mbe   = float(np.mean(resid))
    w1    = float(np.mean(np.abs(resid) <= 1.0))
    w2    = float(np.mean(np.abs(resid) <= 2.0))

    # Calibration at multiple levels
    picp = {}
    mpiw = {}
    if std_col in valid.columns:
        for ci, z in [(50, 0.674), (68, 1.0), (80, 1.282), (90, 1.645), (95, 1.96)]:
            lo = y_pred - z * valid[std_col].values
            hi = y_pred + z * valid[std_col].values
            picp[ci] = float(np.mean((y_true >= lo) & (y_true <= hi)))
            mpiw[ci] = float(np.mean(hi - lo))
    elif model == "PF":
        # Use PF quantiles directly
        pf05 = valid.get("pf_soh_p05", valid[pred_col] - 2).values
        pf95 = valid.get("pf_soh_p95", valid[pred_col] + 2).values
        picp[90] = float(np.mean((y_true >= pf05) & (y_true <= pf95)))
        mpiw[90] = float(np.mean(pf95 - pf05))

    result = {
        "model": model, "subset": label, "n": len(valid),
        "mae": round(mae, 4), "rmse": round(rmse, 4),
        "r2": round(r2, 4), "mbe": round(mbe, 4),
        "within_1pct": round(w1, 4), "within_2pct": round(w2, 4),
    }
    for ci in picp:
        result[f"picp{ci}"] = round(picp[ci], 4)
        result[f"mpiw{ci}"] = round(mpiw.get(ci, np.nan), 4)
    return result


def per_vehicle_metrics(dfs: dict) -> pd.DataFrame:
    """MAE per vehicle per model (test set)."""
    rows = []
    for name, df in dfs.items():
        pred_col = f"{name}_pred"
        if pred_col not in df.columns:
            continue
        test_df = df[df["split"] == "test"] if "split" in df.columns else df
        for reg, grp in test_df.groupby("registration_number"):
            valid = grp.dropna(subset=["cycle_soh_obs", pred_col])
            mae   = float(np.mean(np.abs(valid[pred_col] - valid["cycle_soh_obs"]))) if len(valid) > 2 else np.nan
            rows.append({"registration_number": reg, "model": name, "mae": mae, "n": len(valid)})
    return pd.DataFrame(rows)


# ── Model agreement ───────────────────────────────────────────────────────────

def compute_model_agreement(eval_df: pd.DataFrame, models: list) -> pd.DataFrame:
    """
    For each session, compute spread across model predictions.
    Flag sessions where spread > 2.0%.
    """
    pred_cols = [f"{m}_pred" for m in models if f"{m}_pred" in eval_df.columns]
    if len(pred_cols) < 2:
        return pd.DataFrame()

    agree_df = eval_df.copy()
    agree_df["pred_spread"] = agree_df[pred_cols].std(axis=1, skipna=True)
    agree_df["pred_mean"]   = agree_df[pred_cols].mean(axis=1, skipna=True)
    agree_df["high_disagree"] = agree_df["pred_spread"] > 2.0

    return agree_df[["session_id", "registration_number", "cum_efc",
                      "cycle_soh_obs", "pred_mean", "pred_spread", "high_disagree"]
                    + pred_cols]


def residual_correlation_matrix(eval_df: pd.DataFrame, models: list) -> pd.DataFrame:
    """Pearson r between per-session residuals of each model pair."""
    resid_dict = {}
    for m in models:
        pred_col = f"{m}_pred"
        if pred_col in eval_df.columns and "cycle_soh_obs" in eval_df.columns:
            r = eval_df[pred_col] - eval_df["cycle_soh_obs"]
            resid_dict[m] = r.values

    if len(resid_dict) < 2:
        return pd.DataFrame()

    m_list = list(resid_dict.keys())
    n      = len(m_list)
    mat    = np.full((n, n), np.nan)
    for i in range(n):
        for j in range(n):
            ri = resid_dict[m_list[i]]
            rj = resid_dict[m_list[j]]
            mask = ~(np.isnan(ri) | np.isnan(rj))
            if mask.sum() > 10:
                r, _ = pearsonr(ri[mask], rj[mask])
                mat[i, j] = r

    return pd.DataFrame(mat, index=m_list, columns=m_list)


# ── Fleet forecast consensus ──────────────────────────────────────────────────

def build_fleet_consensus(today_str: str) -> pd.DataFrame:
    """
    Merge all fleet_forecast_*.csv into one consolidated table.
    One row per vehicle with SoH from every model + consensus.
    """
    fleet_dfs = {}
    for model, path in FLEET_CSVS.items():
        if not os.path.exists(path):
            continue
        df = pd.read_csv(path)
        # Standardise column names
        if "current_soh" in df.columns:
            df = df.rename(columns={"current_soh": f"{model}_current_soh"})
        if "soh_90d" in df.columns:
            df = df.rename(columns={"soh_90d": f"{model}_soh_90d"})
        elif "soh_90d_p50" in df.columns:
            df = df.rename(columns={"soh_90d_p50": f"{model}_soh_90d"})
        fleet_dfs[model] = df[["registration_number"] +
                               [c for c in df.columns if c != "registration_number"]]

    if not fleet_dfs:
        return pd.DataFrame()

    base_name = next(iter(fleet_dfs))
    base_df   = fleet_dfs[base_name][["registration_number"]].copy()

    for model, df in fleet_dfs.items():
        keep = ["registration_number"]
        for c in [f"{model}_current_soh", f"{model}_soh_90d", "risk_flag", "rul_days_p50"]:
            if c in df.columns:
                keep.append(c)
                if c in ["risk_flag", "rul_days_p50"] and model != base_name:
                    df = df.rename(columns={c: f"{model}_{c}"})
                    keep[-1] = f"{model}_{c}"
        base_df = base_df.merge(df[[c for c in keep if c in df.columns]],
                                on="registration_number", how="outer")

    # Consensus SoH 90d
    soh90_cols = [c for c in base_df.columns if c.endswith("_soh_90d")]
    if len(soh90_cols) > 0:
        base_df["consensus_soh_90d"] = base_df[soh90_cols].mean(axis=1, skipna=True).round(2)
        base_df["forecast_disagree"] = base_df[soh90_cols].std(axis=1, skipna=True).round(3)

    # Current SoH consensus
    cur_cols = [c for c in base_df.columns if c.endswith("_current_soh")]
    if cur_cols:
        base_df["consensus_current_soh"] = base_df[cur_cols].mean(axis=1, skipna=True).round(2)

    base_df["report_date"] = today_str
    return base_df


# ─────────────────────────────────────────────────────────────────────────────
# PLOTS
# ─────────────────────────────────────────────────────────────────────────────

def _risk_colour(rul_days) -> str:
    if rul_days is None or not np.isfinite(float(rul_days)):
        return FLEET_GREY
    if float(rul_days) < 180:   return RISK_RED
    if float(rul_days) < 365:   return RISK_AMBER
    return RISK_GREEN


def plot_model_metrics_bar(metrics_list: list, out_path: str):
    """Grouped bar chart: MAE, RMSE, Within-2% for each model."""
    models   = [m["model"] for m in metrics_list]
    mae_vals = [m.get("mae", np.nan) for m in metrics_list]
    rmse_vals = [m.get("rmse", np.nan) for m in metrics_list]
    w2_vals  = [m.get("within_2pct", np.nan) for m in metrics_list]

    x    = np.arange(len(models))
    width = 0.25

    fig, axes = plt.subplots(1, 3, figsize=(14, 5))

    for ax, vals, ylabel, title, cmap_base in zip(
        axes,
        [mae_vals, rmse_vals, w2_vals],
        ["MAE (%)", "RMSE (%)", "Within ±2% (fraction)"],
        ["Mean Absolute Error", "Root Mean Squared Error", "Within ±2% Accuracy"],
        [0.3, 0.4, 0.5]
    ):
        bar_cols = [MODEL_COLOURS.get(m, FLEET_BLUE) for m in models]
        bars     = ax.bar(x, vals, color=bar_cols, edgecolor="white", width=0.6)
        for bar, val in zip(bars, vals):
            if np.isfinite(val):
                ax.text(bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + 0.001,
                        f"{val:.3f}", ha="center", va="bottom", fontsize=8)
        ax.set_xticks(x)
        ax.set_xticklabels(models, fontsize=9)
        ax.set_ylabel(ylabel, fontsize=9)
        ax.set_title(title, fontsize=10)
        if "Accuracy" in title:
            ax.set_ylim(0, 1.05)
            ax.axhline(0.9, color="green", ls="--", lw=1, label="90% target")
            ax.legend(fontsize=8)

    plt.suptitle("Model Comparison — Quality-Gated Test Set",
                 fontsize=12, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out_path}")


def plot_calibration_comparison(metrics_list: list, out_path: str):
    """Calibration curves: PICP vs nominal coverage for each probabilistic model."""
    fig, ax = plt.subplots(figsize=(8, 6))

    nominal = [50, 68, 80, 90, 95]

    for m_dict in metrics_list:
        model  = m_dict["model"]
        picps  = [m_dict.get(f"picp{ci}", np.nan) for ci in nominal]
        if all(np.isnan(p) for p in picps):
            continue
        col = MODEL_COLOURS.get(model, FLEET_BLUE)
        ax.plot([c / 100 for c in nominal], picps,
                "o-", color=col, lw=2, ms=7, label=model)

    ax.plot([0, 1], [0, 1], "k--", lw=1.5, label="Perfect calibration")
    ax.fill_between([0, 1], [0, 1], [1, 1], alpha=0.05, color="green", label="Overconfident region")
    ax.fill_between([0, 1], [0, 0], [0, 1], alpha=0.05, color="red",   label="Underconfident region")

    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_xlabel("Nominal Coverage", fontsize=11)
    ax.set_ylabel("Empirical Coverage", fontsize=11)
    ax.set_title("Calibration Comparison — Empirical vs Nominal Coverage\n"
                 "On/above diagonal = well calibrated", fontsize=11)
    ax.legend(fontsize=9)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out_path}")


def plot_soh_fleet_ribbons(eval_df: pd.DataFrame, models: list, out_path: str):
    """Per-vehicle SoH timeline ribbons for each model."""
    vehicles = eval_df["registration_number"].unique()[:12]  # cap at 12
    n_v  = len(vehicles)
    fig, axes = plt.subplots(n_v, 1, figsize=(12, 2.5 * n_v), sharex=False)
    if n_v == 1:
        axes = [axes]

    for ax, reg in zip(axes, vehicles):
        sub = eval_df[eval_df["registration_number"] == reg].sort_values("cum_efc")

        # Observed SoH
        obs = sub.dropna(subset=["cycle_soh_obs"])
        ax.scatter(obs["cum_efc"], obs["cycle_soh_obs"], s=8, c="black",
                   zorder=10, label="Observed" if ax == axes[0] else "")

        for m in models:
            pred_col = f"{m}_pred"
            std_col  = f"{m}_std"
            if pred_col not in sub.columns:
                continue
            sub_m = sub.dropna(subset=[pred_col])
            col   = MODEL_COLOURS.get(m, FLEET_BLUE)
            ax.plot(sub_m["cum_efc"], sub_m[pred_col], color=col, lw=1.2,
                    alpha=0.8, label=m if ax == axes[0] else "")
            if std_col in sub_m.columns:
                ax.fill_between(sub_m["cum_efc"],
                                sub_m[pred_col] - 1.96 * sub_m[std_col],
                                sub_m[pred_col] + 1.96 * sub_m[std_col],
                                color=col, alpha=0.1)

        ax.axhline(EOL_SOH, color=RISK_RED, ls="--", lw=1)
        ax.set_ylim(85, 102)
        ax.set_ylabel(reg, fontsize=7, rotation=0, labelpad=60, va="center")
        ax.tick_params(labelsize=7)

    axes[0].legend(fontsize=7, ncol=len(models) + 1, loc="upper right")
    axes[-1].set_xlabel("Charge Cycles (EFC)", fontsize=10)
    fig.suptitle("Model SoH Trajectories — Fleet Ribbons\n(bands = ±2σ where available)",
                 fontsize=11, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out_path}")


def plot_model_agreement_heatmap(eval_df: pd.DataFrame, perveh_df: pd.DataFrame,
                                 corr_mat: pd.DataFrame, out_path: str):
    """2-panel: per-vehicle × model MAE heatmap + pairwise residual correlation."""
    fig, axes = plt.subplots(1, 2, figsize=(14, max(5, len(perveh_df["registration_number"].unique()) * 0.25 + 3)))

    # Panel 1: MAE heatmap
    ax = axes[0]
    if len(perveh_df) > 0:
        pivot = perveh_df.pivot_table(index="registration_number",
                                      columns="model", values="mae", aggfunc="mean")
        im = ax.imshow(pivot.values, aspect="auto", cmap="YlOrRd")
        ax.set_xticks(range(len(pivot.columns)))
        ax.set_xticklabels(pivot.columns, fontsize=9)
        ax.set_yticks(range(len(pivot.index)))
        ax.set_yticklabels(pivot.index, fontsize=6)
        ax.set_title("Per-Vehicle MAE by Model\n(darker = higher error)", fontsize=10)
        plt.colorbar(im, ax=ax, label="MAE (%)")

    # Panel 2: Pairwise residual correlation
    ax = axes[1]
    if len(corr_mat) > 0:
        im2 = ax.imshow(corr_mat.values, cmap="coolwarm", vmin=-1, vmax=1, aspect="auto")
        n   = len(corr_mat)
        ax.set_xticks(range(n)); ax.set_yticks(range(n))
        ax.set_xticklabels(corr_mat.columns, fontsize=9, rotation=45)
        ax.set_yticklabels(corr_mat.index, fontsize=9)
        for i in range(n):
            for j in range(n):
                v = corr_mat.values[i, j]
                if np.isfinite(v):
                    ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=8)
        ax.set_title("Pairwise Residual Correlation\n(off-diagonal shows model diversity)", fontsize=10)
        plt.colorbar(im2, ax=ax, label="Pearson r")

    plt.suptitle("Model Agreement Analysis", fontsize=11, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out_path}")


def plot_rul_comparison(out_path: str):
    """Horizontal bars per vehicle for EKF/ECM/PF RUL with CI."""
    rul_sources = {
        "EKF": (EKF_CSV,  "ekf_rul_days", None, None),
        "ECM": (ECM_CSV,  "ecm_rul_days", "ecm_rul_days_lo", "ecm_rul_days_hi"),
        "PF":  (PF_CSV,   "pf_rul_p50",   "pf_rul_p05",      "pf_rul_p95"),
    }

    combined = {}
    for model, (path, p50_col, lo_col, hi_col) in rul_sources.items():
        if not os.path.exists(path):
            continue
        df    = pd.read_csv(path)
        if p50_col not in df.columns:
            continue
        latest = df.sort_values("cum_efc").groupby("registration_number").last().reset_index()
        combined[model] = latest[["registration_number", p50_col,
                                   lo_col or p50_col, hi_col or p50_col]].rename(
            columns={p50_col: "rul_p50", lo_col or p50_col: "rul_lo", hi_col or p50_col: "rul_hi"}
        )

    if not combined:
        print(f"  [SKIP] {out_path} — no RUL data available")
        return

    # Merge on registration_number
    all_regs = sorted(set.union(*[set(df["registration_number"]) for df in combined.values()]))
    fig, axes = plt.subplots(1, 2, figsize=(14, max(5, len(all_regs) * 0.3 + 2)))

    ax = axes[0]
    y_pos  = np.arange(len(all_regs))
    offsets = {"EKF": -0.25, "ECM": 0.0, "PF": 0.25}

    for model, df_m in combined.items():
        col = MODEL_COLOURS.get(model, FLEET_BLUE)
        off = offsets.get(model, 0.0)
        for yi, reg in enumerate(all_regs):
            row = df_m[df_m["registration_number"] == reg]
            if len(row) == 0:
                continue
            p50 = float(row["rul_p50"].values[0])
            lo  = float(row["rul_lo"].values[0])
            hi  = float(row["rul_hi"].values[0])
            if np.isfinite(p50):
                ax.plot([lo, hi], [yi + off, yi + off], color=col, lw=2, alpha=0.5)
                ax.scatter(p50, yi + off, color=col, s=40, zorder=5)

    ax.axvline(365, color="black", ls=":", lw=1.5, label="1 year")
    ax.axvline(180, color=RISK_AMBER, ls=":", lw=1, label="6 months")
    ax.set_yticks(y_pos)
    ax.set_yticklabels(all_regs, fontsize=7)
    ax.set_xlabel("Remaining Useful Life (days)")
    ax.set_title("RUL per Vehicle — EKF / ECM / PF\n(bars = CI range)")
    for model, col in MODEL_COLOURS.items():
        if model in combined:
            ax.plot([], [], color=col, lw=3, label=model)
    ax.legend(fontsize=8, loc="lower right")

    # Panel 2: scatter EKF vs PF RUL p50
    ax = axes[1]
    if "EKF" in combined and "PF" in combined:
        merged = combined["EKF"].merge(combined["PF"], on="registration_number", suffixes=("_ekf", "_pf"))
        merged = merged.dropna(subset=["rul_p50_ekf", "rul_p50_pf"])
        ax.scatter(merged["rul_p50_ekf"], merged["rul_p50_pf"], s=50, color=FLEET_BLUE, alpha=0.7)
        lim = max(merged[["rul_p50_ekf", "rul_p50_pf"]].max().max(), 100)
        ax.plot([0, lim], [0, lim], "r--", lw=1.5, label="45° line")
        ax.set_xlabel("EKF RUL p50 (days)")
        ax.set_ylabel("PF RUL p50 (days)")
        ax.set_title("EKF vs PF RUL Estimates\n(above 45° = PF gives more time)")
        ax.legend(fontsize=8)

    plt.suptitle("Remaining Useful Life Comparison — EKF / ECM / PF",
                 fontsize=11, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out_path}")


def plot_uncertainty_comparison(eval_df: pd.DataFrame, models: list, out_path: str):
    """Mean posterior std vs cum_efc bins — one line per probabilistic model."""
    prob_models = [m for m in models if f"{m}_std" in eval_df.columns]
    if not prob_models:
        print(f"  [SKIP] {out_path} — no std columns available")
        return

    fig, ax = plt.subplots(figsize=(9, 5))

    bins      = np.linspace(eval_df["cum_efc"].min(), eval_df["cum_efc"].max(), 20)
    eval_copy = eval_df.copy()
    eval_copy["efc_bin"] = pd.cut(eval_copy["cum_efc"], bins=bins, labels=False)

    for model in prob_models:
        std_col = f"{model}_std"
        grp     = eval_copy.groupby("efc_bin")[std_col].mean()
        bin_mids = [(bins[i] + bins[i + 1]) / 2 for i in range(len(bins) - 1)]
        col = MODEL_COLOURS.get(model, FLEET_BLUE)
        ax.plot(bin_mids[:len(grp)], grp.values, "o-", color=col, lw=2, ms=5, label=model)

    ax.set_xlabel("Charge Cycles (EFC)")
    ax.set_ylabel("Mean Posterior Std (%)")
    ax.set_title("SoH Prediction Uncertainty vs Charge Cycles\n(lower = more confident)")
    ax.legend(fontsize=9)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out_path}")


def plot_fleet_scorecard(scorecard_df: pd.DataFrame, today_str: str, out_path: str):
    """
    STAKEHOLDER — Fleet Health Scorecard.
    One row per vehicle. Columns: Current SoH, 90d forecast, RUL, Risk.
    Clean table design, no axes.
    """
    df = scorecard_df.copy()

    # Pick best available SoH and RUL columns
    cur_col  = "consensus_current_soh" if "consensus_current_soh" in df.columns else \
               next((c for c in df.columns if "current_soh" in c), None)
    soh90_col = "consensus_soh_90d" if "consensus_soh_90d" in df.columns else \
                next((c for c in df.columns if "soh_90d" in c), None)
    rul_col   = next((c for c in df.columns if "rul_days_p50" in c), None)

    # Sort by RUL ascending (most critical first)
    if rul_col and rul_col in df.columns:
        df = df.sort_values(rul_col, ascending=True, na_position="last")
    elif cur_col:
        df = df.sort_values(cur_col, ascending=True)

    n = len(df)
    fig, ax = plt.subplots(figsize=(14, max(5, n * 0.45 + 2)))
    ax.set_xlim(0, 10)
    ax.set_ylim(-0.5, n - 0.5)
    ax.axis("off")

    # Column positions and headers
    cols     = ["Vehicle", "Current\nSoH (%)", "90-Day\nForecast (%)",
                "RUL\n(days)", "Risk", "Forecast\nConfidence"]
    x_pos    = [0.3, 2.2, 3.8, 5.4, 6.8, 8.3]
    col_widths = [1.8, 1.4, 1.4, 1.4, 1.2, 1.6]

    # Header
    for xi, label in zip(x_pos, cols):
        ax.text(xi, n - 0.1, label, fontsize=9, fontweight="bold",
                ha="center", va="bottom")

    ax.axhline(n - 0.3, color="black", lw=1.5, xmin=0, xmax=1)

    for yi, (_, row) in enumerate(df.iterrows()):
        row_y = n - 1 - yi

        # Background stripe
        bg_col = "#f8f9fa" if yi % 2 == 0 else "white"
        ax.barh(row_y, 10, left=0, height=0.85, color=bg_col, zorder=0)

        # Vehicle name
        ax.text(x_pos[0], row_y, str(row.get("registration_number", ""))[:12],
                ha="center", va="center", fontsize=8)

        # Current SoH — colour-coded circle
        cur_soh = row.get(cur_col) if cur_col else np.nan
        if cur_soh is not None and np.isfinite(float(cur_soh or np.nan)):
            soh_pct = float(cur_soh)
            soh_col = RISK_GREEN if soh_pct > 95 else (RISK_AMBER if soh_pct > 90 else RISK_RED)
            circ = plt.Circle((x_pos[1], row_y), 0.32, color=soh_col, zorder=2)
            ax.add_patch(circ)
            ax.text(x_pos[1], row_y, f"{soh_pct:.1f}",
                    ha="center", va="center", fontsize=8, color="white", fontweight="bold")
        else:
            ax.text(x_pos[1], row_y, "—", ha="center", va="center", fontsize=8)

        # 90d forecast
        soh90 = row.get(soh90_col) if soh90_col else np.nan
        if soh90 is not None and np.isfinite(float(soh90 or np.nan)):
            ax.text(x_pos[2], row_y, f"{float(soh90):.1f}",
                    ha="center", va="center", fontsize=8)
        else:
            ax.text(x_pos[2], row_y, "—", ha="center", va="center", fontsize=8)

        # RUL
        rul = row.get(rul_col) if rul_col else np.nan
        if rul is not None and np.isfinite(float(rul or np.nan)):
            rul_val = float(rul)
            rul_col_colour = RISK_RED if rul_val < 180 else (RISK_AMBER if rul_val < 365 else RISK_GREEN)
            ax.text(x_pos[3], row_y, f"{rul_val:.0f}",
                    ha="center", va="center", fontsize=8, color=rul_col_colour, fontweight="bold")
        else:
            ax.text(x_pos[3], row_y, "—", ha="center", va="center", fontsize=8)

        # Risk flag
        risk = str(row.get("risk_flag", row.get("ECM_risk_flag", "green"))).lower()
        risk_col  = RISK_RED if "red" in risk else (RISK_AMBER if "amber" in risk else RISK_GREEN)
        risk_label = "HIGH" if "red" in risk else ("MED" if "amber" in risk else "OK")
        rect = plt.Rectangle((x_pos[4] - 0.4, row_y - 0.28), 0.8, 0.56,
                              color=risk_col, zorder=2, alpha=0.85)
        ax.add_patch(rect)
        ax.text(x_pos[4], row_y, risk_label,
                ha="center", va="center", fontsize=7, color="white", fontweight="bold")

        # Forecast confidence (disagreement)
        disagree = row.get("forecast_disagree", np.nan)
        if disagree is not None and np.isfinite(float(disagree or np.nan)):
            conf_str = "High" if float(disagree) < 0.5 else ("Medium" if float(disagree) < 1.5 else "Low")
            conf_col = RISK_GREEN if conf_str == "High" else (RISK_AMBER if conf_str == "Medium" else RISK_RED)
            ax.text(x_pos[5], row_y, conf_str,
                    ha="center", va="center", fontsize=8, color=conf_col)
        else:
            ax.text(x_pos[5], row_y, "—", ha="center", va="center", fontsize=8)

    ax.axhline(-0.25, color="black", lw=1, xmin=0, xmax=1)

    # Legend
    legend_elements = [
        Patch(facecolor=RISK_GREEN, label="SoH > 95% / RUL > 1 year"),
        Patch(facecolor=RISK_AMBER, label="SoH 90-95% / RUL 6-12 months"),
        Patch(facecolor=RISK_RED,   label="SoH < 90% / RUL < 6 months"),
    ]
    ax.legend(handles=legend_elements, loc="lower right",
              bbox_to_anchor=(1.0, -0.05), ncol=3, fontsize=8)

    ax.set_title(f"Fleet Battery Health Scorecard — {today_str}\n"
                 f"Sorted by urgency (most critical first). "
                 f"90-day forecast = consensus across all models.",
                 fontsize=11, fontweight="bold", pad=12)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out_path}")


def plot_forecast_consensus(scorecard_df: pd.DataFrame, out_path: str):
    """
    STAKEHOLDER — Per vehicle: grouped bars of 90d SoH from each model.
    Vehicles on Y, color by model, sorted by risk.
    """
    soh90_cols = [(m, c) for m in MODEL_COLOURS for c in scorecard_df.columns
                  if c == f"{m}_soh_90d"]

    if not soh90_cols:
        print(f"  [SKIP] {out_path} — no 90d forecast columns found")
        return

    # Sort by mean 90d SoH ascending
    vals_list = [scorecard_df[c].values for _, c in soh90_cols]
    if vals_list:
        scorecard_df = scorecard_df.copy()
        scorecard_df["_mean90"] = np.nanmean(np.column_stack(vals_list), axis=1)
        scorecard_df = scorecard_df.sort_values("_mean90", ascending=True)

    vehicles = scorecard_df["registration_number"].values
    n_v      = len(vehicles)
    n_m      = len(soh90_cols)
    y_pos    = np.arange(n_v)
    bar_h    = 0.8 / max(n_m, 1)

    fig, ax = plt.subplots(figsize=(12, max(5, n_v * 0.4 + 2)))

    for mi, (model, col) in enumerate(soh90_cols):
        offset = (mi - (n_m - 1) / 2) * bar_h
        vals   = scorecard_df[col].values
        colour = MODEL_COLOURS.get(model, FLEET_BLUE)
        ax.barh(y_pos + offset, vals, height=bar_h * 0.85,
                color=colour, alpha=0.8, label=model)

    ax.axvline(EOL_SOH, color=RISK_RED, ls="--", lw=1.5, label="EOL 80%")
    ax.axvline(95, color=RISK_GREEN, ls=":", lw=1, label="95% healthy threshold")
    ax.set_yticks(y_pos)
    ax.set_yticklabels(vehicles, fontsize=8)
    ax.set_xlabel("Projected Battery Health (%) — 90 Days from Now", fontsize=10)
    ax.set_title("90-Day SoH Forecast — All Models per Vehicle\n"
                 "Narrow bar spread = high model confidence  ·  Wide spread = uncertain",
                 fontsize=11, fontweight="bold")
    ax.legend(fontsize=9, loc="lower right")
    ax.set_xlim(80, 105)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out_path}")


# ── Summary report ────────────────────────────────────────────────────────────

def write_summary_report(metrics_df: pd.DataFrame,
                         agree_df: pd.DataFrame,
                         scorecard_df: pd.DataFrame,
                         ekf_vs_pf: dict,
                         today_str: str,
                         out_path: str):
    lines = [
        "=" * 72,
        "FLEET BATTERY SOH — MODEL EVALUATION SUMMARY REPORT",
        f"Generated: {today_str}",
        "=" * 72,
        "",
    ]

    # Fleet context
    lines.append("FLEET CONTEXT")
    lines.append("-" * 40)
    if "consensus_current_soh" in scorecard_df.columns:
        cur = scorecard_df["consensus_current_soh"].dropna()
        lines.append(f"  Fleet SoH (consensus, current): {cur.mean():.2f}% ± {cur.std():.2f}%")
        lines.append(f"  Range: {cur.min():.2f}% – {cur.max():.2f}%")
    for flag, colour in [("red", "HIGH"), ("amber", "MEDIUM"), ("green", "LOW")]:
        flag_col = next((c for c in scorecard_df.columns if "risk_flag" in c.lower()), None)
        if flag_col:
            n_flag = (scorecard_df[flag_col].str.lower() == flag).sum()
            lines.append(f"  Risk {colour}: {n_flag} vehicle(s)")
    lines.append("")

    # Metrics table
    lines.append("MODEL METRICS (quality-gated test set)")
    lines.append("-" * 72)
    if len(metrics_df) > 0:
        test_qg = metrics_df[metrics_df["subset"] == "test_quality_gated"] if "subset" in metrics_df.columns else metrics_df
        for _, row in test_qg.iterrows():
            model = row.get("model", "?")
            lines.append(f"  {model:6s}  MAE={row.get('mae','N/A'):.4f}%  "
                         f"RMSE={row.get('rmse','N/A'):.4f}%  "
                         f"R²={row.get('r2','N/A'):.3f}  "
                         f"Within-2%={row.get('within_2pct','N/A'):.1%}  "
                         f"PICP90={row.get('picp90','N/A')}")
    lines.append("")

    # Best and worst
    lines.append("MODEL RANKING (by test MAE, lower is better)")
    lines.append("-" * 40)
    if len(metrics_df) > 0:
        test_qg = metrics_df[metrics_df["subset"] == "test_quality_gated"] if "subset" in metrics_df.columns else metrics_df
        ranked  = test_qg.dropna(subset=["mae"]).sort_values("mae")
        for rank, (_, row) in enumerate(ranked.iterrows(), 1):
            lines.append(f"  {rank}. {row.get('model','?'):6s}  MAE = {row.get('mae',np.nan):.4f}%")
    lines.append("")

    # Model agreement
    if len(agree_df) > 0:
        lines.append("MODEL AGREEMENT")
        lines.append("-" * 40)
        high_dis = agree_df[agree_df["high_disagree"]] if "high_disagree" in agree_df.columns else pd.DataFrame()
        lines.append(f"  Sessions with spread > 2%: {len(high_dis):,} "
                     f"({100*len(high_dis)/max(len(agree_df),1):.1f}%)")
        if len(high_dis) > 0:
            lines.append("  Top-5 disagreement sessions:")
            top5 = high_dis.sort_values("pred_spread", ascending=False).head(5)
            for _, r in top5.iterrows():
                lines.append(f"    session_id={r.get('session_id','?')}  "
                             f"vehicle={r.get('registration_number','?')}  "
                             f"spread={r.get('pred_spread','?'):.3f}%")
        lines.append("")

    # PF vs EKF
    if ekf_vs_pf:
        lines.append("PF vs EKF (non-Gaussianity justification)")
        lines.append("-" * 40)
        for k, v in ekf_vs_pf.items():
            lines.append(f"  {k}: {v}")
        lines.append("")

    lines.append("=" * 72)
    lines.append("END OF REPORT")
    lines.append("")

    with open(out_path, "w") as f:
        f.write("\n".join(lines))
    print(f"  Saved: {out_path}")


# ── Main ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    today_str = datetime.now().strftime("%Y-%m-%d")

    print("=" * 70)
    print("model_eval_comparison.py — Cross-Model Evaluation")
    print(f"Date: {today_str}")
    print("=" * 70)

    # ── Load all model predictions ─────────────────────────────────────────
    print("\nLoading model predictions ...")
    dfs = load_model_predictions()
    models_loaded = list(dfs.keys())
    print(f"  Models available: {models_loaded}")

    if not dfs:
        print("[ERROR] No model CSVs found. Run all model scripts first.")
        sys.exit(1)

    # ── Build common evaluation set ────────────────────────────────────────
    print("\nBuilding common evaluation set (quality-gated test) ...")
    eval_df = build_common_eval_set(dfs)
    print(f"  Eval set: {len(eval_df):,} sessions, {eval_df['registration_number'].nunique()} vehicles")

    # ── Compute metrics ────────────────────────────────────────────────────
    print("\nComputing metrics ...")
    metrics_rows = []
    for model in models_loaded:
        m_all  = model_metrics(eval_df, model, "test_quality_gated")
        metrics_rows.append(m_all)

        # Also compute on all-test and train for each model
        if model in dfs:
            df_m = dfs[model]
            if "split" in df_m.columns:
                test_all = df_m[df_m["split"] == "test"]
                metrics_rows.append(model_metrics(test_all, model, "test_all"))

    metrics_df = pd.DataFrame(metrics_rows)
    metrics_df.to_csv(OUT_METRICS, index=False)
    print(f"\n  Metrics saved: {OUT_METRICS}")

    qg_rows = metrics_df[metrics_df["subset"] == "test_quality_gated"] if "subset" in metrics_df.columns else metrics_df
    if len(qg_rows) > 0:
        print(qg_rows[["model", "n", "mae", "rmse", "r2", "within_2pct",
                       "picp90"]].to_string(index=False))

    # ── Per-vehicle metrics ────────────────────────────────────────────────
    print("\nPer-vehicle metrics ...")
    perveh_df = per_vehicle_metrics(dfs)
    perveh_df.to_csv(OUT_PER_VEH, index=False)

    # ── Residual correlation ───────────────────────────────────────────────
    print("\nResidual correlation ...")
    corr_mat = residual_correlation_matrix(eval_df, models_loaded)
    if len(corr_mat) > 0:
        print(corr_mat.round(3).to_string())

    # ── Model agreement ────────────────────────────────────────────────────
    print("\nModel agreement ...")
    agree_df = compute_model_agreement(eval_df, models_loaded)
    if len(agree_df) > 0:
        agree_df.to_csv(OUT_AGREE, index=False)
        print(f"  Spread mean: {agree_df['pred_spread'].mean():.4f}%")
        print(f"  High disagreement (>2%): {agree_df['high_disagree'].sum():,} sessions")

    # ── PF vs EKF comparison ───────────────────────────────────────────────
    ekf_vs_pf = {}
    if "PF" in dfs and os.path.exists(EKF_CSV):
        print("\nPF vs EKF comparison ...")
        from particle_filter_soh import pf_vs_ekf
        ekf_vs_pf = pf_vs_ekf(dfs["PF"])
        for k, v in ekf_vs_pf.items():
            print(f"  {k}: {v}")

    # Young-fleet extrapolation sanity check
    print("\nYoung-fleet extrapolation sanity check (project to EFC=3000) ...")
    for model in models_loaded:
        df_m = dfs[model]
        pred_col = f"{model}_pred"
        if pred_col not in df_m.columns:
            continue
        max_pred = df_m[pred_col].min()
        if max_pred < 60.0 or max_pred > 100.5:
            print(f"  [WARN] {model}: min predicted SoH = {max_pred:.2f}% — check for extrapolation error")
        else:
            print(f"  {model}: min predicted SoH = {max_pred:.2f}% [OK]")

    # ── Fleet forecast consensus ───────────────────────────────────────────
    print("\nBuilding fleet forecast consensus ...")
    scorecard_df = build_fleet_consensus(today_str)
    if len(scorecard_df) > 0:
        scorecard_df.to_csv(OUT_SCORECARD, index=False)
        print(f"  Saved scorecard: {OUT_SCORECARD} ({len(scorecard_df)} vehicles)")
        print(scorecard_df[["registration_number"] +
                            [c for c in scorecard_df.columns
                             if "soh" in c.lower() or "rul" in c.lower() or "risk" in c.lower()
                             ][:6]].to_string(index=False))
    else:
        print("  [WARN] No fleet forecast files found — scorecard will be empty")
        scorecard_df = pd.DataFrame()

    # ── Summary report ─────────────────────────────────────────────────────
    print("\nWriting summary report ...")
    write_summary_report(metrics_df, agree_df if len(agree_df) > 0 else pd.DataFrame(),
                         scorecard_df, ekf_vs_pf, today_str, OUT_REPORT)

    # ── Plots ──────────────────────────────────────────────────────────────
    print("\nGenerating plots ...")

    if len(qg_rows) > 0:
        plot_model_metrics_bar(qg_rows.to_dict("records"),
                               os.path.join(PLOTS_DIR, "eval_model_metrics_bar.png"))

        plot_calibration_comparison(qg_rows.to_dict("records"),
                                    os.path.join(PLOTS_DIR, "eval_calibration_comparison.png"))

    if len(eval_df) > 0:
        plot_soh_fleet_ribbons(eval_df, models_loaded,
                               os.path.join(PLOTS_DIR, "eval_soh_fleet_ribbons.png"))

        plot_model_agreement_heatmap(eval_df, perveh_df, corr_mat,
                                     os.path.join(PLOTS_DIR, "eval_model_agreement_heatmap.png"))

        plot_uncertainty_comparison(eval_df, models_loaded,
                                    os.path.join(PLOTS_DIR, "eval_uncertainty_comparison.png"))

    plot_rul_comparison(os.path.join(PLOTS_DIR, "eval_rul_comparison.png"))

    if len(scorecard_df) > 0:
        plot_fleet_scorecard(scorecard_df, today_str,
                             os.path.join(PLOTS_DIR, "eval_fleet_scorecard.png"))
        plot_forecast_consensus(scorecard_df,
                                os.path.join(PLOTS_DIR, "eval_forecast_consensus.png"))

    # ── Final summary ──────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("MODEL EVALUATION COMPLETE")
    print("=" * 70)
    print(f"\n  Models compared: {models_loaded}")
    if len(qg_rows) > 0:
        best = qg_rows.dropna(subset=["mae"]).sort_values("mae")
        if len(best) > 0:
            print(f"  Best MAE (quality-gated test): {best.iloc[0]['model']} "
                  f"({best.iloc[0]['mae']:.4f}%)")
    print(f"\n  Key outputs:")
    print(f"    {OUT_METRICS}")
    print(f"    {OUT_SCORECARD}")
    print(f"    {OUT_REPORT}")
    print(f"    {os.path.join(PLOTS_DIR, 'eval_fleet_scorecard.png')}")
    print(f"    {os.path.join(PLOTS_DIR, 'eval_forecast_consensus.png')}")
    print()
