"""
rul_report.py — Reliability-annotated RUL report with LFP prior-anchored
                exponential decay model.

Reads:
  rul_estimates.csv  — linear-fit RUL estimates per vehicle
  soh_trends.csv     — per-cycle SoH trajectory (efc_cumulative, soh_smooth)

Writes:
  rul_report.csv     — all existing columns + 9 new reliability/exp-model columns

Usage:
  python rul_report.py
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from config import EOL_SOH, ARTIFACTS_DIR

# ── Input / output paths ───────────────────────────────────────────────────────
RUL_ESTIMATES = os.path.join(ARTIFACTS_DIR, "rul_estimates.csv")
SOH_TRENDS    = os.path.join(ARTIFACTS_DIR, "soh_trends.csv")
RUL_REPORT    = os.path.join(ARTIFACTS_DIR, "rul_report.csv")

# ── LFP degradation prior ──────────────────────────────────────────────────────
# Source: LFP cells typically warrant 80% SoH at 2000–4000 full cycles.
# Conservative anchor: 2500 EFC (fleet is commercial EVs, moderate stress).
LFP_PRIOR_CYCLE_LIFE = 2500          # full cycles to 80% SoH
LFP_PRIOR_SOH0       = 100.0         # assumed start SoH for prior
# Derived: k_prior = ln(100/80) / 2500
LFP_PRIOR_K          = np.log(LFP_PRIOR_SOH0 / EOL_SOH) / LFP_PRIOR_CYCLE_LIFE

# Prior weight in EFC units: acts as if we have 200 EFC of prior observations.
# Low enough to be overridden when a vehicle accumulates real data (>200 EFC).
LFP_PRIOR_WEIGHT_EFC = 200.0

# ── Calendar-day degradation prior ────────────────────────────────────────────
# Commercial LFP EV fleet expected to reach 80% SoH in ~10 years (combined
# calendar aging + cycling). Used when EFC accumulation is too sparse.
FLEET_EXPECTED_LIFE_YEARS = 10          # configurable
DAY_EOL_PRIOR = FLEET_EXPECTED_LIFE_YEARS * 365.25  # ≈ 3652 days
# k_day_prior = ln(100/80) / 3652
LFP_PRIOR_K_DAY = np.log(100.0 / EOL_SOH) / DAY_EOL_PRIOR  # ≈ 6.09e-5 /day

# Prior weight in days: acts as if we have 180 days of prior observations.
LFP_PRIOR_WEIGHT_DAYS = 180.0

# R² thresholds for reliability tiers
R2_RELIABLE   = 0.70
R2_INDICATIVE = 0.50
MIN_DATA_DAYS = 60   # below this, flag as short-span regardless of R²

# EFC at EOL under pure prior: solve 100·exp(-k_prior·EFC) = 80
EFC_EOL_PRIOR = -np.log(EOL_SOH / 100.0) / LFP_PRIOR_K   # ≈ 2500 EFC


# ── Step 1: Reliability tier ───────────────────────────────────────────────────
def reliability_tier(row) -> str:
    slope = row.get("soh_slope_%per_day", np.nan)
    r2    = row.get("soh_r2", np.nan)
    span  = row.get("data_span_days", 0) or 0
    if pd.isna(slope) or slope >= 0:
        return "no_degradation_signal"
    if pd.isna(r2) or span < MIN_DATA_DAYS:
        return "insufficient_data"
    if r2 >= R2_RELIABLE:
        return "reliable"
    if r2 >= R2_INDICATIVE:
        return "indicative"
    return "unreliable"


# ── Step 2: Exponential decay fit ─────────────────────────────────────────────
def fit_exp_lfp(efc_cum: np.ndarray, soh: np.ndarray) -> tuple:
    """Fit SoH = A * exp(-k * EFC). Returns (A, k) or (nan, nan) on failure."""
    try:
        popt, _ = curve_fit(
            lambda x, A, k: A * np.exp(-k * x),
            efc_cum, soh,
            p0=[soh[0] if len(soh) > 0 else 100.0, LFP_PRIOR_K],
            bounds=([80.0, 1e-6], [105.0, 0.5]),
            maxfev=5000,
        )
        return float(popt[0]), float(popt[1])
    except Exception:
        return np.nan, np.nan


def rul_efc_from_exp(A: float, k: float, efc_current: float,
                     eol: float = EOL_SOH) -> float:
    """EFC remaining from exponential model."""
    if np.isnan(A) or np.isnan(k) or k <= 0 or A <= eol:
        return np.inf
    efc_eol = -np.log(eol / A) / k
    return max(0.0, efc_eol - efc_current)


# ── Calendar-day exponential fit ──────────────────────────────────────────────
def fit_exp_day(days: np.ndarray, soh: np.ndarray) -> tuple:
    """Fit SoH = A * exp(-k * t_days). Returns (A, k) or (nan, nan) on failure."""
    try:
        popt, _ = curve_fit(
            lambda x, A, k: A * np.exp(-k * x),
            days, soh,
            p0=[soh[0] if len(soh) > 0 else 100.0, LFP_PRIOR_K_DAY],
            bounds=([80.0, 1e-7], [105.0, 5e-4]),
            maxfev=5000,
        )
        return float(popt[0]), float(popt[1])
    except Exception:
        return np.nan, np.nan


def blended_k_day(k_fitted: float, data_span_days: float) -> float:
    """Weighted average of fitted k_day and fleet-lifetime prior."""
    if np.isnan(k_fitted):
        return LFP_PRIOR_K_DAY
    w_data  = max(data_span_days, 0.0) if not np.isnan(data_span_days) else 0.0
    w_prior = LFP_PRIOR_WEIGHT_DAYS
    return (w_data * k_fitted + w_prior * LFP_PRIOR_K_DAY) / (w_data + w_prior)


# ── Step 3: Prior-anchored k (Bayesian blend) ─────────────────────────────────
def blended_k(k_fitted: float, efc_total: float) -> float:
    """Weighted average of fitted k and LFP prior k."""
    if np.isnan(k_fitted):
        return LFP_PRIOR_K
    w_data  = efc_total if not np.isnan(efc_total) else 0.0
    w_prior = LFP_PRIOR_WEIGHT_EFC
    return (w_data * k_fitted + w_prior * LFP_PRIOR_K) / (w_data + w_prior)


# ── Step 5: Recommended RUL ───────────────────────────────────────────────────
def recommended_rul(row) -> object:
    tier = row["fit_quality"]
    if tier in ("reliable", "indicative"):
        v = row["rul_days_exp"]
        if v is not None and not pd.isna(v) and np.isfinite(v):
            return v
        # Fallback to linear RUL
        return row.get("rul_days", None)
    elif tier in ("unreliable", "insufficient_data"):
        return row.get("rul_days_prior_only", None)
    else:  # no_degradation_signal
        return None


def rul_note(row) -> str:
    tier = row["fit_quality"]
    if tier == "reliable":
        return (
            f"R²={row.get('soh_r2', '?')} ≥ 0.70 over {row.get('data_span_days', '?'):.0f}d; "
            "prior-blended exponential model used"
        )
    elif tier == "indicative":
        return (
            f"R²={row.get('soh_r2', '?')} (0.50–0.70); "
            "use for ranking only, not absolute date"
        )
    elif tier == "unreliable":
        return (
            f"R²={row.get('soh_r2', '?')} < 0.50; "
            "too noisy — pure LFP-prior estimate used"
        )
    elif tier == "insufficient_data":
        span = row.get("data_span_days", 0) or 0
        return (
            f"Only {span:.0f} days of data (< {MIN_DATA_DAYS}d threshold); "
            "pure LFP-prior estimate used"
        )
    else:  # no_degradation_signal
        return "Slope ≥ 0 — no degradation detected in BMS SoH readings"


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print(f"LFP prior: k={LFP_PRIOR_K:.6e}  ({LFP_PRIOR_CYCLE_LIFE} EFC to 80% SoH)")

    rul_df   = pd.read_csv(RUL_ESTIMATES)
    trend_df = pd.read_csv(SOH_TRENDS)

    # Filter trends to discharge sessions only (matching soh_rul.py convention)
    if "session_type" in trend_df.columns:
        trend_df = trend_df[trend_df["session_type"] == "discharge"].copy()

    # ── Compute data_span_days from first_date / last_date ─────────────────────
    if "first_date" in rul_df.columns and "last_date" in rul_df.columns:
        rul_df["data_span_days"] = (
            pd.to_datetime(rul_df["last_date"]) - pd.to_datetime(rul_df["first_date"])
        ).dt.days.astype(float)
    else:
        rul_df["data_span_days"] = np.nan

    # ── Step 1: Reliability tier ───────────────────────────────────────────────
    rul_df["fit_quality"] = rul_df.apply(reliability_tier, axis=1)

    # ── Steps 2 & 3: Exp fit + blended k, per vehicle ─────────────────────────
    exp_rows = []
    for reg, grp in trend_df.groupby("registration_number"):
        grp = grp.sort_values("efc_cumulative")
        efc_arr = grp["efc_cumulative"].values
        soh_arr = grp["soh_smooth"].values

        # Drop NaN
        mask = ~(np.isnan(efc_arr) | np.isnan(soh_arr))
        efc_arr, soh_arr = efc_arr[mask], soh_arr[mask]

        A, k_fit = np.nan, np.nan
        if len(efc_arr) >= 3:
            A, k_fit = fit_exp_lfp(efc_arr, soh_arr)

        # Look up efc_total from rul_df for blending
        rul_row = rul_df[rul_df["registration_number"] == reg]
        efc_tot = float(rul_row["efc_total"].values[0]) if len(rul_row) else np.nan

        k_blend = blended_k(k_fit, efc_tot if not pd.isna(efc_tot) else 0.0)

        # EFC remaining using blended k (use A from fit, or 100 if fit failed)
        A_use   = A if not np.isnan(A) else 100.0
        efc_rem = rul_efc_from_exp(A_use, k_blend, efc_tot if not np.isnan(efc_tot) else 0.0)

        # Day-axis fit
        day_arr = grp.sort_values("date_days")["date_days"].values if "date_days" in grp.columns else np.array([])
        soh_day = grp.sort_values("date_days")["soh_smooth"].values if "date_days" in grp.columns else np.array([])
        mask_d  = ~(np.isnan(day_arr) | np.isnan(soh_day)) if len(day_arr) > 0 else np.array([], dtype=bool)
        day_arr, soh_day = day_arr[mask_d], soh_day[mask_d]

        A_day, k_day_fit = np.nan, np.nan
        if len(day_arr) >= 3:
            A_day, k_day_fit = fit_exp_day(day_arr, soh_day)

        span = float(rul_row["data_span_days"].values[0]) \
               if len(rul_row) and not pd.isna(rul_row["data_span_days"].values[0]) else np.nan
        k_day_blend = blended_k_day(k_day_fit, span if not np.isnan(span) else 0.0)

        A_day_use = A_day if not np.isnan(A_day) else 100.0
        if k_day_blend > 0 and A_day_use > EOL_SOH:
            t_eol = -np.log(EOL_SOH / A_day_use) / k_day_blend
            rul_day_exp = max(0.0, t_eol - (span if not np.isnan(span) else 0.0))
        else:
            rul_day_exp = np.inf

        exp_rows.append({
            "registration_number": reg,
            "exp_A":        round(A, 4) if not np.isnan(A) else np.nan,
            "exp_k_fitted": round(k_fit, 8) if not np.isnan(k_fit) else np.nan,
            "exp_k_blended": round(k_blend, 8),
            "_rul_efc_exp": efc_rem,   # intermediate; converted to days below
            "exp_A_day":         round(A_day, 4) if not np.isnan(A_day) else np.nan,
            "exp_k_day_fitted":  round(k_day_fit, 10) if not np.isnan(k_day_fit) else np.nan,
            "exp_k_day_blended": round(k_day_blend, 10),
            "_rul_days_exp_day": rul_day_exp,
            "exp_fn_efc": (f"SoH = {round(A, 2):.2f} * exp(-{round(k_blend, 8):.3e} * EFC)"
                           if not (np.isnan(A) or np.isnan(k_blend)) else "fit_failed"),
            "exp_fn_day": (f"SoH = {round(A_day_use, 2):.2f} * exp(-{k_day_blend:.3e} * days)"
                           if k_day_blend > 0 else "prior_only"),
        })

    exp_df = pd.DataFrame(exp_rows)
    rul_df = rul_df.merge(exp_df, on="registration_number", how="left")

    # ── Convert rul_efc_exp → rul_days_exp ────────────────────────────────────
    def efc_to_days(row):
        efc_rem = row.get("_rul_efc_exp", np.nan)
        epd     = row.get("avg_efc_per_day", np.nan)
        if pd.isna(efc_rem) or pd.isna(epd) or epd <= 0:
            return np.nan
        if not np.isfinite(efc_rem):
            return np.inf
        return efc_rem / epd

    rul_df["rul_efc_exp"] = rul_df["_rul_efc_exp"].apply(
        lambda v: round(v, 1) if (v is not None and not pd.isna(v) and np.isfinite(v)) else (None if pd.isna(v) else None)
    )
    rul_df["rul_days_exp"] = rul_df.apply(efc_to_days, axis=1)
    rul_df["rul_days_exp"] = rul_df["rul_days_exp"].apply(
        lambda v: round(v, 0) if (not pd.isna(v) and np.isfinite(v)) else None
    )

    # ── Prior-only estimate for unreliable / insufficient vehicles ─────────────
    def prior_only_days(row):
        efc_tot = row.get("efc_total", np.nan)
        epd     = row.get("avg_efc_per_day", np.nan)
        if pd.isna(efc_tot) or pd.isna(epd) or epd <= 0:
            return None
        rul_efc_prior = max(0.0, EFC_EOL_PRIOR - float(efc_tot))
        return round(rul_efc_prior / epd, 0)

    rul_df["rul_days_prior_only"] = rul_df.apply(prior_only_days, axis=1)

    # ── Step 5: Recommended RUL ───────────────────────────────────────────────
    rul_df["rul_days_recommended"] = rul_df.apply(recommended_rul, axis=1)

    # ── Human-readable note ───────────────────────────────────────────────────
    rul_df["rul_note"] = rul_df.apply(rul_note, axis=1)

    # ── Materialise day-axis RUL columns ──────────────────────────────────────
    rul_df["rul_days_exp_day"] = rul_df["_rul_days_exp_day"].apply(
        lambda v: round(v, 0) if (v is not None and not pd.isna(v) and np.isfinite(v)) else None
    )
    rul_df["rul_years_exp_day"] = rul_df["rul_days_exp_day"].apply(
        lambda v: round(v / 365.25, 1) if (v is not None and not pd.isna(v)) else None
    )

    # ── Years conversions of existing estimates (new columns only) ─────────────
    rul_df["rul_years_exp"] = rul_df["rul_days_exp"].apply(
        lambda v: round(v / 365.25, 1) if (v is not None and not pd.isna(v)) else None
    )
    rul_df["rul_years_linear"] = rul_df["rul_days"].apply(
        lambda v: round(v / 365.25, 1) if (v is not None and not pd.isna(v)) else None
    )
    rul_df["rul_years_recommended"] = rul_df["rul_days_recommended"].apply(
        lambda v: round(v / 365.25, 1) if (v is not None and not pd.isna(v)) else None
    )

    # ── Drop internal columns ──────────────────────────────────────────────────
    rul_df.drop(columns=["_rul_efc_exp", "_rul_days_exp_day"], inplace=True)

    # ── Save ──────────────────────────────────────────────────────────────────
    rul_df.to_csv(RUL_REPORT, index=False)
    print(f"Saved: {RUL_REPORT}  ({len(rul_df)} vehicles, {len(rul_df.columns)} columns)")

    # ── Step 6: Console summary ───────────────────────────────────────────────
    print("\n" + "=" * 100)
    print("RELIABILITY-ANNOTATED RUL REPORT  (sorted by composite degradation score, worst to best)")
    print("=" * 100)

    show = rul_df[[
        "registration_number", "fit_quality", "current_soh",
        "rul_years_linear",
        "exp_fn_efc", "rul_years_exp",
        "exp_fn_day", "rul_years_exp_day",
        "rul_years_recommended",
    ]].copy()

    # Format for display
    def fmt(v):
        if v is None or (isinstance(v, float) and np.isnan(v)):
            return "-"
        if isinstance(v, float) and not np.isfinite(v):
            return "inf"
        return str(int(v))

    for col in ["rul_years_linear", "rul_years_exp", "rul_years_exp_day", "rul_years_recommended"]:
        if col in show.columns:
            show[col] = show[col].apply(fmt)

    print(show.to_string(index=False))

    # -- Tier distribution -------------------------------------------------------
    print("\n-- Fit-quality distribution --")
    print(rul_df["fit_quality"].value_counts().to_string())

    print(f"\nEFC prior   : SoH = 100 * exp(-{LFP_PRIOR_K:.2e} * EFC)  [{LFP_PRIOR_CYCLE_LIFE} EFC -> 80% SoH]")
    print(f"Day prior   : SoH = 100 * exp(-{LFP_PRIOR_K_DAY:.2e} * days)  [{FLEET_EXPECTED_LIFE_YEARS} yr -> 80% SoH]")
    print(f"Prior weight (EFC) : {LFP_PRIOR_WEIGHT_EFC}")
    print(f"Prior weight (days): {LFP_PRIOR_WEIGHT_DAYS}")
    print(f"EFC to EOL (prior) : {EFC_EOL_PRIOR:.1f} full cycles")

    # -- Sanity checks ----------------------------------------------------------
    print("\n-- Sanity checks --")
    low_efc  = rul_df[rul_df["efc_total"].fillna(0) < LFP_PRIOR_WEIGHT_EFC]
    if len(low_efc):
        k_vals = low_efc["exp_k_blended"].dropna()
        print(f"Vehicles with EFC < {LFP_PRIOR_WEIGHT_EFC}: {len(low_efc)}")
        print(f"  exp_k_blended range: [{k_vals.min():.6e}, {k_vals.max():.6e}]  "
              f"(should be approx {LFP_PRIOR_K:.6e})")
    no_sig = rul_df[rul_df["fit_quality"] == "no_degradation_signal"]
    print(f"no_degradation_signal vehicles: {len(no_sig)}  "
          f"(rul_days_recommended = None: "
          f"{no_sig['rul_days_recommended'].isna().sum()})")
