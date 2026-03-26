# Battery Degradation ML Model

A production ML pipeline for **State of Health (SoH) estimation**, **Remaining Useful Life (RUL) prediction**, and **anomaly detection** for EV battery fleets. Built on real-world BMS and GPS telemetry data.

---

## Overview

This pipeline processes raw BMS and GPS data from an EV fleet, engineers session-level features, and applies a multi-model stack to track battery degradation, forecast end-of-life, and flag anomalous behaviour — per vehicle, per charging/discharge session.

```
Raw BMS + GPS telemetry
        │
        ▼
  data_prep_1.py       ← data ingestion, session segmentation, feature engineering
        │
        ├──▶ ekf_soh.py      ← Extended Kalman Filter SoH tracking + RUL
        │
        ├──▶ soh_rul.py      ← OLS/Bayesian degradation trends + composite scoring
        │
        ├──▶ anomaly.py      ← Isolation Forest, CUSUM, LightGBM, UMAP+HDBSCAN
        │
        └──▶ plot_rul.py     ← fleet-wide visualisation & diagnostic plots
```

---

## Scripts

### `data_prep_1.py` — Data Ingestion & Feature Engineering

Processes raw BMS telemetry and GPS records into structured per-session features.

**What it does:**
- Loads BMS (voltage, current, SoC, SoH, cell voltages, temperatures, IR), GPS (lat/lon, speed, altitude), and VCU (odometer) data
- Segments time-series into charging and discharge sessions using current-direction thresholds and gap detection
- Merges short discharge gaps (≤5 min) to reconstruct full trip cycles
- Computes per-session aggregates: `capacity_soh`, `ir_ohm_mean`, `cell_spread_mean`, `temp_rise_rate`, `energy_per_km`, etc.
- Flags health events: voltage sags, high-IR sessions, low-SoC, rapid heating, fast/slow charging
- Applies quality gates: minimum SOC swing (10% charging, 15% discharge), data density checks
- Derives aging features: `cum_efc` (cumulative equivalent full cycles), `days_since_first`, EWM-smoothed trend slopes
- Outputs: `cycles.csv` — one row per session, ~60 engineered features

**Key outputs:**

| Column | Description |
|---|---|
| `capacity_soh` | Coulomb-counted SoH estimate (%) |
| `cum_efc` | Cumulative equivalent full cycles |
| `ir_ohm_mean` | Mean internal resistance (Ω) |
| `cell_spread_mean` | Mean min–max cell voltage spread (V) |
| `energy_per_km` | Energy efficiency (kWh/km) |
| `n_vsag` | Voltage sag event count per session |

---

### `ekf_soh.py` — Extended Kalman Filter SoH Tracker

Physics-informed SoH estimation using a 3-state Extended Kalman Filter applied per vehicle.

**State vector:** `x = [SoH (%), IR_drift (Ω), spread_drift (V)]`

**Process model** (per charging session):
```
SoH(k+1)    = SoH(k) − α·ΔEFC − β·Δdays·(CAL_AGING_RATE/365)
IR(k+1)     = IR(k) + γ·ΔEFC
spread(k+1) = spread(k) + δ·ΔEFC
```

**Observation model:** fuses `capacity_soh`, BMS `soh`, `ir_ohm_mean`, and `cell_spread_mean` — gracefully handling NaN observations per step.

**RUL estimation:** projects days to EOL (default 80% SoH) using per-vehicle EFC/day rate, with 95% uncertainty bands propagated from the EKF covariance.

**Output:** `ekf_soh.csv`

| Column | Description |
|---|---|
| `ekf_soh` | EKF-filtered SoH estimate (%) |
| `ekf_soh_std` | 1-sigma uncertainty on SoH |
| `ekf_ir` | Estimated IR drift (Ω) |
| `ekf_spread` | Estimated cell spread drift (V) |
| `ekf_rul_days` | Point RUL estimate (days) |
| `ekf_rul_days_lo/hi` | 95% confidence interval on RUL |

---

### `soh_rul.py` — Degradation Trends & RUL Estimation

Per-vehicle statistical modelling of SoH degradation trajectories and fleet-wide composite scoring.

**What it does:**
- Smooths BMS SoH (rolling median, window=5) to remove integer-step quantisation noise
- Fits OLS linear degradation model: `SoH(t) = a·t + b`
- Fits dual-axis degradation model separating cycle aging (EFC) and calendar aging (days)
- Computes bootstrap confidence intervals (n=200) on RUL from both time and EFC axes
- Fits secondary trend slopes for `energy_per_km`, `temp_rise_rate`, `cell_spread_mean`, `ir_ohm_mean`, voltage sag rate
- Computes a **composite degradation rank** across 6 normalised signal slopes:

| Signal | Weight |
|---|---|
| SoH health deficit (EKF or OLS) | 30% |
| Voltage sag slope | 15% |
| IR growth slope | 15% |
| Energy/km slope | 15% |
| Temp rise rate slope | 13% |
| Cell spread slope | 12% |

**Outputs:** `soh_trends.csv`, `rul_estimates.csv`

---

### `anomaly.py` — Multi-Model Anomaly Detection

Three complementary anomaly detection approaches applied per discharge cycle.

**Model A — LightGBM SoH Regressor (supervised)**
- Trains on charging sessions with temporal 70/30 per-vehicle split
- Predicts `capacity_soh` from ~25 stress and aging features
- Optional Optuna hyperparameter tuning (60 trials)
- Reports RMSE, MAE, and directional accuracy on test set
- Output: `lgbm_soh_predictions.csv`

**Model B — UMAP + HDBSCAN Regime Clustering (unsupervised)**
- Reduces ~15 degradation features to 2D with UMAP
- Clusters sessions with HDBSCAN to identify operating regimes
- Flags vehicles that have transitioned out of the healthy regime cluster
- Output: `regime_clusters.csv`

**Model C — CUSUM Change-Point Detection**

Two CUSUM implementations:

*Basic CUSUM* on 4 per-vehicle signals:
- BMS SoH (downward shift)
- Energy per km (upward shift)
- Temperature rise rate (upward shift)
- Cell voltage spread (upward shift)

*Enhanced CUSUM* on composite degradation score + IR trend slope + spread trend slope.

**Isolation Forest** (fleet-wide, 5% contamination) on 18 discharge cycle features for one-off global outliers.

**Output:** `anomaly_scores.csv` with per-cycle `if_score`, `if_anomaly`, CUSUM alarm flags, and combined `anomaly` flag.

---

### `plot_rul.py` — Fleet Visualisation

Generates diagnostic and summary plots from all pipeline outputs.

**Figures produced (saved to `plots/`):**

| Figure | Description |
|---|---|
| `fig1_fleet_soh_trajectories` | Per-vehicle SoH over time with OLS trend lines |
| `fig2_rul_rankings` | Fleet RUL ranking bar chart |
| `fig3_exponential_fits` | Exponential degradation curve fits |
| `fig4_bayesian_blend` | Bayesian-blended RUL vs OLS comparison |
| `fig5_degradation_heatmap` | Multi-signal degradation heatmap |
| `fig6_anomaly_summary` | Per-vehicle anomaly counts |
| `fig7/8/9_composite` | Composite degradation score rankings |
| `fig10_rul_day_simple` | Simplified RUL-by-day view |
| `fig11–19` | Neural model errors, anomaly timelines, CUSUM heatmap, EKF SoH trace |
| `fig20_ekf_soh_trace` | EKF SoH tracking per vehicle |

---

## Pipeline Execution Order

```bash
python data_prep_1.py   # 1. process raw data → cycles.csv
python ekf_soh.py       # 2. EKF SoH tracking → ekf_soh.csv
python soh_rul.py       # 3. degradation trends → rul_estimates.csv, soh_trends.csv
python anomaly.py       # 4. anomaly detection → anomaly_scores.csv
python plot_rul.py      # 5. generate all plots
```

---

## Dependencies

```
numpy
pandas
scipy
scikit-learn
lightgbm          # Model A (optional, skip if not installed)
optuna            # LightGBM hyperparameter tuning (optional)
umap-learn        # Model B (optional)
hdbscan           # Model B (optional)
matplotlib
seaborn
tqdm
```

Install all:
```bash
pip install numpy pandas scipy scikit-learn lightgbm optuna umap-learn hdbscan matplotlib seaborn tqdm
```

---

## Configuration

All tunable constants (file paths, thresholds, noise matrices, aging rates, EOL SoH) are centralised in `config.py`. Key parameters:

| Parameter | Default | Description |
|---|---|---|
| `EOL_SOH` | 80% | End-of-life SoH threshold for RUL calculation |
| `EKF_ALPHA` | — | SoH loss per EFC (cycle aging rate) |
| `CAL_AGING_RATE` | — | Annual SoH loss from calendar aging (%) |
| `EFC_MAX` | — | Rated cycle life of the battery pack |
| `MIN_SOC_RANGE_PCT` | 10% | Minimum SOC swing for a valid charging session |

---

## Output Files

| File | Generated by | Description |
|---|---|---|
| `cycles.csv` | `data_prep_1.py` | Session-level feature table |
| `ekf_soh.csv` | `ekf_soh.py` | EKF SoH + RUL per charging session |
| `soh_trends.csv` | `soh_rul.py` | Smoothed SoH trajectories |
| `rul_estimates.csv` | `soh_rul.py` | Per-vehicle RUL + composite rank |
| `anomaly_scores.csv` | `anomaly.py` | Per-cycle anomaly flags |
| `lgbm_soh_predictions.csv` | `anomaly.py` | LightGBM SoH predictions |
| `regime_clusters.csv` | `anomaly.py` | UMAP+HDBSCAN cluster labels |
| `plots/` | `plot_rul.py` | All diagnostic figures |
