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

### `cell_diagnostics.py` — Cell & Pack Replacement Ranking

Identifies which physical battery subsystem (pack module) and cell is most degraded and needs replacing, using probe and subsystem number fields from the BMS raw data.

**Signals used:**

| Field | What it tells us |
|---|---|
| `min_cell_voltage_subsystem_number` | Which subsystem hosts the weakest cell each timestamp |
| `min_cell_voltage_number` | Index of the weakest cell within that subsystem |
| `temperature_highest_subsystem_number` | Which subsystem runs hottest |
| `temperature_highest_probe_number` | Which temperature probe is the hotspot |
| `insulation_resistance` | Isolation fault trend — declining = cell/harness degradation |
| `subsystem_voltage` | Per-subsystem pack voltage — std across subsystems = imbalance |

**Method:**
1. Stamps each BMS row with its session using `merge_asof` per vehicle
2. Per session: takes the modal (most frequent) subsystem/cell index for voltage and temperature signals
3. Per vehicle: counts how often each subsystem appears as weakest / hottest across all discharge sessions
4. Computes a **composite replacement score** per subsystem per vehicle:
   - 60% — % of sessions where subsystem is the weakest (voltage)
   - 20% — % of sessions where subsystem is the hottest (thermal)
   - 20% — how low the mean min cell voltage is (inverted, normalised)
5. Ranks subsystems within each vehicle — rank 1 = highest replacement priority

**ML integration (via `data_prep_1.py`):**

Six new features are added to `cycles.csv` at session level:

| Feature | Description |
|---|---|
| `weak_subsystem_id` | Modal subsystem with min cell voltage in session |
| `weak_cell_id` | Modal cell index with min cell voltage in session |
| `weak_subsystem_consistency` | Fraction of session rows where modal subsystem is weakest (0–1) |
| `hot_subsystem_id` | Modal subsystem with highest temperature in session |
| `hot_probe_id` | Modal temperature probe with highest reading |
| `subsystem_voltage_std` | Std of subsystem voltages in session (pack imbalance) |

These are consumed by `anomaly.py`'s LightGBM and Isolation Forest models:
- A **recurring `weak_subsystem_id`** across sessions flags a degraded pack module
- **Low `weak_subsystem_consistency`** (< 0.5) means degradation is spreading — multiple cells competing for weakest
- **Rising `subsystem_voltage_std`** indicates growing pack imbalance between modules
- A **persistent `hot_subsystem_id`**  with rising temperature trend points to a cooling or cell issue in a specific module

**Outputs:**

| File | Description |
|---|---|
| `artifacts/cell_health_ranking.csv` | Per-vehicle subsystem replacement ranking with scores |
| `plots/cell_weak_subsystem_heatmap.png` | Fleet heatmap — % sessions each subsystem is weakest |
| `plots/cell_hot_subsystem_heatmap.png` | Fleet heatmap — % sessions each subsystem is hottest |
| `plots/cell_min_voltage_trend.png` | Weakest-cell voltage trend over time (top 10 vehicles) |
| `plots/cell_insulation_trend.png` | Insulation resistance trend per vehicle |
| `plots/cell_pack_imbalance.png` | Per-vehicle subsystem voltage std (pack imbalance bar chart) |

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

## Data Extraction (AWS Athena)

All input data is pulled from Athena. Run each query in the Athena console (or via `boto3`) and export as CSV to the `data/` directory before running the pipeline.

---

### 1. BMS Telemetry — `data/bms_full_ultratech_intangles_more_cols_full.csv`

Core BMS signals: voltage, current, SoC, SoH, cell voltages, temperatures, relay states, power limits, and insulation resistance.

```sql
SELECT
    registration_number,
    gps_time,
    event_datetime,
    vendor,
    spv,
    voltage,
    current,
    soc,
    soh,
    battery_operating_state,
    status_heating_control,
    status_cooling_control,
    status_charge_relay_off,
    status_charge_relay_on,
    status_precharge_relay,
    status_positive_relay,
    status_negative_relay,
    max_discharge_power_limit,
    max_charge_power_limit,
    max_discharge_current_limit,
    max_charge_current_limit,
    min_cell_voltage,
    max_cell_voltage,
    min_cell_voltage_number,
    max_cell_voltage_number,
    min_cell_voltage_subsystem_number,
    max_cell_voltage_subsystem_number,
    temperature_lowest,
    temperature_highest,
    temperature_lowest_probe_number,
    temperature_highest_probe_number,
    temperature_lowest_subsystem_number,
    temperature_highest_subsystem_number,
    insulation_resistance,
    subsystem_voltage,
    subsystem_number,
    subsystem_total_number,
    subsystem_current
FROM "unified"."bms_raw_parsed"
WHERE spv = 'ULTRATECH'
  AND vendor = 'intangles';
```

---

### 2. GPS / Location — `data/gps_full_ultratech_intangles.csv`

GPS telemetry: coordinates, altitude, heading, speed.

```sql
SELECT
    registration_number,
    gps_time,
    event_datetime,
    latitude,
    longitude,
    altitude,
    head,
    speed,
    vendor,
    spv
FROM "unified"."location"
WHERE spv = 'ULTRATECH'
  AND vendor = 'intangles';
```

---

### 3. MCU Data — `data/mcu_full_ultratech_intangles.csv`

Motor controller unit raw signals.

```sql
SELECT *
FROM "unified"."mcu_raw_parsed"
WHERE spv = 'ULTRATECH'
  AND vendor = 'intangles';
```

---

### 4. VCU Data — `data/vcu_full_ultratech_intangles.csv`

Vehicle controller unit signals (includes odometer).

```sql
SELECT *
FROM "unified"."vcu_raw_parsed"
WHERE spv = 'ULTRATECH'
  AND vendor = 'intangles';
```

---

### 5. High-Resolution Current Table — `data/bms_ultratech_current_full.csv`

Supplementary table with high-resolution `hves1_current` and `hves1_voltage_level` used for Source B SOH comparison. Approximately 32 million rows across 66 vehicles.

```sql
SELECT
    registration_number,
    timestamp,
    hves1_voltage_level,
    hves1_current
FROM "processed"."intangles_bms"
WHERE event_datetime > date '2025-10-01'
  AND spv = 'ULTRATECH';
```

> **Note:** Remove the `event_datetime` filter to pull the full history. The filter above limits to data from October 2025 onwards. Adjust the date range as needed for the analysis window.

---

### Export instructions

1. Run each query in the [Athena console](https://console.aws.amazon.com/athena/)
2. Download the result CSV from the S3 output location (shown under **Query results** after execution)
3. Rename the file to match the name shown above and place it in the `data/` directory
4. The `data/` directory is excluded from git (see `.gitignore`) — data files are never committed to the repo

---

## Pipeline Execution Order

```bash
python code/data_prep_1.py      # 1. process raw data → cycles.csv (with cell/subsystem features)
python code/cell_diagnostics.py # 2. pack & cell replacement ranking → cell_health_ranking.csv
python code/ekf_soh.py          # 3. EKF SoH tracking → ekf_soh.csv
python code/soh_rul.py          # 4. degradation trends → rul_estimates.csv, soh_trends.csv
python code/anomaly.py          # 5. anomaly detection → anomaly_scores.csv
python code/plot_rul.py         # 6. generate all plots

# Optional — standalone source comparison & analysis
python code/soh_comparison.py   # compare BMS vs hves1_current SOH estimates
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

All CSVs and NPY files are written to `artifacts/`. All plots are written to `plots/`. Neither directory is committed to git.

---

### `data_prep_1.py`

| File | Description |
|---|---|
| `artifacts/cycles.csv` | One row per session — ~100 engineered features covering SoC, capacity SOH, IR, cell spread, temperatures, block linkage, subsystem location, aging index, EWM trends |
| `artifacts/sequences.npy` | NumPy array shaped `(N, NUM_BINS, N_FEATURES)` — binned discharge sequences for the neural model |
| `artifacts/sequence_meta.csv` | Index mapping: sequence position → `registration_number`, `cycle_number` |

---

### `cell_diagnostics.py`

| File | Description |
|---|---|
| `artifacts/cell_health_ranking.csv` | Per-vehicle, per-subsystem replacement score and rank. Columns: `subsystem_id`, `replacement_score`, `pct_sessions_weakest`, `pct_sessions_hottest`, `mean_min_cell_v`, `mean_insulation_ohm` |
| `artifacts/_cell_diag_cache.csv` | Cached per-session cell stats (weak/hot subsystem IDs, min voltage, insulation) — re-used on reruns to skip the merge_asof pass |
| `plots/cell_weak_subsystem_heatmap.png` | Fleet heatmap — % of discharge sessions each subsystem is the weakest (lowest min cell voltage) |
| `plots/cell_hot_subsystem_heatmap.png` | Fleet heatmap — % of discharge sessions each subsystem is the hottest |
| `plots/cell_min_voltage_trend.png` | Weekly min cell voltage trend for the top 10 highest-score vehicles |
| `plots/cell_insulation_trend.png` | Weekly insulation resistance trend per vehicle — declining trend signals isolation fault |
| `plots/cell_pack_imbalance.png` | Mean subsystem voltage std per vehicle — high std = modules out of equilibrium |

---

### `ekf_soh.py`

| File | Description |
|---|---|
| `artifacts/ekf_soh.csv` | One row per charging session per vehicle. Columns: `ekf_soh`, `ekf_soh_std`, `ekf_ir` (internal resistance drift), `ekf_spread` (cell spread drift), `ekf_rul_days`, `ekf_rul_days_lo`, `ekf_rul_days_hi` (95% CI) |

---

### `soh_rul.py`

| File | Description |
|---|---|
| `artifacts/soh_trends.csv` | Per-cycle SoH smoothed with rolling median — used for OLS/exponential fitting and visualisation |
| `artifacts/rul_estimates.csv` | Per-vehicle RUL table. Columns include: `rul_years_exp_day`, `exp_A_day`, `exp_k_day_blended`, `composite_degradation_score`, bootstrap CI bounds |

---

### `anomaly.py`

| File | Description |
|---|---|
| `artifacts/anomaly_scores.csv` | One row per discharge cycle. Columns: `if_score`, `if_anomaly` (Isolation Forest), CUSUM alarm flags for SoH / energy-per-km / temperature / cell spread, combined `anomaly` flag |
| `artifacts/lgbm_soh_predictions.csv` | LightGBM SOH predictions on the test split — `capacity_soh_pred`, actual `capacity_soh`, residual, per vehicle |
| `artifacts/regime_clusters.csv` | UMAP 2D embedding + HDBSCAN cluster label per discharge session — used to flag regime transitions |

---

### `neural_model.py`

| File | Description |
|---|---|
| `artifacts/neural_soh_model.keras` | Trained dual-input autoencoder — sequence encoder + scalar feature branch, reconstructs discharge sequences |
| `artifacts/neural_predictions.csv` | Per-cycle reconstruction error, anomaly percentile threshold, and binary `anomaly_neural` flag |

---

### `build_soh_blocks.py`

| File | Description |
|---|---|
| `artifacts/capacity_soh_blocks.csv` | Block-level SOH — one row per discharge block with `block_capacity_ah`, `block_soc_diff`, `norm_cap`, `capacity_soh` |

---

### `soh_comparison.py`

| File | Description |
|---|---|
| `artifacts/_soh_comparison_cache.csv` | Cached per-session Ah counts from `hves1_current` — skips the 32M-row merge_asof pass on reruns |
| `plots/soh_comparison_discharge.png` | SOH distribution + delta histogram: Source A (BMS current) vs Source B (hves1_current), discharge sessions |
| `plots/soh_comparison_charging.png` | SOH distribution + delta histogram: Source A vs Source B, charging sessions |
| `plots/soh_comparison_scatter.png` | Scatter: SOH_A vs SOH_B per session, discharge and charging side by side |
| `plots/soh_ref_capacity_dist.png` | Per-vehicle ref_capacity_ah distribution vs 436 Ah nominal, with implied pack energy curve |
| `plots/soh_A_ref_fix_effect.png` | Source A SOH before/after fixing ref_capacity_ah from p90 (~197 Ah) to 436 Ah |
| `plots/soh_discharge_idle_adj.png` | Three-way discharge comparison: Source A / Source B / Source B + idle Ah correction |

---

### `diagnostics.py`

| File | Description |
|---|---|
| `plots/fig_diag_1_fleet_age_soh_quality.png` | Fleet age distribution and SoH data quality assessment |
| `plots/fig_diag_2_session_filter_analysis.png` | Effect of session filtering thresholds on session counts |
| `plots/fig_diag_3_ols_vs_ekf_rul.png` | OLS linear RUL vs EKF RUL comparison per vehicle |
| `plots/fig_diag_4_lgbm_diagnostics.png` | LightGBM model residuals, feature importance, and cross-validation |
| `plots/fig_diag_5_ekf_quality.png` | EKF filter convergence and uncertainty band quality |
| `plots/fig_diag_6_rul_improvement_scenarios.png` | RUL sensitivity under varying degradation rate scenarios |

---

### `verify_exp.py`

| File | Description |
|---|---|
| `artifacts/efc_vs_day_rul.csv` | Vehicle-level comparison of EFC-axis vs day-axis blended RUL |
| `plots/verify_exp_all6.png` | 3×2 grid — linear and semi-log degradation panels for 6 selected vehicles |
| `plots/verify_exp_<VEH_ID>.png` | Individual 2-panel verification plot per vehicle (one file per vehicle) |

---

### `plot_rul.py`

| File | Description |
|---|---|
| `plots/fig1_fleet_soh_trajectories.png` | Per-vehicle SoH over time with OLS trend lines |
| `plots/fig2_rul_rankings.png` | Fleet RUL ranking horizontal bar chart |
| `plots/fig3_exponential_fits.png` | Exponential degradation curve fits for top degraded vehicles |
| `plots/fig4_bayesian_blend.png` | Bayesian-blended RUL vs OLS comparison |
| `plots/fig5_degradation_heatmap.png` | Multi-signal degradation heatmap across fleet |
| `plots/fig6_anomaly_summary.png` | Per-vehicle anomaly flag count stacked bar chart |
| `plots/fig7_composite_ranking.png` | Composite degradation score ranking with signal component scatter |
| `plots/fig8_composite_simple.png` | Simplified composite score ranking |
| `plots/fig9_composite_vertical.png` | Vertical layout composite score ranking |
| `plots/fig10_rul_day_simple.png` | Simplified RUL (day-axis) ranking |
| `plots/fig11_neural_error_dist.png` | Neural autoencoder reconstruction error distribution *(if model exists)* |
| `plots/fig12_neural_anomaly_bar.png` | Neural anomaly count per vehicle *(if model exists)* |
| `plots/fig13_neural_error_vs_efc.png` | Reconstruction error vs cumulative EFC *(if model exists)* |
| `plots/fig14_neural_anomaly_timeline.png` | Neural anomaly timeline across fleet *(if model exists)* |
| `plots/fig15_neural_vs_if.png` | Neural score vs Isolation Forest score comparison *(if model exists)* |
| `plots/fig16_neural_vs_composite.png` | Neural score vs composite degradation score *(if model exists)* |
| `plots/fig17_if_score_dist.png` | Isolation Forest anomaly score distribution |
| `plots/fig18_cusum_heatmap.png` | CUSUM change-point detection heatmap |
| `plots/fig19_anomaly_timeline.png` | Fleet-wide anomaly event timeline |
| `plots/fig20_ekf_soh_trace.png` | EKF SoH state trace per vehicle *(if EKF data exists)* |

---

### `rul_report.py`

| File | Description |
|---|---|
| `artifacts/rul_report.csv` | Comprehensive per-vehicle RUL report — trend data, anomaly counts, degradation scores, fit quality flags, all sources consolidated |

---

### `fleet_power_stats.py`

Prints fleet-wide power and energy statistics to the console. **No files written.**
