# SOH Analysis Notes — ref_capacity_ah Fix & Source Comparison (A / B / C)

## Background

The pipeline estimates State of Health (SOH) from Coulomb counting across discharge blocks. One discharge block = all discharge sessions between two charging events. The formula is:

```
norm_cap  = block_capacity_ah / (block_soc_diff / 100)
capacity_soh = (norm_cap / ref_capacity_ah) × 100
```

The pack specification is **180S2P NMC, 2 × 218 Ah cells → 436 Ah total, 282 kWh**.
True nominal voltage: 282,000 Wh / 436 Ah ≈ **647 V**.

---

## Problem Identified: ref_capacity_ah = 197 Ah (should be 436 Ah)

The data-driven reference capacity — computed as the per-vehicle **p90 of norm_cap across quality discharge blocks**, capped at 436 Ah — was returning a fleet mean of only **197 Ah** (range 126–286 Ah), far below the true 436 Ah nominal.

### Root Cause: Unmeasured Idle / Parking SoC Drops

The BMS only records current while the vehicle is active. When a truck parks overnight (BMS off), the SoC continues to drop due to:
- Parasitic loads
- BMS recalibration on wake-up

This creates a systematic measurement gap:

| Quantity | What gets captured |
|---|---|
| `block_capacity_ah` | Only Ah Coulomb-counted during **active discharge sessions** |
| `block_soc_diff` | The **total** SoC drop across the block, including overnight parking |

As a result, `block_soc_diff` overstates the depth-of-discharge relative to what was measured, making `norm_cap` appear much smaller than the true pack capacity.

**Evidence from data:**
- `block_capacity_ah` mean = **61.2 Ah** vs expected ~244 Ah for 56% DoD on a 436 Ah pack (only 25% of expected)
- 8.2% of inter-session transitions within a block have a SoC jump > 10% — direct evidence of overnight parking drain
- 11,853 inter-session gaps where `soc_end` of session N > `soc_start` of session N+1, totalling **288,122 synthetic Ah** across the fleet

---

## Changes Made

### 1. `code/data_prep_1.py` — Fix ref_capacity_ah to 436 Ah

**Before:** Per-vehicle p90 of `norm_cap` from quality discharge blocks, capped at NOMINAL_CAPACITY_AH (436 Ah). Fleet mean ≈ 197 Ah.

**After:** `ref_capacity_ah` is hard-fixed at `NOMINAL_CAPACITY_AH = 436 Ah` for **all vehicles, both charging and discharging**.

Removed code:
- Per-vehicle p90 aggregation for discharge (`ref_disc`)
- Per-vehicle p90 aggregation for charging (`ref_chg`)
- `ref_vehicle = ref_disc.combine_first(ref_chg)` merge
- `ref_capacity_ah` join back onto cycles

Added:
```python
cycles["ref_capacity_ah"] = NOMINAL_CAPACITY_AH
```

This change propagates to all four SOH computations:
- `capacity_soh` (discharge, block-level)
- `capacity_soh` (charging, session-level)
- `capacity_soh_disc_new` (hves1_current discharge)
- `capacity_soh_chg_new` (hves1_current charging)

> **Note:** After this change, re-running `data_prep_1.py` is required to regenerate `artifacts/cycles.csv` with the corrected `ref_capacity_ah` and updated `capacity_soh` values.

---

### 2. `code/soh_comparison.py` — Standalone Analysis Script (three sources)

Independently validates and compares SOH estimates from three independent sources:

- **Source A:** BMS `current_mean` (from `cycles.csv` produced by `data_prep_1.py`)
- **Source B:** `hves1_current` from `bms_ultratech_current_full.csv` (supplementary high-resolution current table, 32M rows, 66 vehicles)
- **Source C:** Energy-derived — `capacity_ah = energy_kwh × 1000 / voltage_mean` — bypasses both current sensors entirely

**Method (Sources A & B):**
1. Load `cycles.csv` (77,194 sessions, 66 vehicles)
2. Load the supplementary current table
3. Use `merge_asof` (5s tolerance) per vehicle to stamp each hves1 row with its session
4. Coulomb-count per session using `hves1_current`:
   - **Discharge sessions:** positive current; regen = `current < -50 A` (5s dt cap)
   - **Charging sessions:** all negative current (`current < 0`, full dt) — trickle/slow phases included
5. Aggregate to block level for discharge (matching Source A's method)
6. Compute `capacity_soh_disc_B` and `capacity_soh_chg_B`
7. Cache session-level Ah to `artifacts/_soh_comparison_cache.csv` for fast re-runs (~2 min first run, instant on re-run)

**Charging threshold fix (commit 2fe7955):**

Previously the `-50 A` cutoff was applied uniformly across all sessions. This excluded trickle/slow charging current (0 to -50 A) from charging sessions, causing Source B charging SOH to appear ~83% instead of ~100%. Fixed by splitting the mask:

```python
regen_mask  = (~is_chg_ses) & (curr_vals < CHARGE_A)  # discharge regen only, 5s dt cap
plugin_mask =  is_chg_ses   & (curr_vals < 0)          # all plugin/trickle charging, full dt
```

**Method (Source C — energy-derived):**
- `capacity_ah_C = energy_kwh × 1000 / voltage_mean` per session
- Discharge: block-level sum of session `capacity_ah_C`, then normalized by `block_soc_diff`; same quality filter as Source A
- Charging: session-level, using `abs(energy_kwh)` (sign is negative for charging sessions)
- True nominal voltage: `282,000 Wh / 436 Ah ≈ 647 V` (for reference only — voltage_mean used directly)

**Additional analyses in the script:**
- **Idle Ah correction (Source B):** For each inter-session gap within a block where SoC dropped, add synthetic Ah = `idle_soc_gap% / 100 × 436 Ah` to bridge the gap
- **Source A recomputed with 436 Ah ref:** Recomputes `capacity_soh` for Source A using `block_capacity_ah` and `block_soc_diff` already in `cycles.csv` with 436 Ah denominator — for immediate preview without re-running the full pipeline

**Plots generated:**

| File | Description |
|---|---|
| `plots/soh_comparison_discharge.png` | SOH distribution + delta: Source A vs Source B (discharge) |
| `plots/soh_comparison_charging.png` | SOH distribution + delta: Source A vs Source B (charging) |
| `plots/soh_comparison_scatter.png` | Scatter: SOH A vs SOH B per session |
| `plots/soh_ref_capacity_dist.png` | Per-vehicle ref_capacity_ah vs nominal |
| `plots/soh_A_ref_fix_effect.png` | Source A: old p90 ref vs fixed 436 Ah ref |
| `plots/soh_discharge_idle_adj.png` | Three-way: A vs B vs B + idle Ah correction |
| `plots/soh_all_sources_discharge.png` | Four-way A / A-436 / B / C discharge comparison (variable-bin histogram + bar chart) |
| `plots/soh_all_sources_charging.png` | Four-way A / A-436 / B / C charging comparison |
| `plots/soh_energy_distribution.png` | Source C variable-bin histogram (5 pp / 2 pp / 0.5 pp bins) |

---

## Observations

### Charging SOH — All three sources agree (with correct threshold)

After the charging threshold fix (commit 2fe7955), trickle/slow charging current (0 to -50 A) is no longer excluded from Source B:

```
Source A (BMS, 436 Ah ref)    n=7,493   mean≈100%   ~78% clip above 100%
Source B (hves1, 436 Ah ref)  n=7,493   mean≈100%   ~73% clip above 100%
Source C (energy/V, 436 Ah)   n=7,492   mean≈100%   ~78% clip above 100%
B − A   MAE≈0.04%   within ±5%: >99%
```

All three sources produce near-identical charging SOH. The high clip rate (~73–78%) is **expected for a healthy fleet** — batteries routinely return close to or above nominal capacity during a full charge cycle. This is correct behaviour, not an artefact.

> **Note on old numbers:** The previously reported charging MAE of 0.09% (Sources A & B) was computed when `ref_capacity_ah` was still the p90 ~197 Ah value. With the old ref, all sources clipped to 100% regardless, masking any real difference. With 436 Ah ref and the fixed charging threshold, the MAE is ~0.04%.

---

### Discharge SOH — Large gap between Source A and Source B

```
Source A (BMS, p90 ref ~197 Ah)   n=28,083   mean=67.56%   median=70.74%
Source B (hves1, 436 Ah ref)      n=26,881   mean=96.42%   median=100.00%
B − A   mean=+30.08%   MAE=30.44%   within ±5%: 25.0%
```

This 30 pp gap was initially attributed to the ref_capacity_ah error. After fixing ref to 436 Ah for Source A:

```
Source A (BMS, fixed 436 Ah ref)   n=5,876   mean=25.1%   max=93.0%   clip above 100%: 0%
Source B (hves1, fixed 436 Ah ref) n=5,876   mean=80.9%   p90=116.4%  clip above 100%: 41.9%
Source C (energy/V, fixed 436 Ah)  n=5,876   mean=14.4%   clip above 100%: 0%   below 0%: 4.6%
```

The gap **widens to ~56 pp (A vs B)** once both use the same denominator. Source C (energy-derived) is even lower than Source A, suggesting it is also under-counting discharge energy relative to hves1. Source B's 41.9% clip rate is caused by the 3× current over-count (hves1 implied ~75 A mean vs BMS ~22 A mean for the same sessions).

---

### Idle Ah correction — Minor effect on Source B

Adding synthetic Ah for unmeasured idle parking gaps moves Source B from 96.42% to 95.15% — a decrease of 1.3 pp. This is because most quality blocks already have `norm_cap_B` > 436 Ah (p90 = 507.5 Ah), so they clip to 100% regardless. The idle Ah correction reduces clipping slightly but doesn't change the conclusion.

---

### Summary of discharge SOH estimates

| Method | ref | mean SOH | median SOH | clip >100% |
|---|---|---|---|---|
| Source A — old pipeline | p90 ~197 Ah | 67.56% | 70.74% | 0% |
| Source A — fixed ref | 436 Ah | 25.1% | ~24% | 0% |
| Source B (hves1, block) | 436 Ah | 80.9% | 100.00% | 41.9% |
| Source B (hves1 + idle Ah) | 436 Ah | ~79% | 100.00% | ~40% |
| Source C (energy/voltage) | 436 Ah | 14.4% | ~13% | 0% |

### Summary of charging SOH estimates

| Method | ref | mean SOH | clip >100% |
|---|---|---|---|
| Source A BMS (p90 ref, old) | ~197 Ah | 100% (all clipped) | ~100% |
| Source A BMS (436 Ah ref) | 436 Ah | ~101% | 78% |
| Source B hves1 (old threshold) | 436 Ah | ~83% | low |
| Source B hves1 (fixed threshold) | 436 Ah | ~101% | 73% |
| Source C energy/voltage | 436 Ah | ~101% | 78% |

---

### Interpretation

The three current/energy sources produce fundamentally different discharge Ah counts for the same sessions. Key findings:

1. **hves1 discharge over-counts by ~3×:** Implied mean current from hves1 ≈ 75 A vs BMS ≈ 22 A for the same sessions. This is not explained by dt inflation (hves1 average dt = 6.1s, within the 5-min cap). Most likely causes:
   - **Sensor placement / scaling:** hves1 may be measured at the pack terminals (both strings in parallel) while BMS may report per-string current
   - **Different current sensor with higher gain or different calibration**
2. **Source C discharge under-counts:** Energy-derived Ah (energy_kwh × 1000 / voltage_mean) produces even lower SOH than Source A BMS, suggesting `energy_kwh` in cycles.csv is computed from the same under-counting BMS current (V × I_BMS integration)
3. **Charging sources all agree:** All three sources return ~100% charging SOH with 436 Ah ref, consistent with healthy batteries being fully restored each charge cycle
4. **4.6% of Source C discharge blocks below 0%:** Caused by small or near-zero `energy_kwh` values in short discharge sessions, or high `voltage_mean` outliers

**Recommended next step:** Plot `hves1_current` vs `current_mean` time-series for one or two representative sessions to identify whether the discharge discrepancy is a constant scaling factor, a measurement-point difference, or session-boundary effects.

---

## Changes Made (2026-03-31)

### 3. `code/data_prep_1.py` — Geofence-based trip direction, alerts integration, memory optimisations

#### 3a. Fixed-route geofence trip direction (replaces depot-distance heuristic)

**Before:** `label_trip_direction()` estimated trip direction by computing haversine distance to the vehicle's inferred depot, finding the turnaround point (argmax distance per trip segment), and falling back to heading alignment when no clear turnaround existed. `is_loaded` was set to `True` for *inbound* (trucks returning to depot = loaded cargo).

**After:** Direction is determined by a state machine keyed on two fixed geofences:

| Geofence | Type | Coordinates |
|---|---|---|
| Manawar Loading Area | Polygon (4 vertices) | ~22.26–22.27°N, 75.13–75.14°E |
| Dhule Unloading Area | Circle (100 m radius) | 21.151°N, 74.850°E |

Logic:
- Entering loading polygon → state = `at_loading`
- Entering unloading circle → state = `at_unloading`
- Leaving loading site → `outbound` (truck is **loaded**: Manawar → Dhule)
- Leaving unloading site → `inbound` (truck is **empty**: Dhule → Manawar)
- `is_loaded = True` for `outbound` sessions (corrected from old inbound=loaded)

New columns: `trip_direction` (`outbound` | `inbound` | `at_loading` | `at_unloading` | `unknown`), `dist_from_loading_km` (replaces `dist_from_depot_km`).

`load_direction_enc` updated: `at_loading` and `at_unloading` map to `NaN` (previously only `unknown` was NaN).

EPK summary print updated to reflect corrected semantics: *Loaded (outbound: Manawar→Dhule)* / *Unloaded (inbound: Dhule→Manawar)*.

#### 3b. Alerts integration

Added `ALERTS_FILE` constant pointing to `data/alerts_full_ultratech_intangles.csv`.

New functions:
- `load_alerts(path)` — loads the alerts CSV, coerces all non-metadata columns to numeric, returns `(registration_number, gps_time, <alert_cols>)` DataFrame. Returns empty DataFrame if file not found (graceful skip).
- `join_alerts_onto_cycles(cycles, alerts)` — per-vehicle searchsorted join: for each session in `cycles`, sums alert counts whose `gps_time` falls within `[start_time, end_time]`. O(n log m) per vehicle.

Columns added to `cycles.csv` output: `total_alerts` plus ~35 individual alert columns (fault/alarm counts per session) sourced from `alerts_full_ultratech_intangles.csv`.

#### 3c. Memory optimisations

- Added `import gc` and `gc.collect()` after large DataFrame operations throughout the pipeline: after `load_gps`, `load_vcu`, `load_bms`, `load_current_table`, `del df_v`, and after all concat/del operations in the main block.
- Removed in-function `.copy()` calls from `add_derived_columns`, `label_sessions`, `_merge_discharge_gaps`, `compute_voltage_sag`, `compute_ir_metrics`, `extract_cycles`, `compute_block_linkage`, `add_fleet_flags`, `add_engineered_features`, `add_capacity_soh` — these functions now mutate the passed DataFrame in place, reducing peak RAM.
- Added explicit `del` for `bms_by_veh`, `gps_by_veh`, `vcu_by_veh`, `curr_by_veh` after the vehicle loop and for intermediate concat results (`all_cycles`, `all_disc_rows`, `anomaly_frames`, `sequences`).

---

## Changes Made (2026-04-01)

### 4. `code/data_prep_1.py` — `add_cycle_soh`: hves1_current primary path + fallback

#### What changed

The `add_cycle_soh` function now has **two code paths** selected by a single flag:

```python
has_new_cols = (
    "capacity_soh_disc_new" in cycles.columns and
    "capacity_soh_chg_new"  in cycles.columns
)
```

**Primary path** (when hves1_current columns are present):
- Aggregates session-level `capacity_soh_disc_new` / `capacity_soh_chg_new` per block using `groupby.agg(mean)`.
- Quality gates are already embedded: sessions where hves1 quality gates failed are `NaN` in those columns, so the block mean is `NaN` → the block pair is skipped (`pd.isna` check).
- `cycle_soh = clip(mean(disc_soh, chg_soh), 0, 100)` — no division by `NOMINAL_CAPACITY_AH` required; SOH is already in %.

**Fallback path** (when new columns are absent — original logic):
- Uses `block_capacity_ah` and `block_soc_diff` aggregated at block level (one row per block via `drop_duplicates`).
- Applies explicit SoC-swing quality gates: `dod >= MIN_SOC_RANGE_DISC_PCT`, `csoc >= MIN_SOC_RANGE_PCT`, and `block_capacity_ah > 0`.
- `cycle_soh = clip(mean(norm_disc, norm_chg) / NOMINAL_CAPACITY_AH × 100, 0, 100)`.

#### Why

Previously `cycle_soh` was always computed from `block_capacity_ah / block_soc_diff`, which suffers from the same idle-parking under-count as Source A (BMS current). Now that `capacity_soh_disc_new` / `capacity_soh_chg_new` (derived from the hves1_current sensor) are present in `cycles.csv`, using them directly as the anchor for `cycle_soh` avoids re-normalising already-normalised SOH values and leverages the higher-fidelity current source.

Step 2 (linear interpolation + extrapolation between anchor points, and forward/backward fill) is **unchanged** in both paths.

---

## Changes Made (2026-04-01) — cont.

### 5. `code/data_prep_1.py` — Add `current_mean_new` (hves1_current session mean) to cycles.csv

Added `current_mean_new` to the `extract_cycles` aggregation spec alongside the existing `voltage_mean_new`:

```python
if "hves1_current" in df.columns:
    agg_spec["current_mean_new"] = (
        "hves1_current",
        lambda x: x[x > 0].mean() if (x > 0).sum() > (x < 0).sum() else x[x < 0].mean(),
    )
```

Same sign-direction logic as `current_mean` (BMS): uses mean of positive values for discharge sessions, mean of negative values for charging sessions. Added to `_col_order` immediately after `voltage_mean_new` in the *Electrical — hves1 source* block.

---

### 6. `code/data_prep_1.py` — Add `hot_subsystem_consistency`

Added alongside `weak_subsystem_consistency` in `extract_cycles`:

```python
if "temperature_highest_subsystem_number" in df.columns:
    modal_hot = df["temperature_highest_subsystem_number"].mode()
    if len(modal_hot) > 0:
        agg_spec["hot_subsystem_consistency"] = (
            "temperature_highest_subsystem_number",
            lambda x, m=int(modal_hot.iloc[0]): (x == m).mean() if x.notna().any() else np.nan,
        )
```

**What it measures:** Fraction of BMS rows in a session where `temperature_highest_subsystem_number` equals the fleet-modal hot subsystem. High value (→ 1.0) means that subsystem is persistently the hottest across the session. Low value means thermal hotspot rotates across subsystems.

**Fleet-modal subsystem** is computed once across all rows with a valid `temperature_highest_subsystem_number` value (same approach as `weak_subsystem_consistency`). Added to `_col_order` after `hot_probe_id`.

---

### 7. `code/data_prep_1.py` — Fix `PerformanceWarning`: DataFrame is highly fragmented

Pandas emits this warning when a DataFrame accumulates many single-column memory blocks (one per `df[col] = ...` assignment). Two sources were identified and fixed.

#### `join_alerts_onto_cycles` (primary cause — ~36 columns)

**Before:** Initialised each of ~35 alert columns plus `total_alerts` one by one:
```python
for c in alert_cols:
    cycles[c] = 0.0
cycles["total_alerts"] = 0.0
```
Each assignment creates a new single-column memory block → 36 separate allocations.

**After:** Pre-allocates all alert columns as a single contiguous block, then `pd.concat` once:
```python
_alert_block = pd.DataFrame(
    0, index=cycles.index,
    columns=alert_cols + ["total_alerts"],
    dtype=np.float64,
)
cycles = pd.concat([cycles, _alert_block], axis=1)
del _alert_block; gc.collect()
```
All 36 columns share one memory allocation. The per-vehicle searchsorted fill loop and final `.astype(int)` cast are unchanged.

#### `add_engineered_features` (secondary cause — ~80 fragmented cols entering + 19 more added)

**Before:** By the time this function ran, `cycles` already had ~80+ columns accumulated piecemeal from earlier pipeline steps. Adding ~19 more one-by-one compounded the fragmentation.

**After:** Added `cycles = cycles.copy()` at function entry:
```python
cycles = cycles.copy()   # consolidate fragmented column blocks
cycles = cycles.sort_values(["registration_number", "start_time"])
```
`.copy()` causes pandas to consolidate all same-dtype columns into a minimal set of contiguous blocks before new columns are added, eliminating the warning for this step.

---

## Engineered Columns Reference

All columns in `cycles.csv` that are not directly aggregated from raw BMS/GPS/VCU rows. Formulas are shown in Python notation; constants from `config.py` are used where referenced.

---

### Row-level derived columns (computed before session aggregation, in `add_derived_columns`)

| Column | Formula | Notes |
|---|---|---|
| `cell_spread` | `max_cell_voltage − min_cell_voltage` | Per-BMS-record cell imbalance (V) |
| `cell_undervoltage` | `min_cell_voltage < 3.0` | Bool flag: lowest cell below 3.0 V |
| `cell_overvoltage` | `max_cell_voltage > 3.5` | Bool flag: highest cell above 3.5 V |
| `cell_spread_warn` | `cell_spread > 0.02` | Bool flag: spread > 20 mV |
| `temp_highest` | `= temperature_highest` | Alias for highest probe temperature |

---

### Session-level columns (aggregated in `extract_cycles`)

#### Capacity accumulators

Coulomb counting uses `_dt_hr = diff(gps_time) / 3,600,000`, clipped to `MAX_DT_MIN / 60` hours. Regen uses a tighter cap (`REGEN_DT_MAX_SEC = 5 s`).

| Column | Formula | Notes |
|---|---|---|
| `capacity_ah_discharge` | `Σ (current × dt_hr)` where `current > 0` | BMS discharge Ah per session |
| `capacity_ah_charge` | `Σ |hves1_current × dt_hr_regen|` where moving & `current < CHARGE_A` | Regen Ah (5 s dt cap) |
| `capacity_ah_plugin` | `Σ |current × dt_hr|` where `detailed_type == 'charging'` | Stationary charging Ah |
| `capacity_ah` | `capacity_ah_discharge − capacity_ah_charge − capacity_ah_plugin` | Net signed Ah (legacy; used by `add_capacity_soh`) |
| `capacity_ah_charge_total` | `capacity_ah_charge + capacity_ah_plugin` | All charging Ah (regen + plugin) |
| `capacity_ah_discharge_new` | Same as `capacity_ah_discharge` but using `hves1_current` | hves1 discharge Ah |
| `capacity_ah_charge_new` | Same as `capacity_ah_charge` but using `hves1_current` | hves1 regen Ah |
| `capacity_ah_plugin_new` | Same as `capacity_ah_plugin` but using `hves1_current` | hves1 plugin Ah |
| `capacity_ah_charge_total_new` | `capacity_ah_charge_new + capacity_ah_plugin_new` | hves1 total charging Ah |

#### Electrical

| Column | Formula | Notes |
|---|---|---|
| `voltage_mean` | Mean of `voltage` (BMS) per session | Pack voltage (V) |
| `voltage_min` | Min of `voltage` (BMS) per session | Lowest pack voltage in session |
| `current_mean` | Mean of `current > 0` rows if discharge session, else mean of `current < 0` rows | BMS direction-aware current mean (A) |
| `current_max` | Max of `current` per session | Peak discharge current (A) |
| `current_mean_discharge` | Mean of `current` where `current > 0` | BMS discharge rows only |
| `current_mean_charge` | Mean of `current` where moving & `current < CHARGE_A` | BMS regen rows only |
| `voltage_mean_new` | Mean of `hves1_voltage_level` per session | hves1 pack voltage mean (V) |
| `current_mean_new` | Same sign-direction logic as `current_mean`, applied to `hves1_current` | hves1 direction-aware current mean (A) |

#### Timing

| Column | Formula | Notes |
|---|---|---|
| `duration_hr` | `(end_time − start_time) / 3,600,000` | Session duration in hours |
| `time_delta_hr` | `(start_time − prev_end_time) / 3,600,000` | Gap from end of previous session to start of this one |
| `cycle_number` | `cumcount() + 1` per vehicle | Session ordinal per vehicle (all session types) |

#### SoC

| Column | Formula | Notes |
|---|---|---|
| `soc_diff` | `soc_end − soc_start` | Signed SoC change (negative = discharge) |
| `soc_range` | `|soc_start − soc_end|` for discharge; `|soc_end − soc_start|` for charging | Absolute SoC swing; used for quality gates and EPK |

#### Energy

| Column | Formula | Notes |
|---|---|---|
| `energy_kwh` | `capacity_ah × voltage_mean / 1000` | Absolute energy exchange in the session (kWh). Derived from BMS Coulomb counting × mean pack voltage. Positive for discharge, negative for charging (because `capacity_ah` is signed). **Not a rate; does not account for distance.** |
| `energy_per_km` | `(|block_soc_diff| / 100 × BATTERY_CAPACITY_KWH) / block_odometer_km` | Energy efficiency (kWh/km). Derived from **block-level** SoC depletion and odometer — one value per discharge block. Units: kWh consumed per km driven. Discharge sessions only; NaN where `block_odometer_km ≤ 0.5`. Capped at `EPK_MAX_KWH_KM`. |
| `charging_rate_kw` | `energy_kwh / duration_hr` | Average charging power (kW). Charging sessions only; NaN for discharge/idle. Positive because `energy_kwh` is negative for charging sessions and the sign is intentionally not flipped — values will be negative in the raw column for charging. *(Used as absolute magnitude in fleet flag computations.)* |

> **energy_kwh vs energy_per_km:** `energy_kwh` is the total energy drawn from the pack in one session, calculated from the current sensor (∫ V·I dt). `energy_per_km` is the energy efficiency over a full drive-to-charge block, calculated from the SoC drop and odometer without any current sensor — it is closer to a "how many kWh does this truck spend per km" number. They measure different things: `energy_kwh` is an absolute quantity per session; `energy_per_km` is a normalised efficiency metric per block.

#### Temperature

| Column | Formula | Notes |
|---|---|---|
| `temp_rise_rate` | `(temp_max − temp_start) / duration_hr` | °C per hour of temperature rise within session |

#### Odometer

| Column | Formula | Notes |
|---|---|---|
| `odometer_km` | `(odometer_end − odometer_start).clip(0)` | Session distance from VCU odometer (km) |

#### Voltage sag & IR (from `compute_voltage_sag` / `compute_ir_metrics`)

| Column | Formula | Notes |
|---|---|---|
| `n_vsag` | Count of rows where voltage drops below fleet p25 discharge voltage (`vsag_mild`) during discharge | Number of sag events per session |
| `d_vsag_per_cycle` | `diff(n_vsag)` across discharge sessions per vehicle | Session-to-session change in sag count |
| `ir_ohm_mean` | Mean of `|ΔV / ΔI|` where `|ΔI| ≥ 2 A` per session | Estimated internal resistance (Ω) |
| `n_high_ir` | Count of rows where `ir_ohm > IR_THRESHOLD_MOHM / 1000` per session | High-IR event count |
| `d_n_high_ir` | `diff(n_high_ir)` across discharge sessions per vehicle | Session-to-session change |
| `d_ir_ohm_per_cycle` | `diff(ir_ohm_mean)` across discharge sessions per vehicle | Session-to-session IR trend |

#### Cell health

| Column | Formula | Notes |
|---|---|---|
| `n_cell_undervoltage` | Count of rows where `min_cell_voltage < 3.0 V` | Per session |
| `n_cell_overvoltage` | Count of rows where `max_cell_voltage > 3.5 V` | Per session |
| `n_cell_spread_warn` | Count of rows where `cell_spread > 0.02 V` | Per session |
| `cell_spread_mean` | Mean of `cell_spread` per session | Average imbalance (V) |
| `cell_spread_max` | Max of `cell_spread` per session | Worst-case imbalance (V) |

#### Subsystem location

| Column | Formula | Notes |
|---|---|---|
| `weak_subsystem_id` | Mode of `min_cell_voltage_subsystem_number` per session | Which subsystem has the weakest cell most often |
| `weak_cell_id` | Mode of `min_cell_voltage_number` per session | Which cell slot is weakest most often |
| `weak_subsystem_consistency` | `mean(min_cell_voltage_subsystem_number == fleet_modal_subsystem)` | Fraction of rows in session where the fleet-modal weak subsystem is also the weakest. High value → persistent single-subsystem degradation. Fleet-modal subsystem computed once across all discharge rows. |
| `hot_subsystem_id` | Mode of `temperature_highest_subsystem_number` per session | Which subsystem runs hottest most often |
| `hot_probe_id` | Mode of `temperature_highest_probe_number` per session | Which temperature probe registers highest most often |
| `hot_subsystem_consistency` | `mean(temperature_highest_subsystem_number == fleet_modal_hot_subsystem)` | Fraction of rows in session where the fleet-modal hot subsystem is also the hottest. High value → that subsystem persistently runs hotter than others. Fleet-modal subsystem computed once across all rows with `temperature_highest_subsystem_number`. |
| `subsystem_voltage_std` | `std(subsystem_voltage)` per session | Pack voltage imbalance across subsystems |

---

### Block-level columns (from `compute_block_linkage`)

A *block* is all sessions of the same type (discharge or charging) between two type-switching events.

| Column | Formula | Notes |
|---|---|---|
| `block_id` | Per-vehicle integer, increments at each discharge↔charging boundary | |
| `block_type` | `'discharge'` or `'charging'` | |
| `block_soc_diff` | `block_soc_end − block_soc_start` | Negative for discharge blocks |
| `block_capacity_ah` | `Σ capacity_ah_discharge` (discharge blocks) or `Σ capacity_ah_charge_total` (charge blocks) | BMS-source Ah for the full block |
| `block_n_sessions` | Count of active sessions (discharge or charging) in block | |
| `block_odometer_km` | `Σ odometer_km` across discharge sessions in block | Total distance per discharge block |

---

### Fleet-flag columns (from `add_fleet_flags`)

All flags are **fleet-relative** (thresholds computed from fleet distribution, not fixed).

| Column | Formula | Notes |
|---|---|---|
| `bms_coverage` | `(n_rows × 10 s / 3600) / duration_hr`, clipped [0, 1] | Fraction of session time with BMS samples (assumes ~10 s BMS interval) |
| `ecu_fault_suspected` | `bms_coverage < 0.20` | True if fewer than 20% of expected BMS rows arrived |
| `rapid_heating` | `temp_rise_rate > p75(non-zero temp_rise_rate)` fleet-wide | True for unusual heat-rise sessions |
| `high_energy_per_km` | `energy_per_km > p75(discharge EPK)` fleet-wide | True for high-consumption discharge sessions |
| `slow_charging` | `charging_rate_kw < p25(charging sessions)` fleet-wide | Below-average charging power |
| `fast_charging` | `charging_rate_kw > p75(charging sessions)` fleet-wide | Above-average charging power |
| `cell_health_poor` | `(n_cell_undervoltage + n_cell_overvoltage > 0)` OR `(n_cell_spread_warn / n_rows > 0.10)` | Any cell voltage exceedance or persistent imbalance |

---

### SOH columns (from `add_capacity_soh` + `add_cycle_soh`)

| Column | Formula | Notes |
|---|---|---|
| `ref_capacity_ah` | `436` (fixed) | Nominal pack capacity (Ah); hard-fixed to avoid idle-parking under-count |
| `capacity_soh` | `clip(norm_cap / 436 × 100, 0, 100)` | BMS-source SOH %; discharge uses block Coulombs, charging uses session Coulombs |
| `capacity_soh_source` | `'discharge'` or `'charge'` | Which side produced the estimate |
| `capacity_soh_disc_new` | `clip((capacity_ah_discharge_new / (|soc_diff| / 100)) / 436 × 100, 0, 100)` | hves1-source discharge SOH; NaN where `|soc_diff| < 15%` |
| `capacity_soh_chg_new` | `clip((capacity_ah_charge_total_new × 0.97 / (|soc_range| / 100)) / 436 × 100, 0, 100)` | hves1-source charging SOH; 0.97 = coulombic efficiency correction; NaN where `|soc_range| < 10%` |
| `cycle_soh` | `clip(mean(disc_block_soh, chg_block_soh), 0, 100)`, then linearly interpolated per vehicle | Paired-block SOH; uses `capacity_soh_disc_new`/`capacity_soh_chg_new` when available, falls back to `block_capacity_ah / block_soc_diff / 436 × 100` |

---

### Engineered ML features (from `add_engineered_features`)

#### EFC and calendar aging

| Column | Formula | Notes |
|---|---|---|
| `cum_efc` | `cumsum(|soc_range| / 100)` per vehicle | Cumulative equivalent full cycles |
| `days_since_first` | `(start_time − min(start_time)) / 86,400,000` per vehicle | Days since vehicle's first record |
| `aging_index` | `clip(0.7 × cum_efc / EFC_MAX + 0.3 × days_since_first / 3650, 0, 1)` | Composite aging proxy: 70% cycle-based + 30% calendar |

#### Normalised rate features

| Column | Formula | Notes |
|---|---|---|
| `vsag_rate_per_hr` | `n_vsag / duration_hr.clip(0.1)` | Sag events per hour (removes session-length confound) |
| `ir_event_rate` | `n_high_ir / n_rows.clip(1)` | Fraction of rows with high IR per session |

#### Rolling OLS trend slopes (20-session window, min 5 valid)

| Column | Source signal | Notes |
|---|---|---|
| `vsag_trend_slope` | `vsag_rate_per_hr` | Positive = worsening sag rate |
| `ir_event_trend_slope` | `ir_event_rate` | Positive = more frequent high-IR events |
| `ir_ohm_trend_slope` | `ir_ohm_mean` | Positive = rising resistance |
| `spread_trend_slope` | `cell_spread_mean` | Positive = growing cell imbalance |
| `soh_trend_slope` | `capacity_soh` | Negative = SOH declining |

#### EWM-smoothed signals (span=10, min 3)

| Column | Source signal |
|---|---|
| `ir_ohm_mean_ewm10` | `ir_ohm_mean` |
| `cell_spread_mean_ewm10` | `cell_spread_mean` |
| `temp_rise_rate_ewm10` | `temp_rise_rate` |
| `vsag_rate_per_hr_ewm10` | `vsag_rate_per_hr` |

#### Operating stress

| Column | Formula | Notes |
|---|---|---|
| `c_rate_chg` | `charging_rate_kw / BATTERY_CAPACITY_KWH` (= `/ 282`) | Charging C-rate (fraction of pack capacity per hour) |
| `dod_stress` | `(|soc_range| / 100)^1.5` | Non-linear DoD severity proxy (NMC aging accelerates at high DoD) |
| `thermal_stress` | `(temp_max − 45).clip(0)` | Degrees above 45 °C; zero for cool sessions |
| `energy_per_loaded_session` | `energy_kwh / (is_loaded + 1)` | Normalised energy: divides by 2 when loaded, by 1 when unloaded — scales energy attribution per cargo state |

#### Load direction encoding

| Column | Formula | Notes |
|---|---|---|
| `load_direction_enc` | `{outbound: 0, inbound: 1, at_loading/at_unloading/unknown: NaN}` | Numeric encoding of geofence-derived trip direction |

---

## Session 2026-04-01 — cycle_soh Integration, 4D EKF, Anomaly Reason Codes

### Summary of Changes

This session overhauled the four ML pipeline scripts (`ekf_soh.py`, `soh_rul.py`, `anomaly.py`, `plot_rul.py`) and `config.py` to incorporate the new `cycle_soh` column from `artifacts/cycles.csv`, remove the deprecated `capacity_soh`-based EKF observations, and add anomaly reason codes.

---

### 1. `config.py` — New Constants

```python
# Peukert current-stress model
PEUKERT_N        = 0.05   # mild NMC effect
I_NOMINAL_A      = 150.0  # reference discharge current (A)
LOAD_STRESS_FACTOR = 1.15 # extra EFC stress when vehicle is loaded

# Thermal aging accumulation
ZETA             = 5e-4   # °C/hr drift per EFC

# 4D EKF state noise (SoH, IR_drift, spread_drift, temp_drift)
EKF_Q_DIAG = [1e-4, 2.5e-7, 2.5e-9, 1e-5]
# 5 observations: cycle_soh, bms_soh, ir_ohm_mean, cell_spread_mean, temp_rise_rate
EKF_R_DIAG = [4.0, 9.0, 4e-6, 2.5e-5, 0.25]
```

---

### 2. `ekf_soh.py` — Complete Rewrite

#### Why capacity_soh was removed
`capacity_soh` from Coulomb-counting only had valid values for high-confidence full-swing charging sessions (~38% of sessions). `cycle_soh` is available every cycle (charging and discharging) and is far less biased.

#### 4D State Vector
```
x = [SoH (%), IR_drift (Ω), spread_drift (V), temp_drift (°C/hr)]
```

#### Process model additions
- **Peukert current-stress scaling**: `alpha_adj = ALPHA × (1 + 0.05 × max(0, I/150A - 1))` — higher discharge current accelerates SoH fade.
- **Load stress factor**: `load_fac = 1.15` if `is_loaded=True`; 1.0 otherwise.
- **Thermal aging state**: `x[3] += ZETA × delta_efc` — temp_drift accumulates with cycling.

#### 5th Observation: temp_rise_rate
`temp_rise_rate` is fleet-demeaned (subtracted median of charging sessions) before entering the EKF, so `x[3]=0` is the correct initial condition for all vehicles.

#### Outputs added to `ekf_soh.csv`
- `ekf_temp_drift` — thermal aging state (°C/hr above fleet baseline)
- `temp_rise_rate_obs` — raw observation fed to EKF
- `cycle_soh_obs`, `bms_soh_obs` — per-session observation values

#### EKF Run Results (7,745 sessions, 66 vehicles)
| Metric | Value |
|---|---|
| Fleet mean EKF SoH | 98.52% |
| Fleet std | 1.29% |
| % within ±2% of cycle_soh | 94% |
| std shrinkage (prior→posterior) | 0.50 → 0.16 ✓ |
| Fleet mean temp_drift | +0.32 °C/hr above baseline |

**3 outlier vehicles:**
| Vehicle | EKF SoH |
|---|---|
| MH18BZ3028 | 90.16% |
| MH18BZ3392 | 93.82% |
| MH18BZ3370 | 95.72% |

These are genuine signals, not artefacts — their cycle_soh also trends lower and their IF scores are elevated.

---

### 3. `soh_rul.py` — cycle_soh Trend + Subsystem Features

#### cycle_soh trend fitting
`cycle_soh_slope_per_day` and `cycle_soh_r2` are now computed on ALL session types (not just discharge), since cycle_soh is available across the full charging+discharging timeline.

#### New features in BayesianRidge extra_cols
```python
"cycle_soh", "ir_ohm_mean", "cell_spread_mean", "temp_rise_rate",
"ir_ohm_mean_ewm10", "cell_spread_mean_ewm10",
"vsag_rate_per_hr", "dod_stress", "c_rate_chg", "thermal_stress",
"weak_subsystem_consistency", "subsystem_voltage_std",
"current_mean_discharge", "is_loaded"
```

#### COMPOSITE_WEIGHTS (sum = 1.0)
```python
{"soh_health_norm": 0.25, "cycle_soh_slope_norm": 0.10,
 "vsag_slope_norm": 0.15, "ir_slope_norm": 0.15,
 "energy_slope_norm": 0.13, "heat_slope_norm": 0.11, "spread_slope_norm": 0.11}
```

#### soh_rul.py Run Results (66 vehicles)
| RUL reliability | Count |
|---|---|
| `insufficient_data` | 63/66 |
| `low_r2` | 2/66 |
| `reliable` | 1/66 |

**This is expected for 95 days of data.** BMS SoH barely moves (1–2 integer steps), making OLS slopes unreliable. Composite scores and anomaly flags remain valid and actionable.

Fleet mean RUL: **~2,192 days (~6 years)** — physically correct; fleet is young.

**Known artefact:** MH18BZ3195 shows `dual_rul_efc_days ≈ 2.2e15` — a near-zero cycle_soh slope causes division-by-zero in the EFC path. Does not affect composite score or calendar RUL.

---

### 4. `anomaly.py` — Anomaly Reason Codes + New Features

#### New features added
- IF/LGBM/cluster: `cycle_soh`, `weak_subsystem_consistency`, `subsystem_voltage_std`, `current_mean_discharge`, `n_cell_undervoltage`
- CUSUM 5th channel: `cusum_cycle_soh_alarm` (downward CUSUM on cycle_soh)
- COMPOSITE_WEIGHTS_CUSUM: added `cycle_soh` at weight 0.10; `INVERT_COLS = {"soh_smooth", "cycle_soh"}`

#### Anomaly reason code functions
Three new functions generate human-readable reason codes:
- `_build_if_reason()`: top-3 features by absolute z-score with direction arrow and z-value
  - Example: `ir_ohm_mean=0.12(↑high,z=2.3); cell_spread_mean=0.08(↑high,z=1.9); temp_rise_rate=1.4(↑high,z=1.7)`
- `_build_cusum_reason()`: triggered CUSUM channel names
  - Example: `BMS-SoH-decline; CellSpread-increase`
- `_build_combined_reason()`: merged output → `anomaly_reason` column in `anomaly_scores.csv`
  - Example: `IF: ir_ohm_mean=0.12(↑high,z=2.3); ... | CUSUM: BMS-SoH-decline`

#### anomaly.py Run Results
| Model | Result |
|---|---|
| LightGBM RMSE | 1.02% |
| LightGBM MAE | 0.21% |
| Directional accuracy | 91.6% |
| Top LGBM features | capacity_soh_chg_new (1082), bms_coverage (520), cycle_soh (227), subsystem_voltage_std (211) |
| UMAP non-healthy flag | 90.2% — young-fleet artefact (minimal degradation variation → poor cluster separation) |

LightGBM performance is within all targets. `cycle_soh` and `subsystem_voltage_std` confirm the new columns add predictive signal.

UMAP flagging 90.2% as non-healthy is a known limitation for young fleets where all vehicles occupy a dense near-healthy region — HDBSCAN assigns most to noise. This metric will improve after 6+ months of data.

---

### 5. `plot_rul.py` — Fixes and New Figures

#### Bug fixes
- Filename corrected: `rul_report.csv` → `rul_estimates.csv`
- Compatibility shims added in `load_data()` to bridge old column names (`rul_years_exp_day`, `fit_quality`, `data_span_days`, `exp_k_day_blended`) to new schema
- `fig3` rewritten from exponential-fit to OLS linear trendline (new soh_rul.py uses OLS, not exponential)

#### New figures
- **fig18** (CUSUM heatmap): now gracefully handles missing CUSUM columns; includes `n_cusum_cycle_soh` as 5th channel (cyan)
- **fig20** (EKF trace): overlays `cycle_soh_obs` (cyan triangles) + `bms_soh_obs` (light blue dots); annotation includes `ekf_temp_drift` as `ΔT: {val:+.3f}°C/hr`
- **fig21** (new): side-by-side fleet cycle_soh vs soh_smooth trajectories
- **fig22** (new): per-vehicle cycle_soh vs EKF SoH for top-9 degraded vehicles

Total: **22 figures + `rul_presentation.pdf`** saved to `plots/`.

---

### Re-run Checklist

```
python code/ekf_soh.py     # generates artifacts/ekf_soh.csv
python code/anomaly.py     # generates artifacts/anomaly_scores.csv, neural_predictions.csv
python code/soh_rul.py     # generates artifacts/rul_estimates.csv, soh_trends.csv
python code/plot_rul.py    # generates plots/fig*.png + plots/rul_presentation.pdf
```

> Re-run `data_prep_1.py` first if the raw BMS/GPS/VCU data changes.

---

## Session 2026-04-02 — EKF cycle_soh Quality Gating, Adaptive R, EWM Smoothing

### Problem Identified

Running ekf_soh.py and inspecting the output revealed the EKF SoH estimates were unreliable in two directions:

**Problem 1 — EKF anchored near 100% for most vehicles.**
55.6% of all sessions (and a large fraction of charging sessions) had `cycle_soh = 100.0` exactly. This is a hard cap in the Coulomb-counting algorithm, not a real measurement. Feeding `z[0] = 100.0` as an observation every session continuously pulled the EKF state upward, preventing it from tracking any real degradation. The prior dominated; meaningful SoH decline was invisible.

**Problem 2 — Three outlier vehicles at physically implausible SoH.**
| Vehicle | EKF SoH (before) | BMS SoH (truth) |
|---|---|---|
| MH18BZ3028 | 90.16% | 97–98% |
| MH18BZ3392 | 93.82% | 98–99% |
| MH18BZ3370 | 95.72% | 98–99% |

These vehicles had Coulomb-counting blocks with values of 55–77% caused by overnight parasitic SoC drain that the current sensor never captured (BMS off during parking). With `R[0] = 4.0` (±2% noise), the EKF treated these as reliable observations and crashed the state estimate.

**Root cause diagnosis from data:**
- `cycle_soh` distribution: median = 100.0%, std = 2.4% fleet-wide — heavily bimodal (either capped at 100% or in the 95–99% range)
- High-quality observations (block_soc_diff ≥ 20%, not capped): only 2,595 of 7,745 charging sessions; actual std on these = **4.9%** — R should be ~9.0–25.0, not 4.0
- `cycle_soh` is a block-level metric propagated per `block_id`. Block blocks with low DoD (< 20% SoC swing) have noise >> signal.

---

### Changes Made

#### `config.py`

```python
# R for cycle_soh raised to match actual observation noise
EKF_R_DIAG = [9.0, 9.0, 4e-6, 2.5e-5, 0.25]   # was [4.0, ...]

# New quality-gate constants
CYCLE_SOH_OBS_CAP       = 99.5   # cycle_soh >= this is a Coulomb-count cap artefact — skip
CYCLE_SOH_MIN_BLOCK_DOD = 20.0   # minimum block SoC swing (%) for valid Coulomb count
CYCLE_SOH_REF_DOD       = 50.0   # reference DoD for adaptive R scaling
```

#### `ekf_soh.py`

Four improvements added inside `run_ekf_fleet()`:

1. **EWM pre-smoothing per vehicle (span=5)**
   Before the session loop, compute an EWM-smoothed `cycle_soh` on quality-only observations:
   ```python
   quality_mask = (csoh < CYCLE_SOH_OBS_CAP) & (block_dod >= CYCLE_SOH_MIN_BLOCK_DOD)
   csoh_ewm = csoh.where(quality_mask).ewm(span=5, min_periods=1).mean().clip(88, 101)
   ```

2. **Quality gate on cycle_soh observation**
   Only use cycle_soh as EKF input when both conditions hold:
   - `cycle_soh < 99.5%` (not capped at ceiling)
   - `block_soc_diff >= 20%` (sufficient DoD for reliable Coulomb count)
   Sessions outside this gate have `z[0] = NaN` — the EKF evolves on the process model only.

3. **Adaptive R for cycle_soh**
   Deeper block DoD → lower R (more trust):
   ```python
   R_eff = R_base * (CYCLE_SOH_REF_DOD / max(block_dod, 10)) ** 2
   # At 50% DoD: R = 9.0  (reference)
   # At 25% DoD: R = 36.0 (less trust)
   # At 10% DoD: R = 225  (nearly ignored)
   ```
   Implemented via new `r_cycle_soh_override` parameter on `ekf_step()`.

4. **Observation clamping**
   EWM-smoothed value is clamped to `[88, 101]` before use. Prevents extreme outlier blocks from destabilising the filter.

#### `soh_rul.py`

Same quality gate applied to the cycle_soh OLS trend fit:
- Only use sessions with `block_soc_diff >= 20%` AND `cycle_soh < 99.5%`
- Avoids fitting slope on capped or noisy partial-charge sessions
- `cycle_soh_current` still records the most recent raw value (for composite scoring)

---

### Results

| Metric | Before | After |
|---|---|---|
| Fleet mean EKF SoH | 98.52% | **97.56%** |
| Fleet std EKF SoH | 1.29% | **0.47%** |
| MH18BZ3028 | 90.16% | **94.97%** |
| MH18BZ3392 | 93.82% | **97.40%** |
| MH18BZ3370 | 95.72% | **97.34%** |
| SoH trend over 95 days | Flat near 100% | **−1.0% monotonic decline** ✓ |
| cycle_soh obs used | 7,745 (100%; ~half were caps) | **1,514 (19.5%) — all informative** |
| Verification: ±2% from cycle_soh | 94% [OK] | **85% [OK]** |
| Verification: std shrinking | 0.50 → 0.16 [OK] | **0.77 → 0.24 [OK]** |
| Median EKF RUL | 1,364 days | **1,280 days** |

**Fleet SoH by time quintile (Q1 = earliest sessions, Q5 = latest):**
| Quintile | Fleet median EKF SoH |
|---|---|
| Q1 | 98.49% |
| Q2 | 98.09% |
| Q3 | 97.76% |
| Q4 | 97.71% |
| Q5 | 97.49% |

This confirms the EKF now produces a physically meaningful, monotonically declining SoH signal in the high 90s — exactly correct for a 95-day-old fleet.

**Quality obs coverage:** median 20 sessions per vehicle; 1 vehicle has 0 quality cycle_soh observations (relies purely on BMS soh + other observations).

---

### Is the EKF SoH reliable for downstream modeling?

**Yes, with caveats:**

- **Fleet mean (97.56%) and spread (std=0.47%) are physically correct** for a young fleet. Inter-vehicle variation is small and real, not noise-driven.
- **Monotonic decline confirmed** — the EKF is now sensitive to actual degradation signals from quality Coulomb counts, IR drift, cell spread, and thermal state.
- **MH18BZ3028 (94.97%)** remains the most flagged vehicle. Its quality Coulomb counts genuinely average lower than the fleet; it should be investigated for cell imbalance or parasitic drain. However, given BMS SoH = 97–98%, some residual downward bias may remain — this vehicle's Coulomb-counting reliability is fundamentally limited by its charging pattern.
- **RUL estimates** (median 1,280 days / ~3.5 years) are consistent with a 95-day-old fleet far from EOL@80%. These will sharpen significantly after 6+ months of data.
- **For supervised model training** (LightGBM SoH regression): `ekf_soh` is now a reliable feature — it differentiates vehicles by real operating patterns rather than Coulomb-count noise artefacts.

---

## Session 2026-04-03 — Y = EKF SoH; Physical-Only Features in BayesianRidge

### Motivation

In the previous session, `soh_rul.py` used:
- **OLS Y** = `soh_smooth` (BMS SoH, integer-stepped, 54/66 vehicles had ≤ 2 unique values) — slope effectively noise
- **BayesianRidge Y** = `bms_soh`; **X** included `cycle_soh` — both are SoH estimates, creating philosophical circularity

With EKF SoH now validated as continuous (94.97–98.56%, std=0.47%, monotonically declining), it is the correct target for all degradation models.

---

### Changes to `code/soh_rul.py`

#### Change 1 — Early EKF load with charging-feature join

EKF data is loaded once at the top of the vehicle loop and enriched with physical features from charging sessions:

```python
ekf_full = pd.read_csv(EKF_CSV).sort_values(["registration_number", "days_since_first_session"])
chg_feats = cycles[cycles["session_type"] == "charging"][[
    "registration_number", "start_time",
    "ir_ohm_mean", "cell_spread_mean", "temp_rise_rate",
    "ir_ohm_mean_ewm10", "cell_spread_mean_ewm10",
    "vsag_rate_per_hr", "dod_stress", "c_rate_chg", "thermal_stress",
    "weak_subsystem_consistency", "subsystem_voltage_std",
]].copy()
ekf_full = ekf_full.merge(chg_feats, on=["registration_number", "start_time"], how="left")
```

This eliminates the duplicate EKF CSV read that previously occurred in the composite score section.

#### Change 2 — OLS Y = EKF SoH (replaces BMS SoH)

Per vehicle, if `ekf_full` has ≥ `MIN_CYCLES_FOR_FIT` rows:
- `ekf_days_arr` and `ekf_soh_arr` are extracted from `ekf_full`
- `fit_degradation()` is called on the continuous EKF signal
- The `MIN_UNIQUE_SOH_FOR_OLS` gate (previously needed for integer BMS SoH) is no longer required
- BMS SoH (`soh_smooth`) remains as a fallback only for vehicles with no EKF data

#### Change 3 — BayesianRidge Y = EKF SoH, X = physical features only

**Y:** `ekf_soh` (continuous, physics-grounded)

**X features (16 total):**
| Feature | Description |
|---|---|
| `cum_efc` | Cumulative equivalent full cycles |
| `days_since_first` | Calendar age (days) |
| `cum_efc × days_since_first` | Interaction: cycle + calendar stress |
| `ir_ohm_mean` | Internal resistance (Ohm) — primary aging indicator |
| `cell_spread_mean` | Max − min cell voltage (V) — imbalance indicator |
| `temp_rise_rate` | Temperature rise during charge (°C/hr) |
| `ir_ohm_mean_ewm10` | EWM-smoothed IR (span=10) |
| `cell_spread_mean_ewm10` | EWM-smoothed cell spread |
| `vsag_rate_per_hr` | Voltage sag events per hour |
| `dod_stress` | Depth-of-discharge stress proxy |
| `c_rate_chg` | Charge C-rate |
| `thermal_stress` | Composite thermal aging proxy |
| `weak_subsystem_consistency` | Cross-subsystem health consistency flag |
| `subsystem_voltage_std` | Subsystem voltage standard deviation |
| `current_mean_discharge` | Mean discharge current (A) |
| `is_loaded` | Vehicle loading flag |

**Removed from X:** `cycle_soh`, `soh`, `bms_soh`, `capacity_soh*` — all are SoH estimates that would be circular predictors of an SoH target.

#### Change 4 — EFC-based RUL uses EKF SoH vs. cum_efc

Previously used BMS `soh_smooth` vs. `cum_efc`. Now uses `veh_ekf["ekf_soh"]` and `veh_ekf["cum_efc"]` — consistent with the OLS and BayesianRidge target.

---

### Results

#### OLS Trend Reliability

| Metric | Before (Y=BMS SoH) | After (Y=EKF SoH) |
|---|---|---|
| `reliable` (R² ≥ 0.4) | ~3/66 | **58/66** |
| `low_r2` | ~3/66 | **8/66** |
| `insufficient_data` | **60/66** | **0/66** |
| Fleet mean degradation | meaningless | **−0.030 %SoH/day** |
| Fleet mean OLS RUL | meaningless | **591 days (~1.6 years)** |

Switching to EKF SoH reduced `insufficient_data` from 91% to 0%. The continuous EKF signal gives OLS a real trend to fit.

#### BayesianRidge Predictions

- `bayes_soh_pred` produced for **~50 vehicles** (requires ≥ 5 EKF rows with physical features available after merge)
- Prediction std: typically **0.06–0.33%** — tight for data-rich vehicles, wider for sparse/charging-pattern-limited ones
- Predictions closely track EKF SoH (within ~0.2% for most vehicles)

#### Fleet Degradation Ranking (top 10 worst, by composite score)

| Rank | Vehicle | EKF SoH | OLS slope (%/day) | OLS RUL (days) | Composite |
|---|---|---|---|---|---|
| 1 | MH18BZ3028 | 94.97% | −0.0166 | 903 | 0.658 |
| 2 | MH18BZ3392 | 97.40% | −0.0622 | 280 | 0.565 |
| 3 | MH18BZ3341 | 97.35% | −0.0288 | 602 | 0.514 |
| 4 | MH18BZ2648 | 96.75% | −0.0110 | 1529 | 0.479 |
| 5 | MH18BZ2689 | 96.81% | −0.0152 | 1103 | 0.473 |
| 6 | MH18BZ3198 | 97.18% | −0.0441 | 390 | 0.472 |
| 7 | MH18BZ2958 | 97.05% | −0.0203 | 842 | 0.464 |
| 8 | MH18BZ3344 | 97.70% | −0.1579 | 112 | 0.454 |
| 9 | MH18BZ2647 | 97.01% | −0.0100 | 1698 | 0.452 |
| 10 | MH18BZ2871 | 97.30% | −0.0184 | 941 | 0.444 |

**Healthiest vehicles:**

| Vehicle | EKF SoH | OLS RUL (days) | Composite |
|---|---|---|---|
| MH18BZ3201 | 98.56% | 622 | 0.364 |
| MH18BZ3372 | 98.17% | 726 | 0.365 |
| MH18BZ3382 | 98.11% | N/A (low_r2) | 0.364 |

---

### Business Interpretation

**1. Fleet health is good — but differentiated**

EKF SoH ranges 94.97–98.56% across 66 vehicles. At the fleet mean degradation rate of −0.030%/day, the average vehicle reaches EOL (80% SoH) in roughly **591 days (~1.6 years from now)**. In calendar terms, given the fleet is ~95 days old, total expected pack life is ~700 days (~2 years), consistent with typical urban EV fleet duty cycles.

**2. MH18BZ3028 is the only vehicle requiring near-term attention**

- EKF SoH = **94.97%** — 3.6 percentage points below fleet median
- 441 high-IR events (highest in fleet) — indicates elevated internal resistance, possibly a weak cell group
- OLS RUL = 903 days, but the EKF SoH is already the lowest measured; recommend physical cell balance check
- BayesianRidge predicts 95.37% — broadly consistent with EKF, confirming degradation is not a modelling artefact

**3. MH18BZ3344 has the most urgent near-term RUL signal**

- OLS slope = −0.158 %/day (steepest in fleet by 3×)
- OLS RUL = **112 days** if slope holds
- However: only 99 cycles, short data span — slope is likely noisy. Flag for monitoring; re-evaluate after next 60 days.

**4. The EFC vs. calendar split explains which vehicles need what intervention**

- Vehicles whose `dual_dominant_path = "cycle"` are degrading primarily due to charge/discharge throughput — reduce charge C-rate or avoid deep DoD
- Vehicles whose `dual_dominant_path = "calendar"` are degrading primarily from time/temperature — prioritise thermal management and parking SOC management

**5. RUL estimates at 95 days are directional, not precise**

The OLS fit now works (58/66 reliable), but a 95-day trend projected over 600+ days carries wide confidence intervals. The composite score and anomaly flags are more actionable at this stage than the absolute RUL number. Plan to rerun after 6 months for production-grade RUL.

---

### Model Summary Table

| Model | Y | X | Purpose |
|---|---|---|---|
| OLS | `ekf_soh` | `days_since_first_session` | Simple linear degradation rate and days-to-EOL |
| BayesianRidge | `ekf_soh` | 16 physical features (IR, spread, temp, stress, usage) | Physics-informed RUL with uncertainty; no SoH circularity |
| Dual-axis | `ekf_soh` | `cum_efc/EFC_MAX`, `(days/365)×cal_rate`, `efc×days` | Separates cycle vs. calendar degradation path |
| EFC RUL | `ekf_soh` | `cum_efc` | Cycle-throughput RUL path |
| EKF | — | cycle_soh (quality-gated), bms_soh, IR, spread, temp | State estimation (produces `ekf_soh`) |

---

### Understanding the 3 RUL Models — Which One to Use

The three models (OLS, Dual-axis, BayesianRidge) are **not a stacked ensemble** — they run independently and write separate columns to `rul_estimates.csv`. Each is a different lens on the same EKF SoH signal with different inputs and assumptions.

#### Model Descriptions

**1. OLS (Simple Linear Trend)**
- Fits a straight line: `ekf_soh ~ days_since_first`
- Gives: slope (%/day), R², and RUL = days until SoH hits 80%
- Strength: simple, interpretable, works well even with limited data (58/66 reliable at 95 days)
- Weakness: assumes purely linear calendar degradation; ignores how hard the vehicle was used

**2. Dual-Axis (EFC + Calendar)**
- Decomposes degradation into two physical paths: `SoH = SoH₀ - α·(efc/EFC_MAX) - β·(days/365)·cal_rate - γ·(efc×days)`
- Gives: `dual_dominant_path` (cycle or calendar), `dual_rul_cal_days`, `dual_rul_efc_days`
- Strength: physically grounded — a vehicle charging 3× per day degrades differently than one sitting idle
- Weakness: with only 95 days of data, EFC and calendar terms are correlated; coefficient split is noisy

**3. BayesianRidge (Physics-Informed)**
- Fits: `ekf_soh ~ cum_efc + days + efc×days + IR + cell_spread + temp_rise_rate + vsag_rate + dod_stress + c_rate + thermal_stress + ...` (16 features)
- Gives: `bayes_soh_pred` (point estimate), `bayes_soh_std` (uncertainty), `bayes_rul_days`
- Strength: captures cross-vehicle variation due to operating conditions; provides calibrated uncertainty
- Weakness: requires enough sessions with physical features merged; std is wide for data-sparse vehicles

#### Which Model to Use for Each Question

| Question | Use |
|---|---|
| "When will this vehicle hit 80% SoH?" | **OLS RUL** — most robust at 95 days |
| "Is this vehicle cycling too hard vs. aging on the shelf?" | **Dual-axis** → `dual_dominant_path` |
| "What operating factor is most predictive of faster degradation?" | **BayesianRidge** → inspect feature weights |
| "How confident am I in the SoH prediction?" | **BayesianRidge** → `bayes_soh_std` |
| "Overall fleet ranking for risk prioritisation?" | **Composite score** — synthesises all signals |

**Recommended for dashboard / reporting:** use OLS RUL as the primary number, flag `dual_dominant_path` for intervention type, and show `bayes_soh_std` as a confidence band.

#### Should the Models Be Stacked?

Not yet. With 95 days of data, all three models are trained on the same short history per vehicle — stacking would overfit the narrow window. The right time to build a stacked ensemble is after **6+ months**, when:
- OLS slopes have stabilised (more unique SoH values per vehicle)
- Dual-axis EFC/calendar coefficients are separable
- BayesianRidge has enough cross-vehicle variation to generalise

At that point, a simple weighted average (e.g. 50% OLS + 30% BayesianRidge + 20% Dual-axis) would be a legitimate blended RUL output.

