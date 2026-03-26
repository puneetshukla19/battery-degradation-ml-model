# SOH Analysis Notes — ref_capacity_ah Fix & Source Comparison

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

### 2. `code/soh_comparison.py` — New Standalone Analysis Script

A new script that independently validates and compares SOH estimates from two current sources:

- **Source A:** BMS `current_mean` (from `cycles.csv` produced by `data_prep_1.py`)
- **Source B:** `hves1_current` from `bms_ultratech_current_full.csv` (supplementary high-resolution current table, 32M rows, 66 vehicles)

**Method:**
1. Load `cycles.csv` (77,194 sessions, 66 vehicles)
2. Load the supplementary current table
3. Use `merge_asof` (5s tolerance) per vehicle to stamp each hves1 row with its session
4. Coulomb-count per session using `hves1_current` (positive = discharge, `< -50 A` = charging)
5. Aggregate to block level for discharge (matching Source A's method)
6. Compute `capacity_soh_disc_B` and `capacity_soh_chg_B`
7. Cache session-level Ah to `artifacts/_soh_comparison_cache.csv` for fast re-runs (~2 min first run, instant on re-run)

**Additional analyses in the script:**
- **Idle Ah correction (Source B):** For each inter-session gap within a block where SoC dropped, add synthetic Ah = `idle_soc_gap% / 100 × 436 Ah` to bridge the gap
- **Source A recomputed with 436 Ah ref:** Recomputes `capacity_soh` for Source A using the `block_capacity_ah` and `block_soc_diff` already in `cycles.csv` but with 436 Ah denominator — for immediate preview without re-running the full pipeline

**Plots generated:**

| File | Description |
|---|---|
| `plots/soh_comparison_discharge.png` | SOH distribution + delta: Source A vs Source B (discharge) |
| `plots/soh_comparison_charging.png` | SOH distribution + delta: Source A vs Source B (charging) |
| `plots/soh_comparison_scatter.png` | Scatter: SOH A vs SOH B per session |
| `plots/soh_ref_capacity_dist.png` | Per-vehicle ref_capacity_ah vs nominal |
| `plots/soh_A_ref_fix_effect.png` | Source A: old p90 ref vs fixed 436 Ah ref |
| `plots/soh_discharge_idle_adj.png` | Three-way: A vs B vs B + idle Ah correction |

---

## Observations

### Charging SOH — Sources A and B agree almost perfectly

```
Source A (BMS)   n=7,493   mean=99.97%   std=1.70%
Source B (hves1) n=7,493   mean=99.95%   std=1.46%
B − A            mean=−0.02%   MAE=0.09%   within ±5%: 99.8%
```

Charging Ah from both sensors are virtually identical. The charging-side SOH of ~100% is consistent with a healthy fleet (charging always restores close to full capacity).

---

### Discharge SOH — Large gap between Source A and Source B

```
Source A (BMS, p90 ref ~197 Ah)   n=28,083   mean=67.56%   median=70.74%
Source B (hves1, 436 Ah ref)      n=26,881   mean=96.42%   median=100.00%
B − A   mean=+30.08%   MAE=30.44%   within ±5%: 25.0%
```

This 30 pp gap was initially attributed to the ref_capacity_ah error. After fixing ref to 436 Ah for Source A:

```
Source A (BMS, fixed 436 Ah ref)   n=26,881   mean=31.82%   median=30.47%
Source B (hves1, fixed 436 Ah ref) n=26,881   mean=95.15%   median=100.00%
```

The gap **widens to ~63 pp** once both use the same denominator. This reveals a genuine sensor-level discrepancy: the BMS `current_mean` (Source A) records far fewer Coulombs than `hves1_current` (Source B) during the same discharge sessions.

---

### Idle Ah correction — Minor effect on Source B

Adding synthetic Ah for unmeasured idle parking gaps moves Source B from 96.42% to 95.15% — a decrease of 1.3 pp. This is because most quality blocks already have `norm_cap_B` > 436 Ah (p90 = 507.5 Ah), so they clip to 100% regardless. The idle Ah correction reduces clipping slightly but doesn't change the conclusion.

---

### Summary of discharge SOH estimates

| Method | ref | mean SOH | median SOH |
|---|---|---|---|
| Source A — old pipeline | p90 ~197 Ah | 67.56% | 70.74% |
| Source A — fixed ref | 436 Ah | 31.82% | 30.47% |
| Source B (hves1, block) | 436 Ah | 96.42% | 100.00% |
| Source B (hves1 + idle Ah) | 436 Ah | 95.15% | 100.00% |

---

### Interpretation

The two current sources produce fundamentally different discharge Ah counts for the same sessions. The most likely explanations are:

1. **Sensor placement / scaling:** `hves1_current` may be measured at a different point in the power path (e.g., pack-side vs motor-side), or may use a different current sensor with a higher gain.
2. **Sampling rate / integration:** The supplementary table (32M rows, ~66 vehicles) may have a higher temporal resolution than what `current_mean` in cycles.csv captures after session-level aggregation.
3. **Sign / offset correction:** The BMS may apply corrections (e.g., zero-offset calibration) that reduce the reported current magnitude.

**Recommended next step:** Plot `hves1_current` vs `current_mean` time-series for one or two representative sessions to identify whether the discrepancy is a scaling factor, an offset, or session-boundary effects.
