"""
Shared configuration for fleet battery SoH / RUL analysis.
All paths and constants live here so every other script imports them.
"""
import os

# ── Paths ──────────────────────────────────────────────────────────────────────
# Project root = parent of the code/ directory (portable across machines)
BASE         = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR     = os.path.join(BASE, "data")        # raw input data
ARTIFACTS_DIR = os.path.join(BASE, "artifacts")  # generated CSVs, models, numpy files
PLOTS_DIR    = os.path.join(BASE, "plots")        # all plot images

# Create output directories if they don't exist
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(ARTIFACTS_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)

BMS_FILE   = os.path.join(DATA_DIR, "bms_full_ultratech_intangles_more_cols_full.csv")
GPS_FILE   = os.path.join(DATA_DIR, "gps_full_ultratech_intangles.csv")
VCU_FILE   = os.path.join(DATA_DIR, "vcu_full_ultratech_intangles.csv")
CYCLES_CSV        = os.path.join(ARTIFACTS_DIR, "cycles.csv")         # one row per session
SESSIONS_ROWS_CSV = os.path.join(ARTIFACTS_DIR, "sessions_rows.csv")  # unaggregated row-level CSV
TELEMETRY_DB      = os.path.join(ARTIFACTS_DIR, "telemetry.db")       # unaggregated row-level SQLite
SEQ_NPY    = os.path.join(ARTIFACTS_DIR, "sequences.npy")      # discharge sequences (N, BINS, FEATS)
SEQ_META   = os.path.join(ARTIFACTS_DIR, "sequence_meta.csv")  # maps seq index → vehicle / cycle

# ── Data-quality filters ───────────────────────────────────────────────────────
# Based on physical limits for this fleet (650 V NMC pack, fleet EV)
VOLTAGE_RANGE   = (400.0, 760.0)    # total pack voltage (V)
CURRENT_RANGE   = (-2000.0, 2000.0) # A — wide; only drops clear sensor faults
                                    # NOTE: upper bound must be >> HIGH_CURRENT_A (150A)
                                    # so high-discharge rows (voltage sag, IR events) are not filtered out
CELL_V_RANGE    = (2.5,   4.3)      # per-cell voltage (V)
TEMP_RANGE      = (-10.0, 80.0)     # °C
SOH_MIN         = 50.0              # drop 0.0 (corrupt) readings

GPS_GAP_MAX_SEC  = 120   # seconds — GPS speed/position is nulled when nearest GPS
                          # record is older than this. Prevents stale-speed regen misclassification.
ODO_GAP_MAX_SEC  = 300   # seconds — VCU odometer is nulled when nearest VCU record
                          # is older than this. Prevents spurious energy/km values.
EPK_MAX_KWH_KM   = 5.0   # physical upper bound for energy/km (kWh/km).
                          # 282 kWh pack × 5 kWh/km = ~56 km range: conservative minimum.
                          # Trips exceeding this are GPS/odometer artifacts → set to NaN.
REGEN_SPEED_KPH  = 5.0   # kph — minimum speed to consider a negative-current event as
                          # regenerative braking rather than plug-in charging

# ── Session detection (from current sign — no GPS needed) ─────────────────────
# Convention in this BMS: positive current = discharging, negative = charging
DISCHARGE_A      =  20.0   # current ABOVE this  → discharging
CHARGE_A         = -50.0   # current BELOW this  → charging (sustained charge events)
MIN_SESSION_MIN  =  10.0   # drop sessions shorter than 10 minutes
TRIP_GAP_MIN     =  15.0   # minutes — inter-session idle gap that ends a trip
MIN_BMS_ROWS     =   5     # minimum rows to keep a session
MAX_DT_MIN       =   5.0   # cap Δt in coulomb counting (avoids inflating Ah
                            # when BMS goes silent during parking/off periods)

# ── Sequence extraction ────────────────────────────────────────────────────────
NUM_BINS     = 20
SEQ_FEATURES = ["voltage", "current", "soc", "cell_spread", "temp_highest"]
# cell_spread = max_cell_voltage - min_cell_voltage  (cell imbalance; grows with aging)

# Session-level scalar health features fed into the neural model's SoH head
# and used as additional Isolation Forest features
SCALAR_FEATURES = [
    "n_vsag",
    "n_high_ir", "n_low_soc", "temp_rise_rate", "energy_per_km",
    "cell_spread_mean",   
    "bms_coverage",       # data quality — helps model weight sequences
    "time_delta_hr",      
]

#  Health flag thresholds 
IR_THRESHOLD_MOHM    =  30.0    #IR threshold (|ΔV/ΔI|)
LOW_SOC_PCT          =  20.0    
BATTERY_CAPACITY_KWH  = 282.0   
NOMINAL_VOLTAGE_V     = 630.0   
NOMINAL_CAPACITY_AH   = 436.0

#  RUL 
EOL_SOH = 80.0  

# ── Battery life & aging parameters ──────────────────────────────────────────
EFC_MAX        = 3000     
CAL_AGING_LO   = 0.03    
CAL_AGING_HI   = 0.06    
CAL_AGING_RATE = 0.045   

#  EKF physical parameters
EKF_ALPHA = 0.005   # %SoH / EFC  (was 0.007)

# Current-stress scaling (Peukert-like): extra SoH fade fraction per unit I/I_nom above 1
PEUKERT_N       = 0.05    # dimensionless — mild effect for LFP chemistry
I_NOMINAL_A     = 150.0   # A — typical fleet discharge current used as reference
LOAD_STRESS_FACTOR = 1.15 # multiplicative EFC stress when vehicle is loaded (is_loaded=True)
ZETA            = 5e-4    # °C/hr drift per EFC (thermal aging accumulation)

#  EKF state-space noise matrices
# State vector: [SoH (%), IR_drift (Ω), spread_drift (V), temp_drift (°C/hr)]
# Process noise Q — 4 diagonal entries
EKF_Q_DIAG = [1e-4, 2.5e-7, 2.5e-9, 1e-5]   # [0.01², 0.0005², 0.00005², ~0.003²]
# Observation noise R — diagonal:
#   [cycle_soh, bms_soh, ir_ohm_mean, cell_spread_mean, temp_rise_rate]
# cycle_soh noise 9.0 (3²): Coulomb-count has ~5% actual std on quality sessions;
#   R is further scaled adaptively by block DoD depth (see ekf_soh.py).
# bms_soh noise 9.0 (3²): integer-stepped, rely less on it.
# temp_rise_rate noise 0.25 (0.5²): high ambient variability; weak thermal signal.
EKF_R_DIAG = [9.0, 9.0, 4e-6, 2.5e-5, 0.25]  # [3², 3², 0.002², 0.005², 0.5²]

# cycle_soh quality gates for EKF observation
# cycle_soh >= this value is treated as an uninformative cap — skipped (set to NaN)
CYCLE_SOH_OBS_CAP        = 99.5
# Minimum block DoD (% SoC swing) for a cycle_soh observation to be accepted
CYCLE_SOH_MIN_BLOCK_DOD  = 20.0
# Adaptive R scaling reference depth: at this DoD, R = EKF_R_DIAG[0]
# Shallower blocks get higher R (less trusted): R_eff = R_base * (REF_DOD/block_dod)^2
CYCLE_SOH_REF_DOD        = 50.0

# EKF output file
EKF_CSV = os.path.join(ARTIFACTS_DIR, "ekf_soh.csv")

#  Session quality filters for trend fitting & supervised models 
# Do NOT apply to: CUSUM, Isolation Forest, EKF (benefit from all sessions).
MIN_SOC_RANGE_FOR_TREND = 10.0   # discharge session
MIN_UNIQUE_SOH_FOR_OLS  = 3      # need at least 3 distinct BMS SoH values before OLS is meaningful (avoids fitting # integer-quantisation noise on young vehicles)
OLS_R2_THRESHOLD        = 0.4    # R² below this -> flag as "low_r2" reliability

#  Reproducibility 
SEED = 42