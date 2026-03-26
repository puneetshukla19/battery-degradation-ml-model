import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) if "__file__" in dir() else os.getcwd())

"""
neural_model.py — Scalar-conditioned sequence autoencoder.

Architecture (dual-input):
  Input 1: discharge sequence  (NUM_BINS, N_SEQ_FEATURES)
    -> iTransformer encoder (feature-as-token attention)
    -> D_MODEL-dim embedding  [emb]
    |
  Input 2: scalar health features  (N_SCALARS,)
    -> Dense(32, gelu) -> Dropout  [s]
    |
    Concatenate([emb, s])
    -> Reconstruction decoder: Dense(128) -> Dense(NUM_BINS * N_SEQ_FEATURES)

Loss = reconstruction_MSE only.
SoH head removed: BMS SoH is 97-98% across fleet (near-constant, ~0 variance)
so a regression head on it wastes 30% of training signal.

Outputs
-------
neural_soh_model.keras    Saved model
neural_predictions.csv    Per-cycle reconstruction error + anomaly score (ALL sequences)
"""

import os, random
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks
from config import SEQ_NPY, SEQ_META, ARTIFACTS_DIR, SEED, NUM_BINS, SEQ_FEATURES, SCALAR_FEATURES

MODEL_PATH = os.path.join(ARTIFACTS_DIR, "neural_soh_model.keras")
PRED_FILE  = os.path.join(ARTIFACTS_DIR, "neural_predictions.csv")

EPOCHS          = 150
BATCH           = 64
D_MODEL         = 32    # reduced from 64 — limits capacity for small dataset
DROPOUT         = 0.40  # increased from 0.2 — stronger regularisation
L2              = 1e-3  # weight decay on Dense layers

random.seed(SEED); np.random.seed(SEED); tf.random.set_seed(SEED)
os.environ["PYTHONHASHSEED"] = str(SEED)


# ── Model ──────────────────────────────────────────────────────────────────────

def build_model(n_bins: int, n_feats: int, n_scalars: int) -> tf.keras.Model:
    """
    Dual-input scalar-conditioned autoencoder.
      seq_in    -> iTransformer encoder -> emb
      scalar_in -> Dense branch         -> s
      Concatenate([emb, s]) -> Reconstruction decoder

    Regularisation: reduced D_MODEL (32), higher dropout (0.4),
    L2 on all Dense layers, single transformer block.
    """
    reg = tf.keras.regularizers.l2(L2)

    # ── Sequence input + iTransformer encoder (1 block) ──────────────────────
    seq_in = layers.Input(shape=(n_bins, n_feats), name="seq")
    x = layers.Permute((2, 1))(seq_in)        # (F, T) — features as tokens
    x = layers.Dense(D_MODEL, kernel_regularizer=reg)(x)
    # Single transformer block (was 2) — reduces capacity further
    a = layers.MultiHeadAttention(num_heads=4, key_dim=D_MODEL // 4,
                                  dropout=DROPOUT)(x, x)
    x = layers.LayerNormalization()(layers.Add()([x, a]))
    f = layers.Dense(D_MODEL * 2, activation="gelu", kernel_regularizer=reg)(x)
    f = layers.Dropout(DROPOUT)(f)
    f = layers.Dense(D_MODEL, kernel_regularizer=reg)(f)
    x = layers.LayerNormalization()(layers.Add()([x, f]))
    emb = layers.GlobalAveragePooling1D()(x)   # (D_MODEL,)

    # ── Scalar health feature branch ─────────────────────────────────────────
    scalar_in = layers.Input(shape=(n_scalars,), name="scalars")
    s = layers.Dense(16, activation="gelu", kernel_regularizer=reg)(scalar_in)
    s = layers.Dropout(DROPOUT)(s)

    # ── Scalar-conditioned reconstruction decoder ─────────────────────────────
    combined = layers.Concatenate()([emb, s])
    rec = layers.Dense(64, activation="gelu", kernel_regularizer=reg)(combined)
    rec = layers.Dropout(DROPOUT)(rec)
    rec = layers.Dense(n_bins * n_feats)(rec)
    rec = layers.Reshape((n_bins, n_feats), name="reconstruction")(rec)

    return models.Model([seq_in, scalar_in], rec, name="SoH_Autoencoder_v4")


# ── Data loading ───────────────────────────────────────────────────────────────

def load_data():
    X    = np.load(SEQ_NPY).astype(np.float32)
    meta = pd.read_csv(SEQ_META)

    groups = meta["registration_number"].values

    # Load scalar health features; fill missing with column median
    missing_scalars = [f for f in SCALAR_FEATURES if f not in meta.columns]
    if missing_scalars:
        print(f"  Warning: scalar features not in seq_meta (re-run data_prep_1.py): {missing_scalars}")
        for f in missing_scalars:
            meta[f] = 0.0

    S = meta[SCALAR_FEATURES].copy()
    S = S.fillna(S.median()).values.astype(np.float32)

    return X, S, groups, meta


# ── Training ───────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Loading sequences ...")
    X, S, groups, meta = load_data()
    n_bins, n_feats = X.shape[1], X.shape[2]
    n_scalars       = S.shape[1]
    print(f"  Sequences : {X.shape}  |  Scalars: {S.shape}")
    print(f"  Vehicles  : {len(np.unique(groups))}")

    # ── Time-based train/test split (within each vehicle) ─────────────────────
    # First TRAIN_FRAC of sessions per vehicle (sorted by cycle_number) → train.
    # Last (1-TRAIN_FRAC) → test.
    # All vehicles appear in both splits so the model learns every vehicle's
    # normal pattern; test error then reflects genuine temporal change, not
    # domain shift from holding out entire unseen vehicles.
    TRAIN_FRAC = 0.80

    tr_idx_list, te_idx_list = [], []
    for veh in np.unique(groups):
        veh_mask   = np.where(groups == veh)[0]
        cycle_nums = meta.iloc[veh_mask]["cycle_number"].values
        order      = veh_mask[np.argsort(cycle_nums)]
        n_train    = max(1, int(len(order) * TRAIN_FRAC))
        tr_idx_list.extend(order[:n_train].tolist())
        te_idx_list.extend(order[n_train:].tolist())

    tr_idx = np.array(tr_idx_list)
    te_idx = np.array(te_idx_list)

    X_tr, X_te = X[tr_idx], X[te_idx]
    S_tr, S_te = S[tr_idx], S[te_idx]
    g_tr, g_te = groups[tr_idx], groups[te_idx]

    print(f"  Train: {len(tr_idx):,} sequences | {len(np.unique(g_tr))} vehicles")
    print(f"  Test : {len(te_idx):,} sequences | {len(np.unique(g_te))} vehicles")

    # ── Scale sequences ────────────────────────────────────────────────────────
    seq_scaler = StandardScaler()
    seq_scaler.fit(X_tr.reshape(-1, n_feats))

    def scale_seq(arr):
        return np.nan_to_num(
            seq_scaler.transform(arr.reshape(-1, n_feats)).reshape(arr.shape), nan=0.0)

    X_tr_s, X_te_s = scale_seq(X_tr).astype(np.float32), scale_seq(X_te).astype(np.float32)

    # ── Scale scalar features ──────────────────────────────────────────────────
    scalar_scaler = StandardScaler()
    scalar_scaler.fit(S_tr)
    S_tr_s = np.nan_to_num(scalar_scaler.transform(S_tr), nan=0.0).astype(np.float32)
    S_te_s = np.nan_to_num(scalar_scaler.transform(S_te), nan=0.0).astype(np.float32)

    # ── Build & compile model ──────────────────────────────────────────────────
    model = build_model(n_bins, n_feats, n_scalars)
    model.compile(
        optimizer=tf.keras.optimizers.AdamW(learning_rate=1e-3, weight_decay=1e-3),
        loss="mse",
    )
    model.summary()

    # ── Train ──────────────────────────────────────────────────────────────────
    print("\nTraining ...")
    cb = [
        callbacks.EarlyStopping(monitor="val_loss", patience=15,
                                restore_best_weights=True),
        callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5,
                                    patience=7, min_lr=1e-6, verbose=0),
        callbacks.ModelCheckpoint(MODEL_PATH, save_best_only=True,
                                  monitor="val_loss", verbose=0),
    ]
    model.fit(
        [X_tr_s, S_tr_s],
        X_tr_s,    # target = input (autoencoder)
        validation_data=([X_te_s, S_te_s], X_te_s),
        epochs=EPOCHS, batch_size=BATCH, callbacks=cb, verbose=2,
    )

    # ── Evaluate ───────────────────────────────────────────────────────────────
    rec_pred = model.predict([X_te_s, S_te_s], verbose=0)
    rec_error = np.mean((X_te_s - rec_pred) ** 2, axis=(1, 2))

    print(f"\n{'='*50}")
    print(f"Reconstruction error (test) — mean: {rec_error.mean():.4f}  "
          f"p95: {np.percentile(rec_error, 95):.4f}")
    print(f"{'='*50}")

    # ── Score all sequences — reconstruction error ─────────────────────────────
    X_all_s = scale_seq(X).astype(np.float32)
    S_all_s = np.nan_to_num(scalar_scaler.transform(S), nan=0.0).astype(np.float32)
    rec_all = model.predict([X_all_s, S_all_s], verbose=0)
    rec_error_all = np.mean((X_all_s - rec_all) ** 2, axis=(1, 2))

    # ── Per-vehicle anomaly scoring ────────────────────────────────────────────
    # Score each sequence against its own vehicle's train-split reconstruction
    # errors (early cycles). This gives "how unusual is this session for THIS
    # vehicle" rather than a fleet-wide comparison that penalises vehicles whose
    # operating pattern the model saw less of.
    # Vehicles with fewer than MIN_BASELINE_N train sequences fall back to the
    # global train baseline.
    from scipy import stats as sp_stats
    MIN_BASELINE_N = 5

    tr_idx_set   = set(tr_idx.tolist())
    split_labels = ["train" if i in tr_idx_set else "test" for i in range(len(meta))]

    # Build per-vehicle train-error lookup
    global_baseline = rec_error_all[tr_idx]
    veh_train_errors: dict = {}
    for i, (veh, sp) in enumerate(zip(groups, split_labels)):
        if sp == "train":
            veh_train_errors.setdefault(veh, []).append(rec_error_all[i])
    veh_train_errors = {v: np.array(errs) for v, errs in veh_train_errors.items()}

    anomaly_scores_all = np.zeros(len(meta), dtype=float)
    for i, veh in enumerate(groups):
        baseline = veh_train_errors.get(veh, global_baseline)
        if len(baseline) < MIN_BASELINE_N:
            baseline = global_baseline
        anomaly_scores_all[i] = float(sp_stats.percentileofscore(baseline, rec_error_all[i]))

    # ── Save predictions for all sequences ────────────────────────────────────
    result_df = meta.copy().reset_index(drop=True)
    result_df["reconstruction_err"] = rec_error_all
    result_df["anomaly_pct"]        = anomaly_scores_all
    result_df["is_anomaly"]         = anomaly_scores_all >= 95
    result_df["split"]              = split_labels

    result_df.to_csv(PRED_FILE, index=False)
    print(f"\nPredictions saved : {PRED_FILE}  ({len(result_df):,} sequences)")
    print(f"Model saved       : {MODEL_PATH}")
    print(f"Split breakdown   :\n{result_df['split'].value_counts().to_string()}")

    print("\nAnomaly cycles (top reconstruction errors):")
    top = result_df.sort_values("reconstruction_err", ascending=False).head(10)
    print(top[["registration_number", "reconstruction_err", "anomaly_pct",
               "split"]].to_string(index=False))
