import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) if "__file__" in dir() else os.getcwd())

import numpy as np
import pandas as pd
from config import SEQ_NPY, SEQ_META, SEQ_FEATURES

# ── Sequences ─────────────────────────────────────────────────────────────────
arr = np.load(SEQ_NPY)
print(f"Shape : {arr.shape}")          # (N_sequences, NUM_BINS, N_features)
print(f"dtype : {arr.dtype}")
print(f"min   : {arr.min():.3f}  max: {arr.max():.3f}  NaNs: {np.isnan(arr).sum()}")

# First sequence — 20 bins × 5 features
print("\nFirst sequence (bins x features):")
print(pd.DataFrame(arr[0], columns=SEQ_FEATURES).round(2).to_string())

# ── Metadata ──────────────────────────────────────────────────────────────────
meta = pd.read_csv(SEQ_META)
print(f"\nMeta shape : {meta.shape}")
print(meta.dtypes)
print("\nSample rows:")
print(meta.head(10).to_string(index=False))

print(f"\nSoH distribution:\n{meta['soh'].value_counts().sort_index()}")
print(f"\nVehicles : {meta['registration_number'].nunique()}")
print(f"\nSequences per vehicle:\n{meta['registration_number'].value_counts()}")
