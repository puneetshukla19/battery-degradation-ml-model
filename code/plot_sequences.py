"""
plot_sequences.py
-----------------
Plot discharge curves from sequences.npy.

Usage:
    python plot_sequences.py                     # random session
    python plot_sequences.py --index 42          # specific row index
    python plot_sequences.py --reg VH001         # random session for a vehicle
    python plot_sequences.py --reg VH001 --n 5   # overlay 5 sessions for a vehicle
"""

import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from config import SEQ_NPY, SEQ_META

FEATURES = ["voltage", "current", "soc", "cell_spread", "temp_highest"]
UNITS    = ["V", "A", "%", "V", "°C"]
BINS     = range(20)


def load():
    arr  = np.load(SEQ_NPY)
    meta = pd.read_csv(SEQ_META)
    return arr, meta


def plot_single(arr, meta, idx):
    session = arr[idx]
    row     = meta.iloc[idx]

    fig, axes = plt.subplots(len(FEATURES), 1, figsize=(10, 12), sharex=True)
    fig.suptitle(
        f"Discharge Curve — {row['registration_number']}  "
        f"session {row['session_id']}  |  SoH {row['soh']:.1f}%  "
        f"(seq index {idx})",
        fontsize=11,
    )

    for i, (ax, feat, unit) in enumerate(zip(axes, FEATURES, UNITS)):
        ax.plot(BINS, session[:, i], marker="o", linewidth=1.8, markersize=4)
        ax.set_ylabel(f"{feat}\n({unit})", fontsize=9)
        ax.grid(True, linestyle="--", alpha=0.4)

    axes[-1].set_xlabel("Bin  (0 = session start → 19 = session end)", fontsize=10)
    plt.tight_layout()
    plt.show()


def plot_overlay(arr, meta, reg, n):
    rows = meta[meta["registration_number"] == reg]
    if rows.empty:
        print(f"No sessions found for vehicle '{reg}'")
        return

    rows = rows.sample(min(n, len(rows)), random_state=42).sort_values("cycle_number")

    fig, axes = plt.subplots(len(FEATURES), 1, figsize=(10, 12), sharex=True)
    fig.suptitle(
        f"Discharge Curves — {reg}  ({len(rows)} sessions overlaid)",
        fontsize=11,
    )

    cmap   = plt.cm.viridis
    colors = [cmap(i / max(len(rows) - 1, 1)) for i in range(len(rows))]

    for color, (_, row) in zip(colors, rows.iterrows()):
        idx     = int(row["seq_index"])
        session = arr[idx]
        label   = f"cycle {int(row['cycle_number'])}  SoH {row['soh']:.1f}%"

        for i, (ax, feat, unit) in enumerate(zip(axes, FEATURES, UNITS)):
            ax.plot(BINS, session[:, i], color=color,
                    linewidth=1.5, alpha=0.8,
                    label=label if i == 0 else None)
            ax.set_ylabel(f"{feat}\n({unit})", fontsize=9)
            ax.grid(True, linestyle="--", alpha=0.4)

    axes[0].legend(fontsize=7, loc="upper right")
    axes[-1].set_xlabel("Bin  (0 = session start → 19 = session end)", fontsize=10)
    plt.tight_layout()
    plt.show()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", type=int,  default=None, help="Row index in sequences.npy")
    parser.add_argument("--reg",   type=str,  default=None, help="Registration number")
    parser.add_argument("--n",     type=int,  default=5,    help="Sessions to overlay when --reg is used")
    args = parser.parse_args()

    arr, meta = load()
    print(f"Loaded sequences: {arr.shape}  |  meta rows: {len(meta)}")

    if args.reg and args.n > 1:
        plot_overlay(arr, meta, args.reg, args.n)
    elif args.reg:
        rows = meta[meta["registration_number"] == args.reg]
        if rows.empty:
            print(f"No sessions for '{args.reg}'")
            return
        plot_single(arr, meta, int(rows.sample(1).iloc[0]["seq_index"]))
    elif args.index is not None:
        plot_single(arr, meta, args.index)
    else:
        idx = np.random.randint(len(arr))
        print(f"Random session index: {idx}")
        plot_single(arr, meta, idx)


if __name__ == "__main__":
    main()
