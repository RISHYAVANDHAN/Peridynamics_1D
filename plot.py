#!/usr/bin/env python3
# One-file plotter: x = X (-1..-H), y = Diff (Right - Left at X)
# Input CSV: csv_files/force_by_position.csv with columns: H,NN,X,Diff

import os
import pandas as pd
import matplotlib.pyplot as plt

CSV_PATH = "csv_files/force_by_position.csv"
OUT_DIR  = "plots_force_by_position"

def main():
    if not os.path.isfile(CSV_PATH):
        print(f"Missing {CSV_PATH}")
        return

    df = pd.read_csv(CSV_PATH)
    # basic sanity
    for col in ["H","NN","X","Diff"]:
        if col not in df.columns:
            raise ValueError(f"CSV missing column: {col}")

    os.makedirs(OUT_DIR, exist_ok=True)

    # Ensure numeric types and sort for consistent plotting
    df["H"] = df["H"].astype(float)
    df["NN"] = df["NN"].astype(float)
    df["X"] = df["X"].astype(int)

    for H, dH in df.groupby("H"):
        plt.figure()
        for NN, dHN in dH.groupby("NN"):
            dHN = dHN.sort_values("X")
            plt.plot(dHN["X"], dHN["Diff"], marker="o", label=f"nn={NN:g}")
        plt.xlabel("Position X (−1 … −H)")
        plt.ylabel("Force difference (RightPatch − LeftPatch)")
        plt.title(f"Force vs Position — Horizon H={int(H)}")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        out = os.path.join(OUT_DIR, f"H{int(H)}.png")
        plt.savefig(out, dpi=180)
        plt.close()
        print(f"Saved {out}")

if __name__ == "__main__":
    main()
