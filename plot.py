#!/usr/bin/env python3
# Combined plotter: 
# 1. x = X (-1..-H), y = Diff (Right - Left at X) for each H
# 2. x = Horizon Size (H), y = Force Difference Error (%) for all H and NN values

import os
import pandas as pd
import matplotlib.pyplot as plt

CSV_PATH = "csv_files/force_by_position.csv"
ERROR_CSV_PATH = "csv_files/force_error.csv"
OUT_DIR = "plots_force_by_position"
ERROR_OUT_DIR = "plots_force_error"

def plot_force_differences():
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

def plot_force_errors():
    if not os.path.isfile(ERROR_CSV_PATH):
        print(f"Missing {ERROR_CSV_PATH}")
        return

    df = pd.read_csv(ERROR_CSV_PATH, names=["H", "NN", "Error"])
    
    # Create output directory
    os.makedirs(ERROR_OUT_DIR, exist_ok=True)
    
    # Convert to appropriate data types
    df["H"] = df["H"].astype(int)
    df["NN"] = df["NN"].astype(float)
    
    # Create a single combined plot with all horizon sizes
    plt.figure(figsize=(10, 6))
    
    # Plot each NN value
    for nn in sorted(df["NN"].unique()):
        nn_data = df[df["NN"] == nn].sort_values("H")
        plt.plot(nn_data["H"], nn_data["Error"], 
                marker="o", label=f"nn={nn:g}", linewidth=2)
    
    plt.xlabel("Horizon Size (H)")
    plt.ylabel("Force Difference Error (%)")
    plt.title("Force Error vs Horizon Size for Different Power Laws")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    
    # Save the single combined plot
    combined_path = os.path.join(ERROR_OUT_DIR, "force_error_combined.png")
    plt.savefig(combined_path, dpi=180)
    plt.close()
    print(f"Saved {combined_path}")

def main():
    print("Generating force difference plots...")
    plot_force_differences()
    
    print("\nGenerating force error plot...")
    plot_force_errors()
    
    print("\nAll plots generated successfully!")

if __name__ == "__main__":
    main()