#!/bin/bash

# Initialize and update submodules
git submodule update --init --recursive

mkdir -p log_files
mkdir -p csv_files

# Parameters
Domain=10.0
DeformationMagnitude=$(echo "0.45 * $Domain" | bc -l)
Force=1.0
Points=(0.1 0.01 0.001 0.0001)
Prescribed=("Force" "Displacement")
Horizon=$(echo "3.01 * $Points" | bc -l)

echo "[INFO] Building project..."
mkdir -p build
cd build
cmake .. && make -j
cd ..

# Loop over combinations
for N in "${Points[@]}"; do 
  for P in "${Prescribed[@]}"; do
    ./build/Peridynamics_1D \
      --domain $Domain \
      --delta $Horizon \
      --spacing $N \
      --patches 3 \
      --rpatches 1 \
      --C1 0.5 \
      --nn 2.0 \
      --d $DeformationMagnitude \
      --force $Force \
      --flag $P \
      --steps 100 \
      --tol 1e-10 \
      --DEFflag EXT \
      --output_dir "Simulation_${P}_${N}"
  done
done
