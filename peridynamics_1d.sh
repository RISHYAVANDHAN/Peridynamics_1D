#!/bin/bash

# Initialize and update submodules
git submodule update --init --recursive

mkdir -p log_files
mkdir -p csv_files

# Parameters
Domain=100.0
DeformationMagnitude=$(echo "0.45 * $Domain" | bc -l)
Force=10.0
Points=(1)
Prescribed=("Force")
Horizon=(1 2 3 4 5 6 7 8 9 10)
NN=(0.25 1.0 2.0 4.0)

echo "[INFO] Building project..."
mkdir -p build
cd build
cmake .. && make -j
cd ..

# Loop over combinations
for N in "${Points[@]}"; do 
  for P in "${Prescribed[@]}"; do
    for H in "${Horizon[@]}"; do
      for n in "${NN[@]}"; do
        
        # Decide rpatches based on Horizon size with a fixed patches, the patches are always the horizon size;
        # since we are spacing them for 100 points in domain of 100, we get horizon size as a natural number and hence can be used for the number of left patches 
        if [[ $H -eq 1 || $H -eq 2 ]]; then
          rpatches=$H
        elif [[ $H -eq 3 || $H -eq 4 ]]; then
          rpatches=5
        elif [[ $H -ge 5 && $H -le 8 ]]; then
          rpatches=10
        elif [[ $H -eq 9 || $H -eq 10 ]]; then
          rpatches=15
        else
          rpatches=50
        fi
        

        ./build/Peridynamics_1D \
          --domain $Domain \
          --delta $H \
          --spacing $N \
          --patches $H \
          --rpatches $rpatches \
          --C1 0.5 \
          --nn $n \
          --d $DeformationMagnitude \
          --force $Force \
          --flag $P \
          --steps 100 \
          --tol 1e-10 \
          --DEFflag EXT \
          --output_dir "Simulation_Force_10N_Domain=100_Horizon=${H}_NN=${n}"

      done
    done
  done
done
