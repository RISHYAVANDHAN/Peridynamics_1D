#!/bin/bash

# Initialize and update submodules
git submodule update --init --recursive

echo "[INFO] Building project..."
mkdir -p build
cd build
cmake .. && make -j
cd ..

# Run the simulation with hardcoded parameters
./build/Peridynamics_1D \
  --domain 10.0 \
  --delta 0.00301 \
  --spacing 0.001 \
  --patches 3 \
  --rpatches 1 \
  --C1 0.5 \
  --nn 2.0 \
  --d 0.1 \
  --force 1.0 \
  --flag Force \
  --steps 10000 \
  --tol 1e-10 \
  --DEFflag EXT
