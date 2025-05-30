#!/bin/bash

# Initialize and update submodules
git submodule update --init --recursive

echo "[INFO] Building project..."
mkdir -p build
cd build
cmake .. && make -j
cd ..

# Run the simulation
./build/Peridynamics_1D
