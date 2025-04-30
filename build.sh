#!/bin/bash

# Initialize and update submodules
git submodule update --init --recursive

# Create build directory
mkdir -p build
cd build

# Run CMake to configure the project
cmake ..

# Build the project
make
