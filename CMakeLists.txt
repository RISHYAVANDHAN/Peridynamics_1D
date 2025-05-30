cmake_minimum_required(VERSION 3.14)
project(Peridynamics_1D)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Simply include Eigen headers
include_directories(eigen)

# Fix paths to source files
add_executable(Peridynamics_1D
    1D/hyperdual.h
    1D/main.cpp
    1D/Points.cpp
    1D/Points.h
)