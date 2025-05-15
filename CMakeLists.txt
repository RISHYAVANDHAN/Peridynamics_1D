cmake_minimum_required(VERSION 3.14)
project(Peridynamics_1D)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Simply include Eigen headers
find_package(Eigen3 REQUIRED)

# Fix paths to source files
add_executable(Peridynamics_1D
        main.cpp
        Points.cpp
        Points.h
)

target_link_libraries(Peridynamics_1D PRIVATE Eigen3::Eigen)