cmake_minimum_required(VERSION 3.14)
project(Peridynamics_1D)

set(CMAKE_CXX_STANDARD 17)

# Add subdirectory for Eigen as it has been added as a submodule
add_subdirectory(eigen)

add_executable(1D
    main.cpp
    Points.cpp
    Points.h
)

target_link_libraries(Peridynamics_1D PRIVATE Eigen3::Eigen)
