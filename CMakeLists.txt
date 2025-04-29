cmake_minimum_required(VERSION 3.14)
project(Peridynamics_1D)

set(CMAKE_CXX_STANDARD 17)

include(FetchContent)

FetchContent_Declare(
  eigen
  GIT_REPOSITORY https://gitlab.com/libeigen/eigen.git
  GIT_TAG 3.4.0  # Use the desired Eigen version
)

FetchContent_MakeAvailable(eigen)

add_executable(Peridynamics_1D
    main.cpp
    Points.cpp
    Points.h
)

target_link_libraries(Peridynamics_1D PRIVATE Eigen3::Eigen)
