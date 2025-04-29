
#  Peridynamics_1d

## Introduction

This repository contains the implementation of a 1D Peridynamics simulation. Peridynamics is a non-local formulation of continuum mechanics, and this code solves problems in 1D using that theory. The simulation is designed for studying materials' behavior under stress, displacement, and other factors in a one-dimensional setting.

## Dependencies

Before running the simulation, ensure that you have the following installed:
- **Eigen** (for matrix and vector operations)

### Eigen Installation
- If you do not already have Eigen, it will be installed automatically when running the build script. You can also modify the `CMakeLists.txt` to point to your pre-installed Eigen directory.

## Setup (Linux / WSL)

### Cloning the Repository

First, clone the repository to your local machine:

```bash
git clone https://github.com/RISHYAVANDHAN/Peridynamics_1D.git
cd Peridynamics_1D
```

### Permissions

Make sure the `build.sh` and `peridynamics_1d.sh` scripts are executable:

```bash
chmod +x build.sh
chmod +x peridynamics_1d.sh
```

This step is only needed the first time you run the code.

### Building and Running the Simulation

To compile and run the simulation, use the following commands:

```bash
./build.sh
./peridynamics_1d.sh
```

### Troubleshooting

1. **Eigen not found**: If Eigen is not installed, the build script will attempt to install it. Ensure you have internet access if this is necessary.
2. **Permissions errors**: If you encounter permission issues, ensure the necessary scripts are executable (`chmod +x`).

## Usage

The simulation will run with default parameters. If needed, you can modify the code or configuration files for different parameters, boundary conditions, or other settings.

## Additional Resources

For more details on Peridynamics and the theory behind this code, please refer to relevant scientific literature on the subject.
