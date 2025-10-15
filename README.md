#  Peridynamics_1d

## Introduction

This repository contains the implementation of a 1D Peridynamics simulation. Peridynamics is a non-local formulation of continuum mechanics, and this code solves problems in 1D using that theory. The simulation is designed for studying materials' behavior under stress, displacement, and other factors in a one-dimensional setting.

### What is Peridynamics?

Traditional continuum mechanics uses partial differential equations (PDEs) that can fail when dealing with discontinuities like cracks. **Peridynamics** reformulates mechanics using integral equations where each material point interacts with all neighbors within a finite distance called the **horizon (δ)**.

**Key concepts:**
- **Horizon (δ)**: The interaction radius — points interact only if they're within this distance
- **Nonlocal**: Forces on a point depend on a region of neighboring points, not just immediate neighbors
- **Bonds**: Connections between points that transmit forces based on their stretch/compression

This makes peridynamics particularly useful for simulating fracture, damage, and material failure without special crack-tracking algorithms.

### How This Code Works

The simulation follows these steps:

1. **Mesh Generation**: Creates a 1D grid of points with boundary patches
2. **Neighbor Assignment**: Each point identifies neighbors within its horizon
3. **Boundary Conditions**: Applies fixed displacement on the left and force/displacement on the right
4. **Newton-Raphson Iteration**: Solves the nonlinear equilibrium equations at each load step
   - Calculates internal forces using peridynamic bonds
   - Assembles global stiffness matrix and residual vector
   - Solves the linear system to get displacement increments
   - Updates point positions and checks convergence
5. **Output**: Logs convergence history and reaction forces

The code uses **hyperdual automatic differentiation** to compute exact derivatives for forces and stiffness, making the Newton-Raphson solver very accurate.

## Dependencies

Before running the simulation, ensure that you have the following installed:
- **Eigen** (for matrix and vector operations)
- **CMake** (≥ 3.10 for building)
- **C++ Compiler** with C++17 support (GCC or Clang)

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

Make sure the `peridynamics_1d.sh` script is executable.
This step is only needed the first time you run the code.

```bash
chmod +x peridynamics_1d.sh
```

### Cleaning up files from previous simulations

Ignore this for your first run, however for your consequent runs, to keep your repository clean and on track you can just remove the previous results by executing the following:

```bash
rm -rf log_files build csv_files/force_by_position.csv plots_force_by_position
```

### Building and Running the Simulation

To compile and run the simulation, use the following command:

```bash
./peridynamics_1d.sh
```

This script will:
- Build the project using CMake
- Run parametric studies across different horizon sizes and nonlocal parameters
- Generate log files in `log_files/` and CSV results in `csv_files/`

### Manual Build (Optional)

If you want to build manually without running the parameter sweep:

```bash
mkdir -p build
cd build
cmake ..
make -j
cd ..
```

Then run with custom parameters:

```bash
./build/Peridynamics_1D \
  --domain 100.0 \
  --delta 5.0 \
  --spacing 1.0 \
  --patches 5 \
  --rpatches 10 \
  --C1 0.5 \
  --nn 2.0 \
  --force 10.0 \
  --flag Force \
  --steps 1000 \
  --tol 1e-10 \
  --output_dir my_simulation
```

## Command Line Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `--domain` | Total domain size | 10.0 |
| `--delta` | Horizon radius (interaction distance) | 0.00301 |
| `--spacing` | Grid spacing between points | 0.001 |
| `--patches` | Number of left boundary patches | 3 |
| `--rpatches` | Number of right boundary patches | 1 |
| `--C1` | Material stiffness constant | 0.5 |
| `--nn` | Nonlocal power-law exponent | 2.0 |
| `--d` | Deformation magnitude (fraction of domain) | 0.1 |
| `--force` | Prescribed force magnitude | 1.0 |
| `--flag` | Boundary condition type: "Force" or "Displacement" | "Force" |
| `--steps` | Number of load steps | 10000 |
| `--tol` | Newton-Raphson convergence tolerance | 1e-10 |
| `--DEFflag` | Deformation flag | "EXT" |
| `--output_dir` | Output file name for logs | (required) |

## Output Files

After running simulations, you'll find:

- **`log_files/*.log`**: Detailed simulation logs including:
  - Simulation parameters
  - Load factor progression
  - Newton-Raphson convergence history
  - Final reaction forces
  - Timing information

- **`csv_files/force_by_position.csv`**: Force distribution data at each patch position
- **`csv_files/force_error.csv`**: Force balance errors for validation
- **`csv_files/timing_results.csv`**: Performance metrics

## Troubleshooting

### Common Issues

**1. Eigen not found**
```bash
# Install Eigen manually
sudo apt-get install libeigen3-dev  # Ubuntu/Debian
brew install eigen                   # macOS
```

**2. Permission errors**
```bash
chmod +x peridynamics_1d.sh
```

**3. Build fails**
- Ensure you have CMake ≥ 3.10: `cmake --version`
- Ensure C++17 support: `g++ --version` (GCC ≥ 7 or Clang ≥ 5)

**4. Convergence issues**
- Try increasing `--tol` (e.g., `1e-8` instead of `1e-10`)
- Reduce load step size by increasing `--steps`
- Check if horizon `--delta` is appropriate for your `--spacing`

## Usage

The simulation will run with default parameters. If needed, you can modify the code or configuration files for different parameters, boundary conditions, or other settings in the `peridynamics_1d.sh` accordingly.

## Reference

Javili, A., Firooz, S., McBride, A. T., & Steinmann, P. (2020). The computational framework for continuum-kinematics-inspired peridynamics. Computational Mechanics, 66(4), 795-824.

## License

This project is available under the MIT License. See LICENSE file for details.
