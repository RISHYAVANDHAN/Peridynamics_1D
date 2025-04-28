#include <iostream>
#include <vector>
#include <cmath>
#include <iomanip>
#include <fstream>
#include <filesystem>
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <string>
#include <algorithm>
#include "Points.h"


void write_vtk_1d(const std::vector<Points>& point_list, const std::string& filename) {
    std::ofstream vtk_file(filename);
    if (!vtk_file.is_open()) {
        std::cerr << "Failed to open VTK file for writing: " << filename << std::endl;
        return;
    }

    vtk_file << "# vtk DataFile Version 4.2\n";
    vtk_file << "1D Peridynamics Output\n";
    vtk_file << "ASCII\n";
    vtk_file << "DATASET POLYDATA\n";
    vtk_file << "POINTS " << point_list.size() << " float\n";

    for (const auto& point : point_list) {
        vtk_file << std::fixed << std::setprecision(6);
        vtk_file << point.x << " 0.0 0.0\n";
    }

    // Add lines connecting adjacent points to emphasize horizontal layout
    vtk_file << "LINES " << (point_list.size()-1) << " " << (point_list.size()-1)*3 << "\n";
    for (size_t i = 0; i < point_list.size()-1; i++) {
        vtk_file << "2 " << i << " " << (i+1) << "\n";
    }

    vtk_file << "POINT_DATA " << point_list.size() << "\n";

    vtk_file << "SCALARS BC int 1\n";
    vtk_file << "LOOKUP_TABLE default\n";
    for (const auto& point : point_list) {
        vtk_file << point.BCflag << "\n";
    }

    vtk_file.close();
    std::cout << "VTK file written to " << filename << std::endl;
}

// --- Main Function ---
int main() {
    std::cout << "Starting 1D Peridynamics simulation!" << std::endl;

    // Parameters
    double domain_size = 1.0;
    double delta = 0.301;
    double Delta = 0.1;
    double d = 1.0;
    int number_of_patches = 3;
    int number_of_right_patches = 1;
    double C1 = 0.5;
    int DOFs = 0;
    int DOCs = 0;

    // Create mesh
    std::vector<Points> points = mesh(domain_size, number_of_patches, Delta, number_of_right_patches, DOFs, DOCs, d);
    std::cout << "Mesh contains " << points.size() << " points with " << DOFs << " DOFs\n";
    neighbour_list(points, delta);

    // Debugging the points and their neighbours
    for (const auto& i : points) {
        std::cout << "Nr: " << i.Nr << std::endl << "X: [";
        std::cout << i.X << ", 0, 0";
        std::cout << "]" << std::endl << "x: [" << i.x << ", 0, 0 ]" << std::endl;
        std::cout << "Volume: " << i.volume << std::endl;
        std::cout << "BC: " << i.BCflag << std::endl << "Flag: " << i.Flag << std::endl;
        std::cout << "Neighbours of " << i.Nr << " are: [";
        for (const auto& n : i.neighbours)
        {
            std::cout << "{ ";
            std::cout << n << " ";
            std::cout << "} ";
        }
        std::cout << "]";
        std::cout << "\nNumber of neighbours for point " << i.Nr << ": " << i.n1 << std::endl;
        std::cout << std::endl;
    }

    // Write initial mesh to VTK
    write_vtk_1d(points, "C:/Users/srini/Downloads/FAU/Semwise Course/Programming Project/peridynamics 1D vtk/initial.vtk");

    // Newton-Raphson setup
    int steps = 100;
    double load_step = (1.0 / steps);
    double tol = 1e-6;
    int max_try = 30;
    double LF = 0.0;

    std::cout << "======================================================" << std::endl;
    std::cout << "Simulation Parameters:" << std::endl;
    std::cout << "Domain Size: " << domain_size << " | Delta: " << Delta<< " | Horizon: " << delta << std::endl;
    std::cout << "Steps: " << steps << " | Load Step: " << load_step<< " | Tolerance: " << tol << std::endl;
    std::cout << "Material constant C1: " << C1 << std::endl;
    std::cout << "======================================================" << std::endl;

    // Initialize Eigen objects
    Eigen::VectorXd R = Eigen::VectorXd::Zero(DOFs);
    Eigen::SparseMatrix<double> K;
    Eigen::VectorXd dx = Eigen::VectorXd::Zero(DOFs);

    // Load stepping loop
    while (LF <= 1.0 + 1e-8) {
        std::cout << "\nLoad Factor: " << LF << std::endl;

        // Apply prescribed displacements
        update_points(points, LF, dx, "Prescribed");

        int error_counter = 1;
        bool isNotAccurate = true;
        double normnull = 0.0;

        dx.setZero();

        // Newton-Raphson iteration
        while (isNotAccurate && error_counter <= max_try) {
            calculate_rk(points, C1, delta);

            assembly(points, DOFs, R, K, "residual");

            double residual_norm = R.norm();
            if (error_counter == 1) {
                normnull = std::max(residual_norm, 1e-10);
                std::cout << "Initial Residual Norm: " << residual_norm << std::endl;
            } else {
                double rel_norm = residual_norm / normnull;
                std::cout << "Iter " << error_counter << ": Residual Norm = " << residual_norm
                          << ", Relative = " << rel_norm << std::endl;

                if (rel_norm < tol || residual_norm < tol) {
                    isNotAccurate = false;
                    std::cout << "Converged after " << error_counter << " iterations." << std::endl;
                }
            }

            assembly(points, DOFs, R, K, "stiffness");

            Eigen::FullPivLU<Eigen::MatrixXd> solver(K);
            dx += solver.solve(-R);

            update_points(points, LF, dx, "Displacement");
            error_counter++;
        }
        std::ostringstream load_filename;
        load_filename << "C:/Users/srini/Downloads/FAU/Semwise Course/Programming Project/peridynamics 1D vtk/load_" << std::fixed << std::setprecision(2) << LF << ".vtk";
        write_vtk_1d(points, load_filename.str());

        LF += load_step;

        // Output current state
        for (const auto& p : points) {
            std::cout << "Point " << p.Nr << ": x = " << p.x << ", displacement = " << (p.x - p.X) << std::endl;
        }
    }

    write_vtk_1d(points, "C:/Users/srini/Downloads/FAU/Semwise Course/Programming Project/peridynamics 1D vtk/final.vtk");

    return 0;
}
