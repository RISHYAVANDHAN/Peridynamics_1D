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
    std::cout << "\nStarting 1D Peridynamics simulation!" << std::endl;

    // Parameters
    double domain_size = 1.0;
    double delta = 0.301;
    double Delta = 0.1;
    double d = 0.1;
    int number_of_patches = 3;
    int number_of_right_patches = 1;
    double C1 = 0.01;
    int DOFs = 0;
    int DOCs = 0;
    double nn = 2.0;

    // Create mesh
    std::vector<Points> points = mesh(domain_size, number_of_patches, Delta, number_of_right_patches, DOFs, DOCs, d);
    neighbour_list(points, delta);

    // Debug mesh information
    std::cout << "\n=== Mesh & neighbour Information ===" << std::endl;
    std::cout << "Mesh contains " << points.size() << " points with " << DOFs << " DOFs\n";
    for (const auto& point : points) {
        std::cout << "Point " << point.Nr << ": X = " << point.X << " , Flag = " << point.Flag 
                  << " , BCflag = " << point.BCflag << " , DOF = " << point.DOF 
                  << " , Neighbors = " << point.n1 << std::endl;
        std::cout << "Neighbours of " << point.Nr << " are: [";
        for (auto& n : point.neighbours)
        {
            std::cout << "{ ";
            std::cout << n << " ";
            std::cout << "} ";
        }
        std::cout << "]";
        std::cout << "\nNumber of neighbours for point " << point.Nr << ": " << point.n1 << std::endl;
        std::cout << std::endl;
    }
    
    // Write initial mesh to VTK
    //write_vtk_1d(points, "C:/Users/srini/Downloads/FAU/Semwise Course/Programming Project/peridynamics 1D vtk/initial.vtk");

    // Newton-Raphson setup
    int steps = 10;
    double load_step = (1.0 / steps);
    double tol = 1e-10;
    int min_try = 0;
    int max_try = 10;
    double LF = 0.0;
    int counter = 0;

    std::cout << "======================================================" << std::endl;
    std::cout << "Simulation Parameters:" << std::endl;
    std::cout << "Domain Size: " << domain_size << " | Delta: " << Delta<< " | Horizon: " << delta << std::endl;
    std::cout << "Steps: " << steps << " | Load Step: " << load_step<< " | Tolerance: " << tol << std::endl;
    std::cout << "Material constant C1: " << C1 << std::endl;
    std::cout << "Material Power Law NN: " << nn << std::endl;
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
            calculate_rk(points, C1, delta, nn);

            assembly(points, DOFs, R, K, "residual");

            double residual_norm = R.norm();
            std::cout << "Residual Norm: " << residual_norm << std::endl;
            if (error_counter == 1) {
                normnull = std::max(residual_norm, 1e-16);
                std::cout << "Iter 1 - Initial Residual Norm: " << residual_norm << std::endl;
            } else {
                double rel_norm = residual_norm / normnull;
                std::cout << "Iter " << error_counter << ": Residual Norm = " << residual_norm
                          << ", Relative = " << rel_norm << std::endl;

                if (rel_norm < tol || residual_norm < tol) {
                    isNotAccurate = false;
                    std::cout << "Converged after " << error_counter << " iterations." << std::endl;
                }
                if (error_counter >= max_try) {
                    std::cout << "Not converged after " << error_counter << " iterations." << std::endl;
                }
            }

            assembly(points, DOFs, R, K, "stiffness");

            Eigen::ConjugateGradient<Eigen::SparseMatrix<double>> solver;
            solver.compute(K);
            Eigen::VectorXd step = solver.solve(R);
            std::cout << "dx norm = " << step.norm() << std::endl;
            dx += step;

            if(solver.info() != Eigen::Success)
            {
                std::cout << "Solver failed to converge!" << std::endl;
            }

            update_points(points, LF, dx, "Displacement");
            
            error_counter += 1;

        }
        //std::ostringstream load_filename;
        //load_filename << "C:/Users/srini/Downloads/FAU/Semwise Course/Programming Project/peridynamics 1D vtk/load_" << std::fixed << std::setprecision(2) << LF << ".vtk";
        //write_vtk_1d(points, load_filename.str());

        counter += 1;

        LF += load_step;

        // Output current state
        for (const auto& p : points) {
            //std::cout << "Point " << p.Nr << ": x = " << p.x << ", displacement = " << (p.x - p.X) << std::endl;
        }
    }

    //write_vtk_1d(points, "C:/Users/srini/Downloads/FAU/Semwise Course/Programming Project/peridynamics 1D vtk/final.vtk");

    return 0;
}