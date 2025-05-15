#include <iostream>
#include <vector>
#include <cmath>
#include <chrono>
#include <iomanip>
#include <fstream>
#include <filesystem>
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <string>
#include <algorithm>
#include <filesystem>
#include "Points.h"

void ensure_directories(const std::string& path)
{
    namespace fs = std::filesystem;
    fs::path p(path);
    fs::path dir = p.parent_path();

    if (!dir.empty() && !fs::exists(dir)) {
        fs::create_directories(dir);
    }
}

void write_vtk_1d(const std::vector<Points>& point_list, const std::string& filename) {

    ensure_directories(filename);

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
    auto start_time = std::chrono::high_resolution_clock::now();
    // Parameters
    double domain_size = 1.0;
    double delta = 0.301;
    double Delta = 0.1;
    double d = 1.0;
    int number_of_patches = 3;
    int number_of_right_patches = 3;
    double C1 = 0.5;
    int DOFs = 0;
    int DOCs = 0;

    // specify node numbers and forces
    std::vector<int> force_nodes = {14}; // Applying force to node 2
    std::vector<double> forces = {0.34902}; // Force magnitude

    // Create mesh
    std::vector<Points> points = mesh(domain_size, number_of_patches, Delta, number_of_right_patches, DOFs, DOCs, d, force_nodes, forces);
    std::cout << "Mesh contains " << points.size() << " points with " << DOFs << " DOFs\n";
    neighbour_list(points, delta);

    // Debugging the points and their neighbours
    for (const auto& i : points) {
        std::cout << "Nr: " << i.Nr << std::endl << "X: [";
        std::cout << i.X << ", 0, 0";
        std::cout << "]" << std::endl << "x: [" << i.x << ", 0, 0 ]" << std::endl;
        std::cout << "Volume: " << i.volume << std::endl;
        std::cout << "BC: " << i.BCflag << std::endl << "Flag: " << i.Flag << std::endl;
        std::cout << "Force Flag: " << i.Forceflag << std::endl;
        if (i.Forceflag) {
            std::cout << "Applied Force: [" << i.ForceMag << ", 0, 0]" << std::endl;
        }
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
    write_vtk_1d(points, "peridynamics 1D vtk/initial.vtk");

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
    Eigen::MatrixXd Kuu;  // Add this
    Eigen::MatrixXd Kpu;
    Eigen::MatrixXd Kpp;
    Eigen::VectorXd f_reaction;
    Eigen::VectorXd dx = Eigen::VectorXd::Zero(DOFs);

    // Load stepping loop
    while (LF <= 1.0 + 1e-8) {
        std::cout << "\nLoad Factor: " << LF << std::endl;

        // Apply prescribed displacements and forces
        update_points(points, LF, dx, "Prescribed");

        int error_counter = 1;
        bool isNotAccurate = true;
        double normnull = 0.0;

        dx.setZero();
		f_reaction.setZero();

        // Newton-Raphson iteration
        while (isNotAccurate && error_counter <= max_try) {
			// Calculate internal forces and account for external forces
            calculate_rk(points, C1, delta);

        	// Assemble residual (F_int - F_ext already done in calculate_rk)
        	assembly(points, DOFs, DOCs, R, K, Kuu, Kpu, Kpp, "residual");

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

        	// Assemble stiffness matrix
        	assembly(points, DOFs, DOCs, R, K, Kuu, Kpu, Kpp, "stiffness");

        	// Solve system
        	Eigen::FullPivLU<Eigen::MatrixXd> solver(Kuu);
        	dx += solver.solve(-R);

        	// Update displacements
        	update_points(points, LF, dx, "Displacement");

        	Eigen::VectorXd u_free = dx.head(DOFs);
        	Eigen::VectorXd u_prescribed = Eigen::VectorXd::Zero(DOCs);
        	for (const auto& p : points) {
            	if (p.BCflag == 0) {
                	u_prescribed(p.DOC-1) = p.BCval * LF;
            	}
        	}

        	// Calculate reactions before updating positions
        	f_reaction = -(Kpu * u_free + Kpp * u_prescribed);

        	error_counter++;
    	}

    	// Output results
    	std::ostringstream load_filename;
    	load_filename << "peridynamics 1D vtk/load_" << std::fixed << std::setprecision(2) << LF << ".vtk";
    	write_vtk_1d(points, load_filename.str());

    	// Output current state
    	for (const auto& p : points) {
        	std::cout << "Point " << p.Nr << ": x = " << p.x
                  << ", displacement = " << (p.x - p.X);
        	if (p.Forceflag == 1) {
            	std::cout << ", applied force = " << p.Forceval[0];
        	}
        	std::cout << std::endl;
    	}
    	std::cout << "Reaction forces:\n" << f_reaction << std::endl;

    	LF += load_step;

        // Output current state
        for (const auto& p : points) {
            std::cout << "Point " << p.Nr << ": x = " << p.x << ", displacement = " << (p.x - p.X) << std::endl;
        }
        std::cout << "force reaction for the current step" << f_reaction << std::endl;
    }

    write_vtk_1d(points, "peridynamics 1D vtk/final.vtk");
    auto end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> diff = end_time - start_time;

    std::cout << "======================================================" << std::endl;
    std::cout << "Total computation time: " << diff.count() << " seconds" << std::endl;
    std::cout << "======================================================" << std::endl;

    return 0;
}
