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


void write_vtk_1d(const std::vector<Point>& point_list, const std::string& filename) {
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
        vtk_file << point.BCflg << "\n";
    }

    vtk_file.close();
    std::cout << "VTK file written to " << filename << std::endl;
}

// --- Main Function ---
int main() {
    std::cout << "Starting 1D Peridynamics simulation!" << std::endl;

    // Parameters
    double domain_size = 1.0;
    double Delta = 0.301;
    double L = 0.1;
    double d = 0.1;
    int number_of_patches = 2;
    int number_of_right_patches = 2;
    double C1 = 0.5;
    int DOFs = 0;
    int DOCs = 0;
    double nn = 2.0;
    std::vector<double> Forces = {0.3,0.3,0.3};
    std::vector<int> Force_nodes = {14,15,16};
    // 1. Compute corners
    std::vector<double> Corners = Compute_Corners(domain_size);

    // 2. Create mesh and patch
    std::vector<double> NLtmp = Mesh(Corners, L);
    std::vector<double> NLext = Patch(Corners, L, Delta);

    std::vector<double> NL;
    NL.insert(NL.end(), NLtmp.begin(), NLtmp.end());
    NL.insert(NL.end(), NLext.begin(), NLext.end());
    std::sort(NL.begin(), NL.end(), [](const double& a, const double& b) {
        return a < b;
    });

    // 3. Create topology
    std::vector<Point> PL = Topology(NL, L, Delta);

    // 4. Assign neighbors
    PL = AssignNgbrs(PL, L, Delta);

    // 5. Assign volumes
    PL = AssignVols(Corners, PL, L);

    // 6. Output info
    std::cout << "======================================================" << std::endl;
    std::cout << "number of nodes                 : " << NL.size() << std::endl;
    std::cout << "number of points                : " << PL.size() << std::endl;

    // 7. Compute FF - done in Points.cpp, no need to do here
    // 8. Assign boundary conditions and DOFs
    auto bc_result = AssignBCs(Corners, PL, d, Force_nodes, Forces);
    PL = bc_result.first;
    auto result = AssignGlobalDOF(PL, DOCs);
    PL = result.first;
    DOFs = result.second;
    std::cout << "number of DOCs                  : " << DOCs << std::endl;
    std::cout << "number of DOFs                  : " << DOFs << std::endl;
    std::cout << "======================================================" << std::endl;


    // Debugging the points and their neighbours
    for (const auto& i : PL) {
        std::cout << "Nr: " << i.Nr << std::endl << "X: [";
        std::cout << i.X << ", 0, 0";
        std::cout << "]" << std::endl << "x: [" << i.x << ", 0, 0 ]" << std::endl;
        std::cout << "Volume: " << i.Vol << std::endl;
        std::cout << "BC: " << i.BCflg <<" & Flag: "<<i.Flag<<std::endl;
        std::cout << "Force Mag: " << i.ForceMag <<" & Flag: "<<i.ForceFlg<<std::endl;
        std::cout << "Neighbours of " << i.Nr << " are: [";
        for (int j = 0; j < i.neighbors.size(); j++)
        {
            std::cout << "{ ";
            std::cout << i.neighbors[j] << " ";
            std::cout << "} ";
        }
        std::cout << "]";
        std::cout << "\nNumber of neighbours for point " << i.Nr << ": " << i.NI << std::endl;
        std::cout << std::endl;
    }
    //verify_displacement_scaling(PL, DOFs, DOCs, C1, Delta, nn, ForceMag);
    //exit(0); // to test independently of Newton
    // Newton-Raphson setup
    int steps = 100;
    double load_step = (1.0 / steps);
    double tol = 1e-12;
    int max_try = 30;
    double LF = 0.0;

    std::cout << "======================================================" << std::endl;
    std::cout << "Simulation Parameters:" << std::endl;
    std::cout << "Domain Size: " << domain_size << " | Lattice Length / Delta: " << L<< " | Horizon: " << Delta << std::endl;
    std::cout << "Steps: " << steps << " | Load Step: " << load_step<< " | Tolerance: " << tol << std::endl;
    std::cout << "Material constant C1: " << C1 << std::endl;
    std::cout << "======================================================" << std::endl;

    // Initialize Eigen objects
    Eigen::VectorXd R = Eigen::VectorXd::Zero(DOFs);
    Eigen::SparseMatrix<double> K;
    Eigen::SparseMatrix<double> Kuu;  // Add thisAdd commentMore actions
    Eigen::SparseMatrix<double> Kpu;
    Eigen::SparseMatrix<double> Kpp;
    Eigen::VectorXd f_reaction = Eigen::VectorXd::Zero(DOCs);
    Eigen::VectorXd dx = Eigen::VectorXd::Zero(DOFs);

    // Load stepping loop
    while (LF <= 1.0 + 1e-8)
    {
        std::cout << "\nLoad Factor: " << LF << std::endl;

        // Apply prescribed displacements
        update_points(PL, LF, dx, "Prescribed", Forces);

        int error_counter = 1;
        bool isNotAccurate = true;
        double normnull = 0.0;

        dx.setZero();
        f_reaction.setZero();
        // Newton-Raphson iteration
        while (isNotAccurate && error_counter <= max_try) {
            calculate_rk(PL, C1, Delta, nn);

            assembly(PL, DOFs, DOCs,R, K, Kuu, Kpu, Kpp,"residual");

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

            assembly(PL, DOFs, DOCs, R, K, Kuu,  Kpu,Kpp,"stiffness");

            Eigen::ConjugateGradient<Eigen::SparseMatrix<double>> solver;
            //Eigen::SparseMatrix<double> Kuu_sparse = Kuu.sparseView();  // Convert to sparse
            solver.compute(Kuu);
            dx = solver.solve(-R);

            if(solver.info() != Eigen::Success)
            {
                std::cout << "Linear Solver failed to converge in this iteration!" << std::endl;
            }

            update_points(PL, LF, dx, "Displacement", Forces);

            Eigen::VectorXd u_free = dx.head(DOFs);
            Eigen::VectorXd u_prescribed = Eigen::VectorXd::Zero(DOCs);
            for (const auto& p : PL) {
                if (p.BCflg == 0 && p.DOC > 0) {
                    u_prescribed(p.DOC - 1) = p.BCval * LF;
                }
            }

            // Calculate reactions before updating positions
            f_reaction = -(Kpu * u_free + Kpp * u_prescribed);

            error_counter++;
            
        }

        //std::cin.get();
        LF += load_step;
        int idx = 0;
        for (const auto& p : PL) {
            std::cout << "Point " << p.Nr << ": x = " << p.x << ", displacement = " << (p.x - p.X) << std::endl;
            if (p.BCflg == 0 && p.DOC > 0){
                std::cout << "Point " << p.Nr << " (X = " << p.X << "): "
                      << "Reaction = " << f_reaction(idx) << std::endl;
                idx++;
            }

        }
    }
    return 0;
}