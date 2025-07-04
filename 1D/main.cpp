// Main.cpp

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
#include "cli.h"

// --- Main Function ---
int main(int argc, char* argv[]) {
    std::cout << "Starting 1D Peridynamics simulation!" << std::endl;

    /////////////////////////////////////////////////////////////////////////////////////////////////
    ///////////////////////////////// SIMULATION SETUP///////////////////////////////////////////////
    /////////////////////////////////////////////////////////////////////////////////////////////////

    // Parameters
    CLIOptions opts = parseArguments(argc, argv);
    int PD = 1;
    double domain_size = opts.domain_size;
    double Delta = opts.Delta;
    double L = opts.L;
    double d = opts.d * domain_size;
    int number_of_patches = opts.number_of_patches;
    int number_of_right_patches = opts.number_of_right_patches;
    double C1 = opts.C1;
    double nn = opts.nn;
    double F_prescribed = opts.F_prescribed;
    std::string Prescribed_Flag = opts.Prescribed_Flag;
    std::string DEFflag = opts.DEFflag;
    int DOFs;
    // 1. Compute corners
    std::vector<double> Corners = Compute_Corners(domain_size);

    // 2. Create mesh and patch
    std::vector<double> NLtmp = Mesh(Corners, L);
    std::vector<double> NLext = Patch(Corners, L, Delta, number_of_patches, number_of_right_patches);

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
    double FF = Compute_FF(PD, d, DEFflag);
    auto bc_result = AssignBCs(Corners, PL, FF, Prescribed_Flag, domain_size);
    PL = bc_result.first;
    auto result = AssignGlobalDOF(PL);
    PL = result.first;
    DOFs = result.second;
    std::cout << "number of DOFs                  : " << DOFs << std::endl;
    std::cout << "======================================================" << std::endl;

    /////////////////////////////////////////////////////////////////////////////////////////////////
    ///////////////////////////////// NEWTON - RAPHSON SOLVER ///////////////////////////////////////
    /////////////////////////////////////////////////////////////////////////////////////////////////
    

    // Newton-Raphson setup
    int steps = opts.steps;
    double load_step = 1.0 / steps;
    double tol = opts.tol;
    int max_try = 10;
    double LF = 0.0;
    double F_rec_patch, F_rec_rightpatch = 0; // this is the reaction force on the right patch after getting displaced.

    std::cout << "======================================================" << std::endl;
    std::cout << "Simulation Parameters:" << std::endl;
    std::cout << "Domain Size: " << domain_size << " | Lattice Length / Delta: " << L<< " | Horizon: " << Delta << std::endl;
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

        // Apply prescribed displacements or force, thats why the prescribed flag is parsed as an argument
        update_points(PL, LF, dx, Prescribed_Flag, F_prescribed); 

        int error_counter = 1;
        bool isNotAccurate = true;
        double normnull = 0.0;

        dx.setZero();
        calculate_rk(PL, C1, Delta, nn);
        // Newton-Raphson iteration
        while (isNotAccurate && error_counter <= max_try) {

            assembly(PL, DOFs, R, K, "residual");

            double residual_norm = R.norm();
            if (error_counter == 1) {
                normnull = std::max(residual_norm, 1e-10);
                std::cout << "Initial Residual Norm: " << residual_norm << std::endl;
            } else {
                double rel_norm = residual_norm / normnull;
                std::cout << "Iter " << error_counter << ": Residual Norm = " << residual_norm << ", Relative = " << rel_norm << std::endl;
                if (rel_norm < tol || residual_norm < tol) {
                    isNotAccurate = false;
                }
            }

            assembly(PL, DOFs, R, K, "stiffness");

            Eigen::SparseLU<Eigen::SparseMatrix<double>> solver;
            solver.compute(K);
            dx = solver.solve(-R);

            if(solver.info() != Eigen::Success)
            {
                std::cout << "Linear Solver failed to converge in this iteration!" << std::endl;
            }

            update_points(PL, LF, dx, "Calculated", F_prescribed);

            for(int i = 0; i < PL.size(); i++)
            {
                if((PL[i].Flag == "Right Patch"))
                {
                    F_rec_rightpatch -= (PL[i].residual);
                }
                if((PL[i].Flag == "Patch"))
                {
                    F_rec_patch += PL[i].residual;
                }
            }
            //std::cout<<"Reaction Force on the RIGHT PATCH at Load Factor    : "<< LF << " is : "<< F_rec_rightpatch <<std::endl;
            //std::cout<<"Reaction Force on the PATCH at Load Factor          : "<< LF << " is : "<< F_rec_patch <<std::endl;
            //std::cout<<"Total Reaction force = Rightpatch - Patch = " << (F_rec_rightpatch - F_rec_patch)<< std::endl<< std::endl;
 
            if(isNotAccurate == false) std::cout << "Converged after " << error_counter << " iterations." << std::endl<< std::endl;
            
            calculate_rk(PL, C1, Delta, nn);        
            error_counter++;
        }


        LF += load_step;

        // Output current state
        for (const auto& p : PL) {
            //std::cout << "Point " << p.Nr << ": x = " << p.x << ",\t displacement = " << (p.x - p.X) << std::endl;
        }
        
    }

    
    return 0;
}









/*
    // Debugging the points and their neighbours
    for (const auto& i : PL) {
        std::cout << "Nr: " << i.Nr << std::endl << "X: [";
        std::cout << i.X << ", 0, 0";
        std::cout << "]" << std::endl << "x: [" << i.x << ", 0, 0 ]" << std::endl;
        std::cout << "Volume: " << i.Vol << std::endl;
        std::cout << "BC: " << i.BCflg <<" & Flag: "<<i.Flag<<std::endl;
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
    }*/