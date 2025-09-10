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
#include <chrono>

#include "Points.h"
#include "cli.h"
#include "logger.h"


// --- Main Function --- //
int main(int argc, char* argv[]) {
    auto total_start = std::chrono::high_resolution_clock::now();
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
    std::string file_name = opts.output_dir;
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

    ////////////////////////////////////////////////////////////////////////////////////////////////
    ///////////////////////////////////    LOGGING THE INFO     ////////////////////////////////////
    ////////////////////////////////////////////////////////////////////////////////////////////////

    std::cout << "[LOG] Writing to: log_files/" << file_name << ".log" << std::endl;
    Logger logger(file_name);
    logger.writeHeader(file_name);
    std::string timing_csv = "csv_files/timing_results.csv";
    bool file_exists = std::filesystem::exists(timing_csv);
    std::ofstream csv_file(timing_csv, std::ios::app);
    if (!file_exists) {
        // Write header only if file doesn't exist
        csv_file << "spacing,number_of_points,simulation_time_sec,total_time_sec,implementation\n";
    }

    /////////////////////////////////////////////////////////////////////////////////////////////////
    ////////////////////////////////// NEWTON - RAPHSON SOLVER //////////////////////////////////////
    /////////////////////////////////////////////////////////////////////////////////////////////////
    

    // Newton-Raphson setup
    int steps = opts.steps;
    double load_step = 1.0 / steps;
    double tol = opts.tol;
    int max_try = 100;
    double LF = 0.0;
    double F_rec_patch, F_rec_rightpatch = 0; // this is the reaction force on the right patch after getting displaced.

    logger.writeParameters(domain_size, L, Delta, PL.size(), steps, C1, nn, Prescribed_Flag, F_prescribed, d, number_of_patches, number_of_right_patches);

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

    // --- Start simulation timer --- //
    auto sim_start = std::chrono::high_resolution_clock::now();

    // Load stepping loop
    while (LF <= 1.0 + 1e-8) {
        std::cout << "\nLoad Factor: " << LF << std::endl;
        logger.writeLoadFactor(LF);
        
        // Apply prescribed displacements or force, thats why the prescribed flag is parsed as an argument
        update_points(PL, LF, dx, Prescribed_Flag, F_prescribed, number_of_right_patches); 

        int error_counter = 1;
        bool isNotAccurate = true;
        double normnull = 0.0;

        dx.setZero();
        // Newton-Raphson iteration
        while (isNotAccurate && error_counter <= max_try) {
            
            calculate_rk(PL, C1, Delta, nn);
            assembly(PL, DOFs, R, K, "residual");

            double residual_norm = R.norm();
            double rel_norm;
            if (error_counter == 1) {
                normnull = std::max(residual_norm, 1e-10);
                std::cout << "Initial Residual Norm = " << residual_norm << std::endl;
            } else {
                rel_norm = residual_norm / normnull;
                std::cout << "Iter " << error_counter << ": Residual Norm = " << residual_norm << ", Relative = " << rel_norm << std::endl;
                if ((rel_norm - tol) < 1e-12 || (residual_norm - tol) < 1e-12) {
                    isNotAccurate = false;
                }
            }
            logger.writeConvergence(error_counter, residual_norm, rel_norm);
            assembly(PL, DOFs, R, K, "stiffness");

            // main.cpp - in Newton-Raphson iteration
            /*
            if(nn == 1.0) {
                Eigen::MatrixXd A = Eigen::MatrixXd(K);
                Eigen::FullPivLU<Eigen::MatrixXd> solver;
                solver.compute(A);
                if(!solver.isInvertible()) {
                    std::cout << "Linear solver failed to compute!" << std::endl;
                    break;
                }
                dx = solver.solve(-R);
            } else {
                Eigen::SparseLU<Eigen::SparseMatrix<double>> solver;
                solver.compute(K);
                if(solver.info() != Eigen::Success) {
                    std::cout << "Nonlinear solver failed to compute!" << std::endl;
                    break;
                }
                dx = solver.solve(-R);
                if(solver.info() != Eigen::Success) {
                    std::cout << "Solver failed to converge in this iteration!" << std::endl;
                }
            }
            */
            Eigen::SparseLU<Eigen::SparseMatrix<double>> solver;
            solver.compute(K);
            if(solver.info() != Eigen::Success) {
                std::cout << "Nonlinear solver failed to compute!" << std::endl;
                break;
            }
            dx = solver.solve(-R);  
                  
            update_points(PL, LF, dx, "Calculated", F_prescribed, number_of_right_patches);

            F_rec_patch = 0.0;
            F_rec_rightpatch = 0.0;

            for (int i = 0; i < PL.size(); i++) {
                if (PL[i].Flag == "Right Patch" && Prescribed_Flag == "Force")
                    F_rec_rightpatch += PL[i].F_ext;
                if (PL[i].Flag == "Right Patch" && Prescribed_Flag == "Displacement")
                    F_rec_rightpatch += PL[i].residual;
                if ((PL[i].Flag == "Patch"))
                    F_rec_patch += PL[i].residual;
            }

            if (!isNotAccurate && LF >= 1.0 - 1e-12) {
                const int H = number_of_patches;

                std::vector<double> left_residuals;
                for (int i = 0; i < PL.size(); ++i)
                    if (PL[i].Flag == "Patch")
                        left_residuals.push_back(PL[i].residual);

                logger.writePatchForces(H, nn, left_residuals, F_rec_rightpatch);

                // append also to CSV
                bool file_exists = std::filesystem::exists("csv_files/force_by_position.csv");
                std::ofstream ofs("csv_files/force_by_position.csv", std::ios::app);
                if (!file_exists) {
                    ofs << "H,NN,X,Diff\n";
                }
                for (int k = 0; k < (int)left_residuals.size(); ++k) {
                    int Xpos = -(k+1);
                    ofs << H << "," << nn << "," << Xpos << "," << (F_rec_rightpatch - left_residuals[k]) << "\n";
                }
            }

            if(isNotAccurate == false) {
                std::cout << "Converged after " << error_counter << " iterations." << std::endl<< std::endl;
                logger.writeConverged(error_counter);
            }
            
            error_counter++;
        }        
        LF += load_step;   
    }
    std::cout<<"Applied / Reaction Force on the RIGHT PATCH is : "<< F_rec_rightpatch  <<std::endl;
    std::cout<<"Reaction Force on the PATCH is : "<< F_rec_patch <<std::endl;
    std::cout<<"Total Reaction force = Rightpatch - Patch = " << (F_rec_rightpatch - F_rec_patch)<< std::endl<< std::endl;
    logger.writeReactoinForce(LF, F_rec_rightpatch, F_rec_patch);
    
    // Output final state
    /*for (const auto& p : PL) {
        std::cout << "Point " << p.Nr << ": x = " << p.x << ",\t displacement = " << (p.x - p.X) << std::endl;
    }*/
    
    // --- End simulation timer --- //
    auto sim_end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> sim_duration = sim_end - sim_start;

    // --- End total timer --- //
    auto total_end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> total_duration = total_end - total_start;

    std::cout << "\nSimulation time: " << sim_duration.count() << " seconds" << std::endl;
    std::cout << "Total program time: " << total_duration.count() << " seconds" << std::endl;
    logger.writeTiming(sim_duration.count(), total_duration.count());

    logger.close();
    return 0;
}