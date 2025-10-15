// Main.cpp
// 1D Peridynamics Simulation Program
// This program implements a peridynamics-based solver for 1D problems using Newton-Raphson iteration

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
    // Start timing the entire program execution
    auto total_start = std::chrono::high_resolution_clock::now();
    std::cout << "Starting 1D Peridynamics simulation!" << std::endl;

    /////////////////////////////////////////////////////////////////////////////////////////////////
    ///////////////////////////////// SIMULATION SETUP///////////////////////////////////////////////
    /////////////////////////////////////////////////////////////////////////////////////////////////

    // Parse command line arguments and initialize simulation parameters
    CLIOptions opts = parseArguments(argc, argv);
    int PD = 1;  // Problem dimension (1D)
    double domain_size = opts.domain_size;  // Total size of the computational domain
    double Delta = opts.Delta;  // Horizon radius (interaction distance)
    double L = opts.L;  // Lattice spacing / grid spacing
    double d = opts.d * domain_size;  // Damage parameter scaled by domain size
    int number_of_patches = opts.number_of_patches;  // Number of boundary patches on left
    int number_of_right_patches = opts.number_of_right_patches;  // Number of boundary patches on right
    double C1 = opts.C1;  // Material constant for peridynamic formulation
    double nn = opts.nn;  // Nonlocal parameter / power law exponent
    double F_prescribed = opts.F_prescribed;  // Prescribed force value
    std::string Prescribed_Flag = opts.Prescribed_Flag;  // Type of boundary condition ("Force" or "Displacement")
    std::string DEFflag = opts.DEFflag;  // Deformation flag
    std::string file_name = opts.output_dir;  // Output file name for logging
    int DOFs;  // Total number of degrees of freedom (to be computed)



    // 1. Compute the domain corners (boundaries)
    std::vector<double> Corners = Compute_Corners(domain_size);

    // 2. Create the interior mesh discretization
    std::vector<double> NLtmp = Mesh(Corners, L);
    
    // Create external patch nodes at boundaries
    std::vector<double> NLext = Patch(Corners, L, Delta, number_of_patches, number_of_right_patches);

    // Combine interior mesh and boundary patches into single node list
    std::vector<double> NL;
    NL.insert(NL.end(), NLtmp.begin(), NLtmp.end());
    NL.insert(NL.end(), NLext.begin(), NLext.end());
    
    // Sort all nodes in ascending order of position
    std::sort(NL.begin(), NL.end(), [](const double& a, const double& b) {
        return a < b;
    });

    // 3. Create point topology (convert node positions to Point objects with properties)
    std::vector<Point> PL = Topology(NL, L, Delta);

    // 4. Assign neighbor relationships (which points interact with each other within horizon)
    PL = AssignNgbrs(PL, L, Delta);

    // 5. Assign volumes to each point for integration
    PL = AssignVols(Corners, PL, L);

    // 6. Output mesh statistics
    std::cout << "======================================================" << std::endl;
    std::cout << "number of nodes                 : " << NL.size() << std::endl;
    std::cout << "number of points                : " << PL.size() << std::endl;

    // 7. Compute FF (force factor) for boundary conditions
    double FF = Compute_FF(PD, d, DEFflag);
    
    // 8. Assign boundary conditions to points based on their location and flags
    auto bc_result = AssignBCs(Corners, PL, FF, Prescribed_Flag, domain_size);
    PL = bc_result.first;
    
    // Assign global degree of freedom numbering to free points
    auto result = AssignGlobalDOF(PL);
    PL = result.first;
    DOFs = result.second;  // Total number of DOFs in the system
    std::cout << "number of DOFs                  : " << DOFs << std::endl;
    std::cout << "======================================================" << std::endl;

    ////////////////////////////////////////////////////////////////////////////////////////////////
    ///////////////////////////////////    LOGGING THE INFO     ////////////////////////////////////
    ////////////////////////////////////////////////////////////////////////////////////////////////

    // Initialize logger for writing simulation data to log files
    std::cout << "[LOG] Writing to: log_files/" << file_name << ".log" << std::endl;
    Logger logger(file_name);
    logger.writeHeader(file_name);
    
    // Prepare CSV file for timing results (append mode, with header if new)
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
    

    // Newton-Raphson solver parameters
    int steps = opts.steps;  // Number of load steps
    double load_step = 1.0 / steps;  // Incremental load factor per step
    double tol = opts.tol;  // Convergence tolerance
    int max_try = 100;  // Maximum number of Newton-Raphson iterations per load step
    double LF = 0.0;  // Current load factor (starts at 0, goes to 1)
    double F_rec_patch, F_rec_rightpatch = 0; // Reaction forces on boundary patches

    // Write simulation parameters to log file
    logger.writeParameters(domain_size, L, Delta, PL.size(), steps, C1, nn, Prescribed_Flag, F_prescribed, d, number_of_patches, number_of_right_patches);

    // Display simulation parameters to console
    std::cout << "======================================================" << std::endl;
    std::cout << "Simulation Parameters:" << std::endl;
    std::cout << "Domain Size: " << domain_size << " | Lattice Length / Delta: " << L<< " | Horizon: " << Delta << std::endl;
    std::cout << "Steps: " << steps << " | Load Step: " << load_step<< " | Tolerance: " << tol << std::endl;
    std::cout << "Material constant C1: " << C1 << std::endl;
    std::cout << "======================================================" << std::endl;

    // Initialize Eigen linear algebra objects for the system
    Eigen::VectorXd R = Eigen::VectorXd::Zero(DOFs);  // Residual vector
    Eigen::SparseMatrix<double> K;  // Stiffness matrix (sparse for efficiency)
    Eigen::VectorXd dx = Eigen::VectorXd::Zero(DOFs);  // Displacement increment vector

    // --- Start simulation timer --- //
    auto sim_start = std::chrono::high_resolution_clock::now();

    // Load stepping loop: gradually increase load from 0 to 1
    while (LF <= 1.0 + 1e-8) {
        std::cout << "\nLoad Factor: " << LF << std::endl;
        logger.writeLoadFactor(LF);
        
        // Apply prescribed boundary conditions (displacement or force) based on current load factor
        update_points(PL, LF, dx, Prescribed_Flag, F_prescribed, number_of_right_patches); 

        // Newton-Raphson iteration control variables
        int error_counter = 1;  // Iteration counter
        bool isNotAccurate = true;  // Convergence flag
        double normnull = 0.0;  // Initial residual norm for relative error calculation

        // Reset displacement increment for new load step
        dx.setZero();
        
        // Newton-Raphson iteration loop: solve nonlinear equilibrium at current load level
        while (isNotAccurate && error_counter <= max_try) {
            
            // Calculate internal forces and stiffness for all peridynamic bonds
            calculate_rk(PL, C1, Delta, nn);
            
            // Assemble global residual vector
            assembly(PL, DOFs, R, K, "residual");

            // Compute residual norm for convergence check
            double residual_norm = R.norm();
            double rel_norm;
            if (error_counter == 1) {
                // Store initial residual for relative error calculation
                normnull = std::max(residual_norm, 1e-10);
                std::cout << "Initial Residual Norm = " << residual_norm << std::endl;
            } else {
                // Check convergence based on relative or absolute tolerance
                rel_norm = residual_norm / normnull;
                std::cout << "Iter " << error_counter << ": Residual Norm = " << residual_norm << ", Relative = " << rel_norm << std::endl;
                if ((rel_norm - tol) < 1e-12 || (residual_norm - tol) < 1e-12) {
                    isNotAccurate = false;
                }
            }
            
            // Log convergence information
            logger.writeConvergence(error_counter, residual_norm, rel_norm);
            
            // Assemble global stiffness matrix
            assembly(PL, DOFs, R, K, "stiffness");

            // Solve linear system K * dx = -R using sparse LU decomposition
            Eigen::SparseLU<Eigen::SparseMatrix<double>> solver;
            solver.compute(K);
            if(solver.info() != Eigen::Success) {
                std::cout << "Nonlinear solver failed to compute!" << std::endl;
                break;
            }
            dx = solver.solve(-R);  
            
            // Update point positions with calculated displacement increment
            update_points(PL, LF, dx, "Calculated", F_prescribed, number_of_right_patches);

            // Reset reaction force accumulators
            F_rec_patch = 0.0;
            F_rec_rightpatch = 0.0;

            // Accumulate reaction forces from boundary patches
            // Different calculation depending on whether force or displacement is prescribed
            for (int i = 0; i < PL.size(); i++) {
                if (PL[i].Flag == "Right Patch" && Prescribed_Flag == "Force")
                    F_rec_rightpatch += PL[i].F_ext;
                if (PL[i].Flag == "Right Patch" && Prescribed_Flag == "Displacement")
                    F_rec_rightpatch += PL[i].residual;
                if ((PL[i].Flag == "Patch"))
                    F_rec_patch += PL[i].residual;
            }

            // At final converged state of final load step, log detailed patch force data
            if (!isNotAccurate && LF >= 1.0 - 1e-12) {
                const int H = number_of_patches;

                // Collect residual forces from left patch nodes
                std::vector<double> left_residuals;
                for (int i = 0; i < PL.size(); ++i)
                    if (PL[i].Flag == "Patch")
                        left_residuals.push_back(PL[i].residual);

                // Write patch forces to log file
                logger.writePatchForces(H, nn, left_residuals, F_rec_rightpatch);

                // Append force distribution data to CSV for post-processing
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

            // Report successful convergence
            if(isNotAccurate == false) {
                std::cout << "Converged after " << error_counter << " iterations." << std::endl<< std::endl;
                logger.writeConverged(error_counter);
            }
            
            error_counter++;
        }
        
        // Increment load factor for next load step
        LF += load_step;   
    }
    
    // Display final reaction forces after all load steps completed
    std::cout<<"Applied / Reaction Force on the RIGHT PATCH is : "<< F_rec_rightpatch  <<std::endl;
    std::cout<<"Reaction Force on the PATCH is : "<< F_rec_patch <<std::endl;
    std::cout<<"Total Reaction force = Rightpatch - Patch = " << (F_rec_rightpatch - F_rec_patch)<< std::endl<< std::endl;
    logger.writeReactoinForce(LF, F_rec_rightpatch, F_rec_patch, number_of_patches ,nn);
    
    // Optional: Output final state of all points (currently commented out)
    /*for (const auto& p : PL) {
        std::cout << "Point " << p.Nr << ": x = " << p.x << ",\t displacement = " << (p.x - p.X) << std::endl;
    }*/
    
    // --- End simulation timer --- //
    auto sim_end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> sim_duration = sim_end - sim_start;

    // --- End total timer --- //
    auto total_end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> total_duration = total_end - total_start;

    // Display timing statistics
    std::cout << "\nSimulation time: " << sim_duration.count() << " seconds" << std::endl;
    std::cout << "Total program time: " << total_duration.count() << " seconds" << std::endl;
    logger.writeTiming(sim_duration.count(), total_duration.count());

    // Close logger and exit program
    logger.close();
    return 0;
}