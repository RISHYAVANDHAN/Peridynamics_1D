//
// Created by srini on 22/04/2025.
//

#ifndef POINTS_H
#define POINTS_H

#pragma once
#include <vector>
#include <string>
#include <Eigen/Dense>
#include <Eigen/Sparse>

class Points {
public:
    int Nr;                          // Point index
    double X;                        // Reference coordinates
    double x;                        // Current coordinates
    std::vector<int> neighbours;     // Neighbor list
    std::vector<double> neighborsx;  // Current coordinates of neighbors
    std::vector<double> neighborsX;  // Reference coordinates of neighbors
    std::string Flag;                // Patch/Point/Right Patch flag
    int BCflag{};                    // 0: Dirichlet; 1: Neumann
    double BCval{};                  // Boundary condition value
    int Forceflag{};                 // 0 = None, 1 = prescribed force
    double Forceval[3];              // Force value
    double ForceMag;                 // Force Magnitude
    int ForceDOF;                    // For augmented approach
    double ReactionForce;            // To store reaction forces
    int DOF{};                       // Global degree of freedom
    int DOC{};                       // Constraint flag
    int n1 = 0;                      // Number of 1-neighbor interactions
    double volume;                   // Volume
    double psi{};                    // Energy
    double residual{};               // Residual
    std::vector<double> stiffness{}; // Tangential stiffness per neighbor
    double JI{};                     // Effective volume

    Points();  // Default constructor
};

// Function declarations
std::vector<Points> mesh(double domain_size, int number_of_patches, double Delta, int number_of_right_patches, int& DOFs, int& DOCs, const std::vector<double>& patch_displacements, const std::vector<int>& force_nodes = {},
                        const std::vector<double>& forces = {});
void neighbour_list(std::vector<Points>& point_list, double& delta);
void calculate_rk(std::vector<Points>& point_list, double C1, double delta, double nn);
void assembly(const std::vector<Points>& point_list, int DOFs, int DOCs, Eigen::VectorXd& R, Eigen::SparseMatrix<double>& K, Eigen::MatrixXd& Kuu,
Eigen::MatrixXd& Kpu, Eigen::MatrixXd& Kpp,   const std::string& flag);
void update_points(std::vector<Points>& point_list, double LF, Eigen::VectorXd& dx,
                  const std::string& Update_flag, Eigen::VectorXd* forces = nullptr);


#endif //POINTS_H
