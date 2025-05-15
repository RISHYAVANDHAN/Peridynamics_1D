
#include <omp.h>
#include "Points.h"
#include <iostream>
#include <cmath>
#include <unordered_map>
#include <Eigen/Sparse>

// Default constructor for the Points class
Points::Points() : Nr(0), X(0.0), x(0.0), volume(0.0) {}

// Mesh generation function
std::vector<Points> mesh(double domain_size, int number_of_patches, double Delta, int number_of_right_patches, int& DOFs, int& DOCs, double d, const std::vector<int>& force_nodes,
                        const std::vector<double>& forces)
{
    std::vector<Points> point_list;
    const int number_of_points = std::floor(domain_size / Delta) + 1;
    int total_points = number_of_patches + number_of_right_patches + number_of_points;
    int index = 0;

    if (force_nodes.size() != forces.size()) {
        std::cerr << "Error: force_nodes and forces vectors must be the same size\n";
        return point_list;
    }

    std::unordered_map<int, double> force_map;
    for (size_t i = 0; i < force_nodes.size(); i++) {
        force_map[force_nodes[i]] = forces[i];

    }
    //#pragma omp parallel for
    for (int i = 0; i < total_points; ++i)
    {
        Points point;
        point.Nr = index++;
        point.X  = Delta/2 + i*Delta;
        point.x  = point.X;

        // ---------- 1. mark force nodes ----------
        if (force_map.count(point.Nr))
         {
            point.Forceflag = 1;
            point.ForceMag  = force_map[point.Nr];
            point.Forceval[0] = 0.0;    // Will be set during loading
            point.BCflag    = 1;        // FREE dof
            point.BCval     = 0.0;      // no prescribed displacement
            point.Flag      = "ForceNode";
        }

        // ---------- 2. displacement‑controlled patches ----------
        if (!point.Forceflag && point.X < number_of_patches*Delta) {
            point.Flag   = "Patch";
            point.BCflag = 0;           // fixed
            point.BCval  = 0.0;
        }
        else if (!point.Forceflag &&
                 point.X > Delta*(number_of_points+number_of_patches)) {
            point.Flag   = "RightPatch";
            point.BCflag = 0;           // fixed
            point.BCval  = d;           // prescribed displacement
                 }
        else if (!point.Forceflag) {    // ordinary free node
            point.Flag   = "Point";
            point.BCflag = 1;
            point.BCval  = (1+d)*point.X - point.X;
        }

        point.volume = 1;
        point_list.push_back(point);
    }

    // Reset counters
    DOFs = 0;
    DOCs = 0;

    // First pass: Assign DOFs to free nodes
    for (auto& point : point_list) {
        if (point.BCflag == 1) {  // Free nodes
            point.DOF = ++DOFs;
        }
    }

    // Second pass: Assign DOCs to constrained nodes
    for (auto& point : point_list) {
        if (point.BCflag == 0) {  // Constrained nodes
            point.DOC = ++DOCs;
        }
    }

    return point_list;
}

// Neighbour list calculation
void neighbour_list(std::vector<Points>& point_list, double& delta)
{
    //#pragma omp parallel for
    for (auto &i : point_list) {
        i.neighbours.clear();
        i.neighborsx.clear();
        i.neighborsX.clear();
        i.n1 = 0;

        for (auto &j : point_list) {
            if ((i.Nr != j.Nr) &&(std::abs(i.X - j.X) < delta))
            {
                i.neighbours.push_back(j.Nr);
                i.neighborsx.push_back(j.x);
                i.neighborsX.push_back(j.X);
                i.n1++;
            }
        }
        i.stiffness.resize(i.n1, 0.0);
    }
}

// === Tangent stiffness
//   (1 / |ξ|^3) * ξ_i ⊗ ξ_i = 1 / |ξ| and I (identity tensor) = 1 in 1D, therefore the expression becomes
//   stiffness = ∂²ψ₁/∂x_i²
//   = C₁ * ( δªͥ - δªᵇ) [ (1/|ξ|) +  ((1/|Σ |) - (1/|ξ|)) ] * JI
//   = C1 * (1/|Σ |) * JI, as (1/|ξ|) terms gets cancelled.
//
// - First term: variation of 1/l term
// - Second term: from derivative of xi term in force
//
// This corresponds to: (while assembly)
//     K_aa = +Kval  when a == b  → (δₐᵦ = 1)
//     stiffness = -Kval  when a ≠ b  → (δₐᵦ = 0)

// Calculate tangent stiffness and energy
void calculate_rk(std::vector<Points>& point_list, double C1, double delta)
{
    constexpr double pi = 3.14159265358979323846;
    double Vh = (4.0 / 3.0) * pi * std::pow(delta, 3);

    //#pragma omp parallel for
    for (auto& i : point_list) {
        // Reset values
        i.residual = 0.0;
        i.psi = 0.0;

        double JI = Vh / i.n1;

        // Create extended neighbor list (including the point itself)
        std::vector<int> neighborsE = i.neighbours;
        std::vector<double> neighborsEx = i.neighborsx;
        std::vector<double> neighborsEX = i.neighborsX;

        // Add the point itself to the extended neighbors
        neighborsE.push_back(i.Nr);
        neighborsEx.push_back(i.x);
        neighborsEX.push_back(i.X);

        const int NNgbrE = neighborsE.size(); // Extended neighbor count

        // Resize stiffness to accommodate all neighbors including self
        i.stiffness.clear();
        i.stiffness.resize(NNgbrE, 0.0);

        for (size_t j = 0; j < i.n1; j++) {
            double XiI = i.neighborsX[j] - i.X;
            double xiI = i.neighborsx[j] - i.x;

            double L = std::abs(XiI);
            double l = std::abs(xiI);

            if (L > 0.0) {  // Avoid division by zero
                double s = (l - L) / L;
                double eta = (xiI / l);

                // Calculate energy and residual
                i.psi += 0.5 * C1 * L * s * s;
                i.residual += C1 * eta * s * JI;

                // Calculate stiffness for each neighbor including self
                for (int b = 0; b < NNgbrE; b++) {
                    // This implements the (neighbors(i)==neighborsE(b))-(a==neighborsE(b)) logic
                    double K_factor = 0.0;
                    if (i.neighbours[j] == neighborsE[b]) {
                        K_factor += 1.0;
                    }
                    if (i.Nr == neighborsE[b]) {
                        K_factor -= 1.0;
                    }

                    // For 1D, the AA1 function simplifies to C1/L
                    double stiffness_contribution = C1 / L * JI * K_factor;
                    i.stiffness[b] += stiffness_contribution;
                }
            }
        }
        // Add external force contribution to residual
        if (i.Forceflag == 1)
        {
            std::cout << "Node " << i.Nr << " pre-force residual: " << i.residual
                      << ", applying force: " << i.Forceval[0] << std::endl;
            i.residual += i.Forceval[0];
            std::cout << "Post-force residual: " << i.residual << std::endl;
        }
    }
}


// Assemble residual or stiffness matrix
void assembly(const std::vector<Points>& point_list, int DOFs, int DOCs, Eigen::VectorXd& R, Eigen::SparseMatrix<double>& K, Eigen::MatrixXd& Kuu
,Eigen::MatrixXd& Kpu, Eigen::MatrixXd& Kpp, const std::string& flag)
{
    if (flag == "residual") {
        // Reset residual vector
        R.setZero();

        // Assemble residual
        for (const auto& point : point_list) {
            double R_P = point.residual;
            double BCflg = point.BCflag;
            int DOF = point.DOF;
            if (BCflg == 1) {
                R(DOF - 1) += R_P; // Adjust for 1-based indexing
                //std::cout << "[Residual] Added residual " << R_P << " at DOF " << DOF << " (Global Point ID: " << point.Nr << ")\n";
            }
        }

        std::cout << "Size of the residual vector is: " << R.size() << "\n";
        std::cout << "\nResidual Vector R:\n" << R << std::endl;
    }
    else if (flag == "stiffness") {
        K.setZero();
        std::vector<Eigen::Triplet<double>> triplets;

        int total_size = DOFs + DOCs;
        Eigen::MatrixXd Total_stiffness_matrix = Eigen::MatrixXd::Zero(total_size, total_size);

        // Reset stiffness matrix
        for (const auto& point : point_list) {
            // Determine global row index
            int row = (point.BCflag == 1) ? point.DOF - 1 : DOFs + point.DOC - 1;

            // Extended neighbor list
            std::vector<int> neighborsE = point.neighbours;
            neighborsE.push_back(point.Nr);

            for (size_t q = 0; q < neighborsE.size(); q++) {
                int nbr_idx = neighborsE[q];
                const auto& neighbor = point_list[nbr_idx];

                // Determine global column index
                int col = (neighbor.BCflag == 1) ? neighbor.DOF - 1 : DOFs + neighbor.DOC - 1;

                double Kval = point.stiffness[q];

                // Fill into dense matrix
                Total_stiffness_matrix(row, col) += Kval;
            }
        }

        // Now build triplets from Total_stiffness_matrix
        for (int i = 0; i < total_size; ++i) {
            for (int j = 0; j < total_size; ++j) {
                if (std::abs(Total_stiffness_matrix(i, j)) > 1e-14) { // Only nonzero entries
                    triplets.emplace_back(i, j, Total_stiffness_matrix(i, j));
                }
            }
        }

        K.resize(total_size, total_size);
        K.setFromTriplets(triplets.begin(), triplets.end());

        // Optional: Convert sparse matrix to dense for debugging or inspection
        Eigen::MatrixXd A = Eigen::MatrixXd(K);

        std::vector<int> unknown_indices;
        std::vector<int> prescribed_indices;

        for (const auto& point : point_list) {
            if (point.BCflag == 1) {
                unknown_indices.push_back(point.DOF - 1);  // DOF is 1-based
            } else {
                prescribed_indices.push_back(DOFs + point.DOC - 1);  // DOC is also 1-based
            }
        }

        // === Step 2: Resize Kuu, Kpu, Kpp ===
        int Kuu_size = static_cast<int>(unknown_indices.size());
        int Kpp_size = static_cast<int>(prescribed_indices.size());
        Kuu.resize(Kuu_size, Kuu_size);
        Kpu.resize(Kpp_size, Kuu_size);
        Kpp.resize(Kpp_size, Kpp_size);
        for (int i = 0; i < Kuu_size; ++i) {
            for (int j = 0; j < Kuu_size; ++j) {
                Kuu(i, j) = Total_stiffness_matrix(unknown_indices[i], unknown_indices[j]);
            }
        }

        // === Step 4: Fill Kpu ===
        for (int i = 0; i < Kpp_size; ++i) {
            for (int j = 0; j < Kuu_size; ++j) {
                Kpu(i, j) = Total_stiffness_matrix(prescribed_indices[i], unknown_indices[j]);
            }
        }

        // === Step 5: Fill Kpp ===
        for (int i = 0; i < Kpp_size; ++i) {
            for (int j = 0; j < Kpp_size; ++j) {
                Kpp(i, j) = Total_stiffness_matrix(prescribed_indices[i], prescribed_indices[j]);
            }
        }
        //std::cout << "\nStiffness matrix size: " << A.rows() << " x " << K.cols() << std::endl;
        //std::cout << "\nStiffness Matrix K:\n" << A << std::endl;
    }
}

// Update points based on displacement or prescribed values
void update_points(std::vector<Points>& point_list, double LF, Eigen::VectorXd& dx,
                  const std::string& Update_flag, Eigen::VectorXd* forces)
{
    if (Update_flag == "Prescribed") {
        // Update prescribed displacement nodes
        for (auto& i : point_list) {
            if (i.Forceflag == 1) {  // Handle force nodes FIRST
                i.Forceval[0] = LF * i.ForceMag;
            }

        }
        for (auto& point : point_list) {
            if (point.BCflag == 0 && point.Forceflag != 1) {  // Displacement-controlled only
                point.x = point.X + (LF * point.BCval);
            }
        }
    }
    else if (Update_flag == "Displacement") {
        // Update free DOFs from Newton-Raphson solution
        for (auto& i : point_list) {
            if (i.BCflag == 1 && i.DOF > 0) {  // Free DOFs
                i.x += dx(i.DOF - 1);
            }

        }
    }

    // Update neighbor coordinates
    for (auto& point : point_list) {
        for (size_t n = 0; n < point.neighbours.size(); n++) {
            int nbr_idx = point.neighbours[n];
            point.neighborsx[n] = point_list[nbr_idx].x;
        }
    }

    // Optional: Update forces if provided

}

