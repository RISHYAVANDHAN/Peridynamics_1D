
#include "Points.h"
#include <iostream>
#include <cmath>
#include <Eigen/Sparse>
#include "hyperdual.h"

// Default constructor for the Points class
Points::Points() : Nr(0), X(0.0), x(0.0), volume(0.0) {}

// Mesh generation function
std::vector<Points> mesh(double domain_size, int number_of_patches, double Delta, int number_of_right_patches, int& DOFs, int& DOCs, double d)
{
    std::vector<Points> point_list;
    const int number_of_points = std::floor(domain_size / Delta) + 1;
    int total_points = number_of_patches + number_of_right_patches + number_of_points;
    int index = 0;
    double FF = 1 + d;

    for (int i = 0; i < total_points; i++) {
        Points point;
        point.Nr = index++;
        point.X = Delta / 2 + i * Delta;
        point.x = point.X;
        point.neighbours.clear();
        point.neighborsx.clear();
        point.neighborsX.clear();

        if (point.X < (number_of_patches * Delta)) {
            point.Flag = "Patch";
            point.BCval = 0;
            point.BCflag = 0;
            point.DOC = ++DOCs;
        }
        else if ((point.X > (Delta * (number_of_points + number_of_patches))))
        {
            point.Flag = "RightPatch";
            point.BCval = d;
            point.BCflag = 0;
            point.DOC = ++DOCs;
        }
        else {
            point.Flag = "Point";
            point.BCflag = 1;
            point.BCval = (FF * point.X) - point.X;
            point.DOF = ++DOFs;
        }

        point.volume = 1;
        point_list.push_back(point);
    }

    // Recalculate DOFs and assign correct indices
    DOFs = 0;
    for (auto& point : point_list) {
        if (point.BCflag == 1) {
            point.DOF = ++DOFs;
        }
    }

    return point_list;
}

// Neighbour list calculation
void neighbour_list(std::vector<Points>& point_list, double& delta)
{
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

void calculate_rk(std::vector<Points>& point_list, double C1, double delta, double nn)
{
    double Vh = 2 * delta;

    for (auto& i : point_list)
    {
        i.residual = 0.0;
        i.psi = 0.0;
        double JI = Vh / i.n1;

        std::vector<int> neighborsE = i.neighbours;
        std::vector<double> neighborsEx = i.neighborsx;
        std::vector<double> neighborsEX = i.neighborsX;

        neighborsE.push_back(i.Nr);
        neighborsEx.push_back(i.x);
        neighborsEX.push_back(i.X);

        const int NNgbrE = neighborsE.size();
        i.stiffness.clear();
        i.stiffness.resize(NNgbrE, 0.0);

        for (size_t j = 0; j < i.n1; j++) {
            double XiI = i.neighborsX[j] - i.X;
            double xiI = i.neighborsx[j] - i.x;
            double LL = std::abs(XiI);

            if (LL > 0.0) {
                hyperdual xiI_HD(xiI, 1.0, 1.0, 0.0);
                hyperdual ll = fabs(xiI_HD);

                // Lambda power-law stretch: s = (1/nn) * [ (l/LL)^nn - 1 ]
                hyperdual s = (1.0 / nn) * (pow(ll / LL, nn) - 1.0);

                hyperdual psi = 0.5 * C1 * LL * s * s;

                i.psi += psi.real();
                i.residual += psi.eps1() * JI;

                for (int b = 0; b < NNgbrE; b++) {
                    double K_factor = 0.0;
                    if (i.neighbours[j] == neighborsE[b]) K_factor += 1.0;
                    if (i.Nr == neighborsE[b]) K_factor -= 1.0;

                    i.stiffness[b] += psi.eps1eps2() * JI * K_factor;
                }
            }
        }
    }
}



// Assemble residual or stiffness matrix
void assembly(const std::vector<Points>& point_list, int DOFs, Eigen::VectorXd& R, Eigen::SparseMatrix<double>& K, const std::string& flag)
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

        //std::cout << "Size of the residual vector is: " << R.size() << "\n";
        //std::cout << "\nResidual Vector R:\n" << R << std::endl;
    }
    else if (flag == "stiffness") {
        // Reset stiffness matrix
        K.setZero();
        std::vector<Eigen::Triplet<double>> triplets;

        // Assemble stiffness
        for (const auto& point : point_list) {
            double BCflg_p = point.BCflag;
            int DOF_p = point.DOF;

            if (BCflg_p == 1) {
                // Create extended neighbor list including the point itself
                std::vector<int> neighborsE = point.neighbours;
                neighborsE.push_back(point.Nr);

                for (size_t q = 0; q < neighborsE.size(); q++) {
                    int nbr_idx = neighborsE[q];
                    double BCflg_q = point_list[nbr_idx].BCflag;
                    int DOF_q = point_list[nbr_idx].DOF;

                    if (BCflg_q == 1) {
                        double Kval = point.stiffness[q];
                        triplets.emplace_back(DOF_p - 1, DOF_q - 1, Kval);
                        //std::cout << "[Stiffness] K(" << DOF_p << "," << DOF_q << ") = " << Kval << " from Point " << point.Nr << " to Point " << nbr_idx << "\n";
                    }
                }
            }
        }

        K.resize(DOFs, DOFs);
        K.setFromTriplets(triplets.begin(), triplets.end());

        Eigen::MatrixXd A = Eigen::MatrixXd(K);

        //std::cout << "\nStiffness matrix size: " << A.rows() << " x " << K.cols() << std::endl;
        //std::cout << "\nStiffness Matrix K:\n" << A << std::endl;
    }
}

// Update points based on displacement or prescribed values
void update_points(std::vector<Points>& point_list, double LF, Eigen::VectorXd& dx, const std::string& Update_flag)
{
    if (Update_flag == "Prescribed") {
        for (auto& i : point_list) {
            if (i.BCflag == 0) {
                i.x = i.X + (LF * i.BCval);
            }
        }
    }
    else if (Update_flag == "Displacement") {
        for (auto& i : point_list) {
            if (i.BCflag == 1 && i.DOF > 0) {
                i.x = i.X + dx(i.DOF - 1);
            }
        }
    }

    // Update neighbor coordinates by directly accessing updated coordinates
    for (auto& point : point_list) {
        for (size_t n = 0; n < point.neighbours.size(); n++) {
            int nbr_idx = point.neighbours[n];
            point.neighborsx[n] = point_list[nbr_idx].x;
        }
    }
}