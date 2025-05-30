#include "Points.h"
#include <iostream>
#include <cmath>
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
    double FF = 1.0 + d;

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
            point.BCval = 0.0;
            point.BCflag = 0;
            point.DOC = ++DOCs;
        }
        else if ((point.X > (Delta * (number_of_points + number_of_patches))))
        {
            point.Flag = "RightPatch";
            point.BCval = (FF * point.X) - point.X;
            point.BCflag = 0;
            point.DOC = ++DOCs;
        }
        else {
            point.Flag = "Point";
            point.BCflag = 1;
            point.BCval = 0.0;
            point.DOF = ++DOFs;
        }

        point.volume = Delta;
        point_list.emplace_back(point);
    }

    // Recalculate DOFs and assign correct indices
    DOFs = 0;
    for (auto& point : point_list) {
        if (point.BCflag == 1.0) {
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
                i.neighbours.emplace_back(j.Nr);
                i.neighborsx.emplace_back(j.x);
                i.neighborsX.emplace_back(j.X);
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


void calculate_rk(std::vector<Points>& point_list, double C1, double delta, double nn)
{
    // i just included nn here, so as to no change everything back and forth when playing around with and without power law.
    // this is without power law.

    double Vh = 2 * delta;
    
    for (size_t i = 0; i < point_list.size() ; i++)
    {
        point_list[i].residual = 0.0;
        point_list[i].psi = 0.0;
        point_list[i].stiffness.clear();

        double JI = Vh / point_list[i].n1;

        // Create extended neighbor list (including the point itself)
        std::vector<int> neighborsE = point_list[i].neighbours;
        std::vector<double> neighborsEx = point_list[i].neighborsx;
        std::vector<double> neighborsEX = point_list[i].neighborsX;

        // Add the point itself to the extended neighbors
        neighborsE.emplace_back(point_list[i].Nr);
        neighborsEx.emplace_back(point_list[i].x);
        neighborsEX.emplace_back(point_list[i].X);

        const int NNgbrE = neighborsE.size(); // Extended neighbor count

        // Resize stiffness to accommodate all neighbors including self
        point_list[i].stiffness.resize(NNgbrE, 0.0);

        for (size_t j = 0; j < point_list[i].n1; j++) {
            double XiI = point_list[i].neighborsX[j] - point_list[i].X;
            double xiI = point_list[i].neighborsx[j] - point_list[i].x;

            double L = std::abs(XiI);
            double l = std::abs(xiI);

            if (L > 0.0) {  // Avoid division by zero
                double s = (l - L) / L;
                double eta = (xiI / l);

                // Calculate energy and residual
                point_list[i].psi += 0.5 * C1 * L * s * s * JI;
                double R1_temp = C1 * eta * s * JI;
                point_list[i].residual += R1_temp;

                // Calculate stiffness for each neighbor including self
                for (size_t b = 0; b < NNgbrE; b++) {
                    // This implements the (neighbors(i)==neighborsE(b))-(a==neighborsE(b)) logic
                    double K_factor = 0.0;
                    if (point_list[i].neighbours[j] == neighborsE[b]) {
                        K_factor -= 1.0;
                    }
                    if (point_list[i].Nr == neighborsE[b]) {
                        K_factor += 1.0;
                    }
                    // For 1D, the AA1 function simplifies to C1/L
                    double stiffness_contribution = C1 * (1.0 / L) * JI * K_factor;
                    point_list[i].stiffness[b] += stiffness_contribution;
                    //std::cout<<"Stiffness vector for 1d of the point: "<<point_list[i].Nr<<" is: "<<point_list[i].stiffness[b]<<std::endl;
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
            //std::cout<<"Nr: "<<point.Nr<<" R_P: "<<R_P<<std::endl; 
            double BCflg = point.BCflag;
            size_t DOF = point.DOF;
            if (BCflg == 1) {
                R[DOF - 1] += R_P; // Adjust for 1-based indexing
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
            double DOF_p = point.DOF;
            size_t ii = DOF_p - 1;

            if (BCflg_p == 1.0) {
                // Create extended neighbor list including the point itself
                std::vector<int> neighborsE = point.neighbours;
                neighborsE.emplace_back(point.Nr);

                for (size_t q = 0; q < neighborsE.size(); q++) {
                    int nbr_idx = neighborsE[q];
                    double BCflg_q = point_list[nbr_idx].BCflag;
                    double DOF_q = point_list[nbr_idx].DOF;
                    size_t jj = DOF_q - 1;

                    if (BCflg_q == 1.0) {
                        double Kval = point.stiffness[q];
                        double vv = Kval;
                        triplets.emplace_back(ii, jj, vv);
                        //triplets.emplace_back(jj, ii, vv);
                        //std::cout << "[Stiffness] K(" << DOF_p << "," << DOF_q << ") = " << Kval << " from Point " << point.Nr << " to Point " << nbr_idx << "\n";
                    }
                }
            }
        }

        K.resize(DOFs, DOFs);
        K.setFromTriplets(triplets.begin(), triplets.end());

        Eigen::MatrixXd A = Eigen::MatrixXd(K);
        Eigen::MatrixXd symmetric_coeff = A - A.transpose();
        //std::cout << "\nStiffness matrix size: " << A.rows() << " x " << K.cols() << std::endl;
        std::cout << "\nStiffness Matrix K:\n" << A << std::endl;
        std::cout<<"Symmetric check: \n"<<symmetric_coeff<<std::endl;
        //Eigen::SparseMatrix<double> K_sym = 0.5 * (K + Eigen::SparseMatrix<double>(K.transpose()));
        //K = K_sym;
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
                i.x += dx(i.DOF - 1);
            }
        }
    }

    // Update neighbor coordinates by directly accessing updated coordinates
    for (auto& point : point_list) {
        //point.neighborsx.clear();
        for (size_t n = 0; n < point.neighbours.size(); n++) {
            size_t nbr_idx = point.neighbours[n];
            point.neighborsx[n] = point_list[nbr_idx].x;
        }
    }
}