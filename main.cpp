#include <iostream>
#include <vector>
#include <cmath>
#include <iomanip>
#include <Eigen/Dense>
#include <Eigen/Sparse>

class Points
{
public:
    int Nr;                         // Point index
    double X;                       // Reference coordinates
    double x;                       // Current coordinates
    std::vector<int> neighbours;    // Neighbor list
    std::vector<double> neighborsx; // Current coordinates of neighbors
    std::vector<double> neighborsX; // Reference coordinates of neighbors
    std::string Flag;               // Patch/Point/Right Patch flag
    int BCflag{};                   // 0: Dirichlet; 1: Neumann
    double BCval{};                 // Boundary condition value
    int DOF{};                      // Global degree of freedom
    int DOC{};                      // Constraint flag
    int n1 = 0;                     // Number of 1-neighbor interactions
    double volume;                  // Volume
    double psi{};                   // Energy
    double R_a{};                   // Residual
    std::vector<double> K_ab{};     // Tangential stiffness per neighbor
    double V_eff{};                 // Effective volume

    Points() : Nr(0), X(0.0), x(0.0), volume(0.0) {}
};

std::vector<Points> mesh(double domain_size, int number_of_patches, double Delta,
                        int number_of_right_patches, int& DOFs, int& DOCs, double d)
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
        point.neighborsx.clear();
        point.neighborsX.clear();

        if (i < number_of_patches) {
            point.Flag = "Patch";
            point.BCval = 0;
            point.BCflag = 0;
            point.DOC = ++DOCs;
        }
        else if (i >= number_of_patches + number_of_points) {
            point.Flag = "RightPatch";
            point.BCval = d;  // Modified: Just apply d as displacement
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

void neighbour_list(std::vector<Points>& point_list, double& delta)
{
    for (auto &i : point_list) {
        i.neighbours.clear();
        i.neighborsx.clear();
        i.neighborsX.clear();
        i.n1 = 0;

        for (auto &j : point_list) {
            if ((i.Nr != j.Nr) && (std::abs(i.X - j.X) < delta))
            {
                i.neighbours.push_back(j.Nr);
                i.neighborsx.push_back(j.x);
                i.neighborsX.push_back(j.X);
                i.n1++;
            }
        }

        // Initialize K_ab vector to match number of neighbors
        i.K_ab.resize(i.n1, 0.0);
    }
}

void calculate_rk(std::vector<Points>& point_list, double C1, double delta)
{
    constexpr double pi = 3.14159265358979323846;
    double Vh = (4.0 / 3.0) * pi * std::pow(delta, 3);

    for (auto& i : point_list)
    {
        i.psi = 0.0;
        i.R_a = 0.0;
        i.K_ab.clear();
        i.V_eff = (i.n1 > 0) ? Vh / i.n1 : 0.0;

        for (size_t n = 0; n < i.neighbours.size(); n++) {
            double XiI = i.neighborsX[n] - i.X;
            double xiI = i.neighborsx[n] - i.x;

            double L = std::abs(XiI);
            double l = std::abs(xiI);
            double s = (l - L) / L;
            double eta = (xiI / l);

            i.psi += 0.5 * C1 * L * s * s;
            i.R_a += C1 * eta * s * i.V_eff;
            i.K_ab[n] = C1 / L * i.V_eff;
            // === Tangent stiffness
            //   (1 / |ξ|^3) * ξ_i ⊗ ξ_i = 1 / |ξ| and I (identity tensor) = 1 in 1D, therefore the expression becomes
            //   K_ab = ∂²ψ₁/∂x_i²
            //   = C₁ * ( δªͥ - δªᵇ) [ (1/|ξ|) +  ((1/|Σ |) - (1/|ξ|)) ] * V_eff
            //   = C1 * (1/|Σ |) * V_eff, as (1/|ξ|) terms gets cancelled.
            //
            // - First term: variation of 1/l term
            // - Second term: from derivative of xi term in force
            //
            // This corresponds to: (while assembly)
            //     K_aa = +Kval  when a == b  → (δₐᵦ = 1)
            //     K_ab = -Kval  when a ≠ b  → (δₐᵦ = 0)
        }
    }
}

void assembly(const std::vector<Points>& point_list, int DOFs, Eigen::VectorXd& R, Eigen::MatrixXd& K, const std::string& flag)
{
    if (flag == "residual") {
        R.setZero();
        for (const auto& point : point_list) {
            if (point.BCflag == 1 && point.DOF > 0 && point.DOF <= DOFs) {
                R(point.DOF - 1) += point.R_a;
            }
        }
        // Print residual vector
        std::cout << "size of the residual vector is: "<< R.size() << "\n";
        std::cout << "\nResidual Vector R:\n" << R << std::endl;
    }
    else if (flag == "stiffness") {
        K.setZero();

        for (const auto& point : point_list) {
            if (point.BCflag == 1 && point.DOF > 0 && point.DOF <= DOFs) {
                int row = point.DOF - 1;
                double diag = 0.0;

                for (size_t n = 0; n < point.neighbours.size(); ++n) {
                    int nbr_idx = point.neighbours[n];
                    const Points& neighbor = point_list[nbr_idx];

                    if (neighbor.BCflag == 1 && neighbor.DOF > 0 && neighbor.DOF <= DOFs) {
                        int col = neighbor.DOF - 1;
                        double k = point.K_ab[n];

                        K(row, col) -= k;
                        diag += k;
                    }
                }

                K(row, row) += diag;
            }
        }
        std::cout << "\nStiffness matrix size: " << K.rows() << " x " << K.cols() << std::endl;
        std::cout << "\nStiffness Matrix K:\n" << K << std::endl;
    }
}

void update_points(std::vector<Points>& point_list, double LF,
                  Eigen::VectorXd& dx, const std::string& Update_flag)
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
        for (size_t n = 0; n < point.neighbours.size(); n++) {
            int nbr_idx = point.neighbours[n];
            point.neighborsx[n] = point_list[nbr_idx].x;
        }
    }
}

int main()
{
    std::cout << "Starting 1D Peridynamics simulation!" << std::endl;

    // Parameters
    double domain_size = 1.0;
    double delta = 0.301;
    double Delta = 0.1;
    double d = 0.0001;
    int number_of_patches = 3;
    int number_of_right_patches = 1;
    double C1 = 0.05;
    int DOFs = 0;
    int DOCs = 0;

    // Create mesh
    std::vector<Points> points = mesh(domain_size, number_of_patches, Delta, number_of_right_patches, DOFs, DOCs, d);
    std::cout << "Mesh contains " << points.size() << " points with " << DOFs << " DOFs\n";
    neighbour_list(points, delta);

    // Debugging the points and their neighbours
    for (const auto& i : points) {
        std::cout << "Nr: " << i.Nr << std::endl << "X: [";
        std::cout << i.X << ", 0, 0";
        std::cout << "]" << std::endl << "x: [" << i.x << ", 0, 0 ]" << std::endl;
        std::cout << "Volume: " << i.volume << std::endl;
        std::cout << "BC: " << i.BCflag << std::endl << "Flag: " << i.Flag << std::endl;
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

    // Newton-Raphson setup
    int steps = 1;
    double load_step = (1.0 / steps);
    double tol = 1e-6;
    int max_try = 30;
    double LF = 0.0;

    // Initialize Eigen objects
    Eigen::VectorXd R = Eigen::VectorXd::Zero(DOFs);
    Eigen::MatrixXd K = Eigen::MatrixXd::Zero(DOFs, DOFs);
    Eigen::VectorXd dx = Eigen::VectorXd::Zero(DOFs);

    // Load stepping loop
    while (LF <= 1.0 + 1e-8) {
        std::cout << "\nLoad Factor: " << LF << std::endl;

        // Apply prescribed displacements
        update_points(points, LF, dx, "Prescribed");

        int error_counter = 1;
        bool isNotAccurate = true;
        double normnull = 0.0;

        // Reset dx to zero for new iteration
        dx.setZero();

        // Newton-Raphson iteration
        while (isNotAccurate && error_counter <= max_try) {
            // Calculate residuals and stiffness
            calculate_rk(points, C1, delta);

            // Assemble residual
            assembly(points, DOFs, R, K, "residual");

            double residual_norm = R.norm();
            if (error_counter == 1) {
                normnull = std::max(residual_norm, 1e-10);  // Prevent division by zero
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

            // Assemble stiffness and solve
            assembly(points, DOFs, R, K, "stiffness");

            // Improved linear solver with safety checks
            Eigen::FullPivLU<Eigen::MatrixXd> solver(K);
            dx = solver.solve(-R);

            // Update displacements
            update_points(points, LF, dx, "Displacement");
            error_counter++;

        }

        LF += load_step;

        // Output current state
        for (const auto& p : points) {
            if (p.Flag == "Point") {
                std::cout << "Point " << p.Nr << ": x = " << p.x
                          << ", displacement = " << (p.x - p.X) << std::endl;
            }
        }
    }

    std::cout << "Simulation completed successfully!" << std::endl;
    return 0;
}