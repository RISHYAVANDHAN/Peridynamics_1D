#include <iostream>
#include <vector>
#include <cmath>
#include <ctime>
#include <numeric>
#include <iomanip>
#include <Eigen/Dense>

class Points
{
public:
    // Problem Definition PD is always 1, pure 1D simulation
    int Nr;                         //  [1x1]    Point index
    double X;                       //  [PD x 1] Reference (Material) coordinates (here 1x1)
    double x;                       //  [PD x 1] Current (Spatial) coordinates (here 1x1)
    std::vector<int> neighbours;    //  [1 x NNbrs] Neighbour list
    std::string Flag;               //  Patch/ Point/ Right Patch flag
    int BCflag{};                   //  0: Dirichlet; 1: Neumann
    double BCval{};                 //  0: Displacement; 1: Force
    int DOF{};                      //  0: Fixed; 1: Free
    int DOC{};                      //  0: Free: 1: Fixed
    int n1 = 0;                     //  Number of 1-Neighbour interactions (1D - so just 1-Neighbour interaction)
    double volume;                  //  Volume occupied by the element
    double psi{};                   //  Energy
    double R_a{};                   //  Residual
    double K_ab{};                  //  Tangential stiffness
    double V_eff{};                 //  Effective volume

    Points()
        : Nr(0), X(0.0), x(0.0), volume(0.0) {}
};

double norm(const std::vector<double>& vec) {
    return std::sqrt(std::accumulate(vec.begin(), vec.end(), 0.0,
                                     [](double acc, double val) {
                                         return acc + val * val;
                                     }));
}

std::vector<Points> mesh(double domain_size, int number_of_patches, double Delta, int number_of_right_patches, int& DOFs, int& DOCs, double d)
{
    std::vector<Points> point_list;
    const int number_of_points = std::floor(domain_size / Delta) + 1;
    const int extended_domain_size = number_of_patches + number_of_right_patches + number_of_points;
    int total_points = number_of_patches + number_of_right_patches + number_of_points;
    int index = 0;
    double FF = 1 + d;

    for (int i = 0; i < total_points; i++) {
        Points point;
        point.Nr = index++;
        point.X = Delta / 2 + i * Delta;
        if (i < number_of_patches) {
            point.Flag = "Patch";
            point.x = point.X;
            point.BCval = 0;
            point.BCflag = 0;
            point.DOC = ++DOCs;
        }
        else if (i >= number_of_patches + number_of_points) {
            point.Flag = "RightPatch";
            point.x = point.X;
            point.BCval = 1 + d;
            point.BCflag = 0;
            point.DOC = ++DOCs;
        }
        else {
            point.Flag = "Point";
            point.x = point.X;
            point.BCflag = 1;
            point.BCval = (FF * point.X) - point.X;
            point.DOF = ++DOFs;
        }

        point.volume = 1;
        point_list.push_back(point);
    }

    return point_list;
}

void neighbour_list(std::vector<Points>& point_list, double& delta)
{
    for (auto &i : point_list)
    {
        //Debugging for checkups
        for (auto &j : point_list) {
            if ((i.Nr != j.Nr) && ((std::abs(i.X - j.X) < (delta)))) {
                i.neighbours.push_back({j.Nr}); // 1 neighbour interaction in 1d
                i.n1++;
            }
        }
    }
}

double compute_psi(double C, double XiI, double xiI)
{
    double s = xiI/ XiI;
    return 0.5 * C * XiI * std::pow((s - 1), 2);
}

void calculate_rk(std::vector<Points>& point_list, double C1, double delta)
{
    constexpr double pi = 3.14159265358979323846;
    double Vh = (4.0 / 3.0) * pi * std::pow(delta, 3);

    // Reset values before calculation
    for (auto& i : point_list) {
        i.psi = 0.0;
        i.R_a = 0.0;
        i.K_ab = 0.0;
    }

    for (auto& i : point_list)
    {
        if (i.n1 == 0) continue;
        i.V_eff = Vh / i.n1;

        for (int nbr : i.neighbours)
        {
            Points& j = point_list[nbr];

            double XiI = std::abs(i.X - j.X);
            double xiI= std::abs(i.x - j.x);

            // === Energy
            double s = xiI/ XiI;
            i.psi += 0.5 * C1 * XiI * std::pow(s - 1.0, 2.0);

            // === Residual
            double R1 = 0;
            double R_temp = C1 * i.V_eff * ((1.0 / XiI) - (1.0 /xiI)) *xiI;
            R1 = R1 + R_temp;
            i.R_a += R1; // R2 and R3 is 0, because it is a 1-neighbour interaction

            // === Tangent stiffness
            //   (1 / |ξ|^3) * ξ_i ⊗ ξ_i = 1 / |ξ| and I (identity tensor) = 1 in 1D, therefore the expression becomes
            //   K_ab = ∂²ψ₁/∂x_i²
            //   = C₁ * ( δªͥ - δªᵇ) [ (1/|ξ|) +  ((1/|Σ |) - (1/|ξ|)) ] * V_eff
            //
            // - First term: variation of 1/l term
            // - Second term: from derivative of xi term in force
            //
            // This corresponds to: (while assembly)
            //     K_aa = +Kval  when a == b  → (δₐᵦ = 1)
            //     K_ab = -Kval  when a ≠ b  → (δₐᵦ = 0)
            //
            double Kval = C1 * i.V_eff * ((1 / xiI) + ((1.0 / XiI) - (1.0 /xiI)));
            i.K_ab += Kval;
        }
    }
}

// Combined assembly function that can handle both residual and stiffness
void assembly(const std::vector<Points>& point_list, int DOFs, Eigen::VectorXd& R, Eigen::MatrixXd& K, const std::string& flag) {
    if (flag == "residual") {
        // Clear the residual vector
        R.setZero();

        // Loop over each point in the mesh
        for (const auto& point : point_list) {
            // Get the residual at each point
            double R_P = point.R_a;

            // If the point has a boundary condition flag set (i.e., DOF is free)
            if (point.BCflag == 1) {
                // Accumulate the residual based on the degree of freedom (DOF)
                if (point.DOF > 0 && point.DOF <= DOFs) {
                    R(point.DOF - 1) += R_P; // Adjusting for 0-based indexing
                }
            }
        }
    }
    if (flag == "stiffness") {
        // Clear the stiffness matrix
        K.setZero();

        // In 1D, for each point we need to consider its neighbors
        for (const auto& point : point_list) {
            // Only process points with free DOFs
            if (point.BCflag == 1 && point.DOF > 0 && point.DOF <= DOFs) {
                int row = point.DOF - 1; // Adjust for 0-based indexing

                // Add the point itself to the neighbor list for processing
                std::vector<int> all_neighbors = point.neighbours;
                all_neighbors.push_back(point.Nr); // Add the point itself

                // Process all connections (point to itself and to its neighbors)
                for (int nbr : all_neighbors) {
                    const Points& neighbor = point_list[nbr];
                    if (neighbor.BCflag == 1 && neighbor.DOF > 0 && neighbor.DOF <= DOFs) {
                        int col = neighbor.DOF - 1; // Adjust for 0-based indexing

                        // Diagonal term (point itself)
                        if (point.Nr == neighbor.Nr) {
                            K(row, col) += point.K_ab;
                        }
                        // Off-diagonal terms (neighbors)
                        else {
                            K(row, col) -= point.K_ab; // Negative for off-diagonal elements
                        }
                    }
                }
            }
        }
    }
}

std::vector<Points> update(std::vector<Points>& point_list, double arg, const std::string& Update_flag)
{
    if (Update_flag == "Prescribed")
    {
        double LF = arg;
        for (auto& i : point_list)
        {
            if(i.BCflag == 0)
            {
                i.x = i.X + (LF * i.BCval);
            }
        }
    }
    else if (Update_flag == "Displacement")
    {
        for (auto& i : point_list)
        {
            if(i.BCflag == 1 && i.DOF > 0)
            {
                i.x = i.x + arg;
            }
        }
    }
    return point_list;
}

int main()
{
    std::cout << "Starting 1D Peridynamics simulation!" << std::endl;
    double domain_size = 1.0;  // Domain size
    double delta = 0.301;      // Horizon size
    double Delta = 0.1;        // Grid spacing
    double d = 0.0001;         // Prescribed displacement
    int number_of_patches = 3; // std::ceil(delta / Delta) but 3 for now
    int number_of_right_patches = 1;
    double C1 = 0.5;           // Material constant
    int DOFs = 0;
    int DOCs = 0;
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

    // ==== SYSTEM SETUP COMPLETE =========//
    // ==== NEWTON RAPHSON STARTS =========//

    int steps = 1;
    double load_step = d/steps;
    double tol = 1e-11;
    int counter = 0;
    int min_try = 0;
    int max_try = 20;
    double LF = 0.0;

    std::cout << "======================================================" << std::endl;
    std::cout << "Simulation Parameters:" << std::endl;
    std::cout << "Domain Size: " << domain_size << " | Delta: " << Delta << " | Horizon: " << delta << std::endl;
    std::cout << "Steps: " << steps << " | Load Step: " << load_step << " | Tolerance: " << tol << std::endl;
    std::cout << "Material constant C1: " << C1 << std::endl;
    std::cout << "======================================================" << std::endl;

    // Create Eigen objects for the solver
    Eigen::VectorXd R = Eigen::VectorXd::Zero(DOFs);
    Eigen::MatrixXd K = Eigen::MatrixXd::Zero(DOFs, DOFs);
    Eigen::VectorXd dx = Eigen::VectorXd::Zero(DOFs);

    while (LF <= 1.0 + 1e-8)
    {
        std::cout << "Load Factor: " << LF << std::endl;
        std::string update_flag = "Prescribed";
        std::vector<Points> point_list = update(points, LF, update_flag);

        int error_counter = 1;
        bool isNotAccurate = true;
        double normnull = 0.0;

        while(isNotAccurate && error_counter <= max_try)
        {
            // Calculate residuals and stiffness
            calculate_rk(point_list, C1, delta);

            // Assembly residual vector
            assembly(point_list, DOFs, R, K, "residual");

            // Get residual norm
            double residual_norm = R.norm();

            // First iteration: store initial residual norm
            if(error_counter == 1)
            {
                normnull = residual_norm;
                std::cout << "Residual Norm @ Increment " << counter
                          << " @ Iteration " << error_counter
                          << " : " << std::scientific << std::setprecision(2)
                          << residual_norm << "    ,    normalized : 1" << std::endl;
            }
            else
            {
                std::cout << "Residual Norm @ Increment " << counter
                          << " @ Iteration " << error_counter
                          << " : " << std::scientific << std::setprecision(2)
                          << residual_norm << "    ,    normalized : "
                          << (residual_norm / normnull) << std::endl;

                // Check convergence
                if ((residual_norm / normnull) < tol || residual_norm < tol)
                {
                    isNotAccurate = false;
                    std::cout << "Converged after " << error_counter << " iterations." << std::endl;
                }
            }

            if (isNotAccurate)
            {
                // Assembly stiffness matrix
                assembly(point_list, DOFs, R, K, "stiffness");

                // Solve system K*dx = -R using Eigen
                dx = K.colPivHouseholderQr().solve(-R);

                // Update point positions
                std::string displ_flag = "Displacement";
                for (int i = 0; i < DOFs; i++)
                {
                    for (auto& p : point_list)
                    {
                        if (p.DOF == i + 1)  // Match DOF index (1-based)
                        {
                            p.x += dx(i);
                        }
                    }
                }

                error_counter++;
            }
        }

        // Save the updated points for next load step
        points = point_list;

        // Increment load factor
        LF += load_step;
        counter++;

        // Output current state
        std::cout << "Current positions:" << std::endl;
        for (const auto& p : points)
        {
            if (p.Flag == "Point")
            {
                std::cout << "Point " << p.Nr << ": x = " << p.x
                          << ", displacement = " << (p.x - p.X) << std::endl;
            }
        }
        std::cout << "======================================================" << std::endl;
    }

    std::cout << "Simulation completed successfully!" << std::endl;
    return 0;
}