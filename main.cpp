#include <iostream>
#include <vector>
#include <cmath>
#include <iomanip>
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <string>
#include <algorithm> // for std::max

// --- Points Class Declaration ---
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
    double residual{};              // Residual
    std::vector<double> stiffness{};// Tangential stiffness per neighbor
    double V_eff{};                 // Effective volume

    Points() : Nr(0), X(0.0), x(0.0), volume(0.0) {}
};

// --- Mesh Function ---
std::vector<Points> mesh(double domain_size, int number_of_patches, double Delta,
                        int number_of_right_patches, int& DOFs, int& DOCs, double d);

// --- Neighbor List Function ---
void neighbour_list(std::vector<Points>& point_list, double& delta);

// --- Tangent Stiffness Calculation ---
void calculate_rk(std::vector<Points>& point_list, double C1, double delta);

// --- Residual and Stiffness Assembly ---
void assembly(const std::vector<Points>& point_list, int DOFs, Eigen::VectorXd& R, Eigen::SparseMatrix<double>& K, const std::string& flag);

// --- Update Points Function ---
void update_points(std::vector<Points>& point_list, double LF,
                  Eigen::VectorXd& dx, const std::string& Update_flag);

// --- Main Function ---
int main()
{
    std::cout << "Starting 1D Peridynamics simulation!" << std::endl;

    // Parameters
    double domain_size = 1.0;
    double delta = 0.301;
    double Delta = 0.1;
    double d = 1.0;
    int number_of_patches = 3;
    int number_of_right_patches = 1;
    double C1 = 1.0;
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
    int steps = 10;
    double load_step = (1.0 / steps);
    double tol = 1e-6;
    int max_try = 30;
    double LF = 0.0;

    // Initialize Eigen objects
    Eigen::VectorXd R = Eigen::VectorXd::Zero(DOFs);
    Eigen::SparseMatrix<double> K ;
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
            dx += solver.solve(-R);

            // Update displacements
            update_points(points, LF, dx, "Displacement");
            error_counter++;

        }

        LF += load_step;

        // Output current state
        for (const auto& p : points) {
            std::cout << "Point " << p.Nr << ": x = " << p.x << ", displacement = " << (p.x - p.X) << std::endl;
        }
    }

    std::cout << "Simulation completed successfully!" << std::endl;
    return 0;
}
