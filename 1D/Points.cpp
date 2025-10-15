// Points.cpp

#include <vector>
#include <cmath>
#include <cstdlib>
#include <algorithm>
#include <string>
#include <iostream>
#include <Eigen/Sparse>
#include "hyperdual.h"
#include "Points.h"

/**
 * @brief Computes the boundary coordinates (corners) for the 1D domain.
 * * Determines the left and right boundaries of the primary 1D simulation domain.
 * * @param SiZe The total size/length of the domain.
 * @return std::vector<double> A vector containing two elements: [0] the left corner (0.0), and [1] the right corner (SiZe).
 */
std::vector<double> Compute_Corners(double SiZe) {
    std::vector<double> Corners(2);
    
    Corners[0] = 0.0; // left corner
    Corners[1] = SiZe; // right corner
    
    return Corners;
}

/**
 * @brief Generates a uniform 1D mesh (list of node positions).
 * * Creates a list of equally spaced node coordinates within the range defined by Corners, 
 * using the specified lattice spacing L.
 * * @param Corners A constant reference to a vector containing the domain boundaries [A, B].
 * @param L The lattice spacing (length of a single element).
 * @return std::vector<double> A vector containing the coordinates of the generated nodes.
 */
std::vector<double> Mesh(const std::vector<double>& Corners, double L) {
    
    // generate NodeList
    double A = Corners[0]; // left south corner
    double B = Corners[1]; // right south corner
    
    // Calculate the number of nodes (Nx = (B-A)/L + 1)
    int Nx = static_cast<int>(round((B - A) / L)) + 1; // number of nodes along x including A and B
    
    std::vector<double> xx(Nx);
    for (int i = 0; i < Nx; i++) {
        // Linear spacing equivalent: xx[i] = A + i * deltaX
        xx[i] = A + i * (B - A) / (Nx - 1); 
    }
    
    int NoNs = Nx;
    
    std::vector<double> NL(NoNs);
    
    for (int i = 0; i < Nx; i++) {
        NL[i] = xx[i];
    }
    
    return NL;
}

/**
 * @brief Helper function to check if a node is in the Peridynamic Patch region.
 * * Determines if a given node lies *outside* the primary domain defined by Corners.
 * This is used to filter for patch nodes, which are only used for neighboring interior points.
 * * @param node The coordinate of the node to check.
 * @param Corners The boundary coordinates of the primary domain [A, B].
 * @return bool True if the node is outside the domain (in the patch), false otherwise (in the domain).
 */
bool PatchNode(const double& node, const std::vector<double>& Corners) {
    bool out = false;    
    double tol = 1e-4; // Tolerance for boundary check
    
    // Check if node is to the left of Corners[0] or to the right of Corners[1]
    if ((node - Corners[0]) < -tol || (node - Corners[1]) > tol) {
        out = true;
    }
    
    return out;
}

/**
 * @brief Generates a list of nodes that constitute the Peridynamic Patch regions.
 * * Extends the mesh on the left and right sides by a number of elements specified by 
 * 'patch' and 'right_patch', then filters out the nodes that fall within the original domain.
 * * @param Corners The boundary coordinates of the primary domain [A, B].
 * @param L The lattice spacing.
 * @param Delta The horizon size (not directly used here, but part of the original signature).
 * @param patch The number of L-length elements to extend the patch on the left (negative direction).
 * @param right_patch The number of L-length elements to extend the patch on the right (positive direction).
 * @return std::vector<double> A vector containing the coordinates of the patch nodes only.
 */
std::vector<double> Patch(const std::vector<double>& Corners, double L, double Delta, int patch, int right_patch) {    
    double l = patch * L;
    double r = right_patch * L;

    // Define the boundaries of the extended domain including the patches
    std::vector<double> Corners_mod(2);
    Corners_mod[0] = Corners[0] + l * (-1); // left south corner (extended left)
    Corners_mod[1] = Corners[1] + r * (1);  // right south corner (extended right)
    
    // Generate a temporary mesh over the extended domain
    std::vector<double> NLtmp = Mesh(Corners_mod, L);
    
    int NoNs = NLtmp.size();
    
    std::vector<double> NL;
    
    // Filter the nodes: only keep those that are *outside* the original domain
    for (int i = 0; i < NoNs; i++) {
        double node = NLtmp[i];
        
        if (PatchNode(node, Corners)) {
            NL.emplace_back(node);
        }
    }
    
    return NL;
}

/**
 * @brief Converts the raw list of node coordinates into a list of Point objects (Topology).
 * * Initializes a vector of 'Point' objects, assigning a unique point ID (NoP) and reference index (Nr)
 * to each point based on its coordinate (X).
 * * @param NL The combined list of coordinates (domain nodes + patch nodes).
 * @param L The lattice spacing (not directly used here, but part of the original signature).
 * @param Delta The horizon size (not directly used here, but part of the original signature).
 * @return std::vector<Point> The vector of initialized Point objects.
 */
std::vector<Point> Topology(const std::vector<double>& NL, double L, double Delta) {
    int NoNs = NL.size();    
    int NoP = 0; // point number/ID
    
    std::vector<Point> PL;
    
    for (int n = 0; n < NoNs; n++) {
        double X = NL[n];
        Point newPoint(NoP, X);
        newPoint.Nr = n; // Assign the raw index in NL as the reference index Nr
        PL.emplace_back(newPoint);    
        NoP = NoP + 1;
    }
    
    return PL;
}

/**
 * @brief Determines and assigns the list of neighbors for every Point.
 * * A point q is considered a neighbor of point p if the initial distance |Xp - Xq| 
 * is less than or equal to the horizon size (Delta). 
 * It also calculates the maximum theoretical number of neighbors (NmaxNgbr) and an 
 * influence area/volume factor (AV) for normalization.
 * * @param PL The vector of Point objects.
 * @param L The lattice spacing (not directly used in neighbor search but needed for NmaxNgbr).
 * @param Delta The horizon radius (interaction distance).
 * @return std::vector<Point> The updated vector of Point objects with assigned neighbors.
 */
std::vector<Point> AssignNgbrs(std::vector<Point> PL, double L, double Delta) {
    int NoPs = PL.size();    
    
    // Calculate NmaxNgbr (Maximum number of neighbors in a full horizon)
    int Del_by_L = static_cast<int>(floor(Delta / L));
    int NmaxNgbr = 0;
    for (signed i = -Del_by_L; i <= Del_by_L; i++) {
        // The condition sqrt(i*i)*L < Delta ensures we only count points strictly within the horizon
        // and i != 0 excludes the point itself.
        if ((sqrt(i * i) * L < Delta) && (i != 0)) { 
            NmaxNgbr = NmaxNgbr + 1;
        }
    }
    
    for (int p = 0; p < NoPs; p++) {
        std::vector<int> neighbors;
        std::vector<double> neighborsX;
        std::vector<double> neighborsx;    
        
        for (int q = 0; q < NoPs; q++) {
            // Check for q being a neighbor of p (q != p and distance <= Delta)
            if ((q != p) && (std::abs((PL[p].X - PL[q].X)) <= Delta)) {
                neighbors.emplace_back(q);
                neighborsX.emplace_back(PL[q].X);
                neighborsx.emplace_back(PL[q].x);
            }
        }
        
        // Store neighbor data in the Point object
        PL[p].neighbors = neighbors;
        PL[p].neighborsx = neighborsx;
        PL[p].neighborsX = neighborsX;
                
        int NNgbr = neighbors.size();
                
        // Calculate influence area/volume factor AV for volume correction/normalization
        double Amax = 2 * Delta;
        double AV = static_cast<double>(NNgbr + 1) / (NmaxNgbr + 1) * Amax;
        
        PL[p].NI = NNgbr; // Actual number of neighbors found
        PL[p].AV = AV;
    }
    
    return PL;
}

/**
 * @brief Assigns the numerical volume/area (Vol) to each point.
 * * In this 1D implementation, the volume is primarily the lattice spacing L, 
 * modified by a factor alpha to account for domain boundaries (0.5 for boundary nodes, 
 * 1 for interior nodes, 0 for patch nodes outside the Corners).
 * * @param Corners The boundary coordinates of the primary domain [A, B].
 * @param PL The vector of Point objects.
 * @param L The lattice spacing.
 * @return std::vector<Point> The updated vector of Point objects with assigned volumes.
 */
std::vector<Point> AssignVols(const std::vector<double>& Corners, std::vector<Point> PL, double L) {
    int NoPs = PL.size();
    
    double tol = 1e-4;
    
    double A = Corners[0]; // left boundary
    double B = Corners[1]; // right boundary
    
    for (int p = 0; p < NoPs; p++) {
        double X = PL[p].X;
        
        double alpha;
        
        if ((X - A) < (-tol) || (X - B) > (tol)) {
            // Point is outside the primary domain (Patch node)
            alpha = 0;
        } else if (abs(X - A) < tol || abs(X - B) < tol) {
            // Point is on the domain boundary
            alpha = 1.0 / 2.0;
        } else {
            // Point is in the domain interior
            alpha = 1;
        }
        
        double V = alpha * L;
        
        PL[p].Vol = V;
    }
    
    return PL;
}

/**
 * @brief Assigns material and simulation parameters to each point.
 * * Sets the lattice spacing (L), horizon (Delta), material ID (Mat), and material parameters (MatPars)
 * for every point in the list.
 * * @param inp The input vector of Point objects.
 * @param L The lattice spacing.
 * @param Delta The horizon radius.
 * @param MatPars The material parameter (e.g., C1, the stiffness coefficient).
 * @return std::vector<Point> The updated vector of Point objects.
 */
std::vector<Point> SetMaterial(const std::vector<Point>& inp, double L, double Delta, double& MatPars) {
    std::vector<Point> PL = inp;
    
    int NoPs = PL.size();
    for (int p = 0; p < NoPs; p++) {
        int mat = 1; // Material ID
        
        PL[p].L = L;
        PL[p].Delta = Delta;
        PL[p].Mat = mat;
        PL[p].MatPars = MatPars;
    }
    
    return PL;
}

/**
 * @brief Computes the Final Factor (FF) for deformation.
 * * Calculates the total prescribed stretch or deformation factor. In this case, 
 * it is a simple stretch factor (1 + nominal strain 'd').
 * * @param PD The problem dimension (1 in this case).
 * @param d The nominal displacement or strain to be applied over the domain.
 * @param DEFflag The flag indicating the type of deformation (not used in this simplified 1D model).
 * @return double The Final Factor (FF).
 */
double Compute_FF(int PD, double d, const std::string& DEFflag) {
    return (1.0 + d);
}

/**
 * @brief Helper function to set homogeneous Neumann boundary conditions (zero force).
 * * Initializes all points to have a boundary condition flag BCflg=1 (free DOF, force prescribed) 
 * and a prescribed value BCval=0.0 (zero force). This is the default state before specific 
 * boundary conditions are applied.
 * * @param PL The vector of Point objects.
 * @return std::vector<Point> The updated vector of Point objects.
 */
std::vector<Point> FreeAllPoints(std::vector<Point> PL) {    
    // free all points with no force ... prescribe homogeneous Neumann BC on all points
    int NoPs = PL.size();
    for (int i = 0; i < NoPs; i++) {
        PL[i].BCflg = 1; // 1 means a free DOF (force is prescribed/known)
        PL[i].BCval = 0.0; // Prescribed force value (zero for a free boundary)
    }
    
    return PL;
}

/**
 * @brief Helper function to assign global DOF and DOC identifiers.
 * * Assigns a global Degree of Freedom (DOF) number to points that are free to move 
 * (BCflg = 1, force prescribed) and a global Degree of Constraint (DOC) number to points 
 * with prescribed displacement (BCflg = 0, displacement known).
 * * @param PL The vector of Point objects.
 * @return std::pair<std::vector<Point>, int> A pair containing the updated PL and the total number of free DOFs.
 */
std::pair<std::vector<Point>, int> AssignGlobalDOF(std::vector<Point> PL) {    
    int NoPs = PL.size();
    int PD = 1; // Problem dimension
    int DOFs = 0; // Total Degrees of Freedom (free nodes)
    int DOCs = 0; // Total Degrees of Constraint (prescribed nodes)
    
    for (int i = 0; i < NoPs; i++) {
        double BCflg = PL[i].BCflg;
        
        int DOF = 0;
        int DOC = 0;
        
        // In 1D, there's only one component (p=0 to PD-1)
        for (int p = 0; p < PD; p++) { 
            if (BCflg == 1) {
                DOFs = DOFs + 1;
                DOF = DOFs; // Assign the next global DOF ID
            }
            else
            {
                DOCs += 1;
                DOC = DOCs; // Assign the next global DOC ID
            }
        }
        
        PL[i].DOF = DOF; // DOF is > 0 for free nodes, 0 for constrained
        PL[i].DOC = DOC; // DOC is > 0 for constrained nodes, 0 for free
    }
    
    return std::make_pair(PL, DOFs);
}

/**
 * @brief Assigns specific boundary conditions to the left and right patches.
 * * 1. Calls FreeAllPoints to initialize all points to BCflg=1.
 * 2. Applies zero displacement (BCflg=0, BCval=0.0) to the left patch (X < 0.0).
 * 3. Applies either prescribed displacement or prescribed force to the right patch (X > domain_size),
 * based on the Prescribed_Flag.
 * 4. Calls AssignGlobalDOF to finalize the DOF assignment.
 * * @param Corners The boundary coordinates of the primary domain [A, B].
 * @param PL The vector of Point objects.
 * @param FF The Final Factor (used to calculate the final prescribed displacement).
 * @param Prescribed_Flag String indicating 'Displacement' or 'Force' BC at the right patch.
 * @param domain_size The size of the primary domain.
 * @return std::pair<std::vector<Point>, int> A pair containing the updated PL and the total number of free DOFs.
 */
std::pair<std::vector<Point>, int> AssignBCs(const std::vector<double>& Corners, std::vector<Point> PL, const double& FF, const std::string& Prescribed_Flag, double domain_size) {
    int NoPs = PL.size();

    // 1. Initialize all points to homogeneous Neumann (zero force/free DOF)
    PL = FreeAllPoints(PL);
    
    for (int i = 0; i < NoPs; i++) {
        double X = PL[i].X;
        int BCflg = 1; // Default
        double BCval = 0.0; // Default
        
        if ((X < 0.0)) {
            // Left Patch: Prescribe zero displacement (Fixed)
            BCflg = 0; // Constrained DOF
            BCval = 0.0; // Prescribed displacement value
                        
            PL[i].BCflg = BCflg;
            PL[i].BCval = BCval;
            PL[i].Flag = "Patch";
        }
        else if ((X > domain_size))
        {
            // Right Patch: Apply displacement or force based on flag
            if (Prescribed_Flag == "Displacement"){
                BCflg = 0; // Constrained DOF
                // Prescribed displacement: (Final Factor * Initial Pos) - Initial Pos
                BCval = (FF * X) - X; 
            }
            else if (Prescribed_Flag == "Force"){
                BCflg = 1; // Free DOF (force prescribed)
                BCval = 0.0; // External force is handled via F_ext later
            }

            PL[i].BCflg = BCflg;
            PL[i].BCval = BCval;
            PL[i].Flag = "Right Patch";
        }
        else
        {
            // Interior points
            PL[i].BCval = 0.0;
            PL[i].Flag = "Point";
        }
    }
    
    // Final assignment of global DOF/DOC numbers
    auto result = AssignGlobalDOF(PL);
    return result;
}

/**
 * @brief Calculates the local peridynamic forces (residual) and stiffness contributions (rk).
 * * Iterates through each point and its neighbors, calculating the bond strain energy density (psi)
 * and its derivatives using hyperdual numbers:
 * - psi.real() -> Strain energy (used for PL[i].psi)
 * - psi.eps1() -> First derivative (used for residual)
 * - psi.eps1eps2() -> Second derivative (used for stiffness)
 * * @param PL The vector of Point objects (will be updated).
 * @param C1 The material constant.
 * @param delta The horizon size.
 * @param nn The exponent in the strain energy function.
 */
void calculate_rk(std::vector<Point>& PL, double C1, double delta, double nn)
{
    double Vh = 2 * delta; // 1D Horizon volume/area factor
    int NoPs = PL.size();

    for (int i = 0; i < NoPs; i++)
    {
        PL[i].residual = 0.0;
        PL[i].psi = 0.0;
        
        // Normalization factor: JI (J-integral term)
        double JI = Vh / PL[i].NI;

        // Create an extended neighbor list including the point itself (for stiffness indexing)
        std::vector<int> neighborsE = PL[i].neighbors;
        neighborsE.push_back(PL[i].Nr);
        const int NNgbrE = neighborsE.size();
        PL[i].stiffness = Eigen::VectorXd::Zero(NNgbrE);

        for (size_t j = 0; j < PL[i].NI; j++) {
            double XiI = PL[i].neighborsX[j] - PL[i].X; // Initial bond vector
            double xiI = PL[i].neighborsx[j] - PL[i].x; // Current bond vector
            double LL = std::abs(XiI); // Initial length

            // Skip bonds with negligible initial length
            if (LL < 1e-12) {
                continue;
            }

            // Hyperdual setup for automatic differentiation: d = x - X
            double epsilon = 1e-6; // Small constant for stability (under sqrt)
            hyperdual xiI_HD(xiI, 1.0, 1.0, 0.0); // Value: xiI, eps1: 1.0, eps2: 1.0, eps1eps2: 0.0
            
            // Current length: l = sqrt(xiI^2 + epsilon)
            hyperdual l = sqrt(xiI_HD * xiI_HD + epsilon);

            // Stretch calculation s = (|l - L| / L)^nn * (1/nn)
            // Using abs(l.real() - LL) for standard stretch calculation:
            double stretch = std::abs(l.real() - LL) / LL; 
            hyperdual s = (1.0 / nn) * (pow((stretch), nn));

            // Strain energy density: psi = 0.5 * C1 * L * s^2
            hyperdual psi = 0.5 * C1 * LL * s * s; 

            // Accumulate strain energy (real part of psi)
            PL[i].psi += psi.real(); 
            // Accumulate internal force (residual) using 1st derivative (psi.eps1) * JI
            PL[i].residual += psi.eps1() * JI; 

            // Accumulate stiffness contribution using 2nd derivative (psi.eps1eps2)
            for (int b = 0; b < NNgbrE; b++) {
                double K_factor = 0.0;
                // K_factor is 1 for the neighbor point, -1 for the current point (i)
                if (PL[i].neighbors[j] == neighborsE[b]) K_factor += 1.0;
                if (PL[i].Nr == neighborsE[b]) K_factor -= 1.0;

                PL[i].stiffness[b] += psi.eps1eps2() * JI * K_factor;
            }
        }
    }
}


/**
 * @brief Assembles the global Residual vector (R) or Stiffness matrix (K).
 * * Iterates through the Point list (PL) and maps the local residual/stiffness contributions
 * to the global system based on the assigned DOFs. Only points with BCflg=1 (free DOFs)
 * contribute to the system.
 * * @param point_list The constant vector of Point objects containing local residual/stiffness.
 * @param DOFs The total number of free degrees of freedom.
 * @param R The global residual vector (will be updated if flag is "residual").
 * @param K The global sparse stiffness matrix (will be updated if flag is "stiffness").
 * @param flag String indicating whether to assemble "residual" or "stiffness".
 */
void assembly(const std::vector<Point>& point_list, int DOFs, Eigen::VectorXd& R, Eigen::SparseMatrix<double>& K, const std::string& flag)
{
    if (flag == "residual") {
        R.setZero();

        // Assemble residual: R = R_internal + R_external
        for (const auto& point : point_list) {
            double R_P = point.residual + point.F_ext; // R_internal (internal forces) + R_external (external force)
            double BCflg = point.BCflg;
            int DOF = point.DOF;

            if (BCflg == 1) { // Only assemble for free DOFs
                R(DOF - 1) += R_P; // Adjust for 1-based indexing
            }
        }
    }
    else if (flag == "stiffness") {
        K.setZero();
        std::vector<Eigen::Triplet<double>> triplets; // Triplet list for sparse matrix

        // Assemble stiffness matrix
        for (const auto& point : point_list) {
            double BCflg_p = point.BCflg;
            int DOF_p = point.DOF;

            if (BCflg_p == 1) { // Only points with free DOFs contribute as a row
                // Create extended neighbor list including the point itself
                std::vector<int> neighborsE = point.neighbors;
                neighborsE.push_back(point.Nr);

                for (size_t q = 0; q < neighborsE.size(); q++) {
                    int nbr_idx = neighborsE[q];
                    double BCflg_q = point_list[nbr_idx].BCflg;
                    int DOF_q = point_list[nbr_idx].DOF;

                    if (BCflg_q == 1) { // Only neighbor points with free DOFs contribute as a column
                        double Kval = point.stiffness[q];
                        // Add contribution to K(DOF_p, DOF_q)
                        triplets.emplace_back(DOF_p - 1, DOF_q - 1, Kval); 
                    }
                }
            }
        }

        K.resize(DOFs, DOFs);
        K.setFromTriplets(triplets.begin(), triplets.end());
    }
}

/**
 * @brief Updates the current position (x) or external force (F_ext) of the points.
 * * Handles three types of updates based on the load factor (LF), calculated displacement (dx), 
 * or prescribed force:
 * 1. "Displacement": Updates 'x' for constrained nodes (BCflg=0) based on prescribed BCval.
 * 2. "Force": Updates 'F_ext' for right patch nodes based on the prescribed total force.
 * 3. "Calculated": Updates 'x' for free nodes (BCflg=1) by adding the displacement increment 'dx' 
 * from the Newton-Raphson solver.
 * Finally, it updates the neighbor coordinates (neighborsx) for all points.
 * * @param PL The vector of Point objects (will be updated).
 * @param LF The current load factor.
 * @param dx The calculated displacement increment vector.
 * @param Update_flag String indicating the type of update ("Displacement", "Force", or "Calculated").
 * @param F_prescribed The total prescribed external force (if applicable).
 * @param number_of_right_patches The number of points on the right patch (for load distribution).
 */
void update_points(std::vector<Point>& PL, double LF, Eigen::VectorXd& dx, const std::string& Update_flag, double F_prescribed, int number_of_right_patches)
{
    int NoPs = PL.size();
    if (Update_flag == "Displacement") {
        // Update constrained displacement nodes (BCflg = 0) based on load factor
        for (int i = 0; i < NoPs; i++) {
            if (PL[i].BCflg == 0) {
                // x = X + LF * u_prescribed (where u_prescribed = BCval)
                PL[i].x = PL[i].X + (LF * PL[i].BCval); 
            }
        }
    }
    else if (Update_flag == "Force") {
        // Update external force F_ext for force-prescribed right patch nodes
        for (int i = 0; i < NoPs; i++) {
            if (PL[i].Flag == "Right Patch") {
                // Distribute total prescribed force F_prescribed among all right patch points
                PL[i].F_ext = LF * F_prescribed / number_of_right_patches;
            }
        }
    }
    else if (Update_flag == "Calculated") {
        // Update free nodes (BCflg = 1) with the calculated displacement increment dx
        for (int i = 0; i < NoPs; i++) {
            if (PL[i].BCflg == 1 && PL[i].DOF > 0) {
                PL[i].x += dx(PL[i].DOF - 1); // Add displacement increment
            }
        }
    }

    // Update neighbor coordinates (neighborsx) based on the new current position (x)
    for (int i = 0; i < NoPs; i++) {
        for (size_t n = 0; n < PL[i].NI; n++) {
            int nbr_idx = PL[i].neighbors[n];
            // Access the updated 'x' coordinate of the neighbor point
            PL[i].neighborsx[n] = PL[nbr_idx].x;
        }
    }
}