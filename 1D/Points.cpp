
#include <vector>
#include <cmath>
#include <algorithm>
#include <string>
#include <iostream>
#include <unordered_map>
#include <Eigen/Sparse>
#include "hyperdual.h"
#include "Points.h"

// Function 1: Corners = Compute_Corners(PD,SiZe);
std::vector<double> Compute_Corners(double SiZe) {
    std::vector<double> Corners(2);
    
    Corners[0] = 0.0; // left corner
    Corners[1] = SiZe; // right corner
    
    // Corners[0] = 1/2 * SiZe * [-1];
    // Corners[1] = 1/2 * SiZe * [1];
    
    return Corners;
}

// 2nd function: [ NLtmp ] = Mesh( Corners , L );
std::vector<double> Mesh(const std::vector<double>& Corners, double L) {
    
    // generate NodeList
    double A = Corners[0]; // left south corner
    double B = Corners[1]; // right south corner
    
    int Nx = static_cast<int>(round((B - A) / L)) + 1; // number of nodes along x including A and B
    
    std::vector<double> xx(Nx);
    for (int i = 0; i < Nx; i++) {
        xx[i] = A + i * (B - A) / (Nx - 1); // linspace equivalent
    }
    
    int NoNs = Nx;
    
    std::vector<double> NL(NoNs);
    
    for (int i = 0; i < Nx; i++) {
        NL[i] = xx[i];
    }
    
    return NL;
}

// Helper function for PatchNode
bool PatchNode(const double& node, const std::vector<double>& Corners) {
    bool out = false;    
    double tol = 1e-4;
    
    if ((node - Corners[0]) < -tol || (node - Corners[1]) > tol) {
        out = true;
    }
    
    return out;
}

// 3rd function: [ NLext ] = Patch( Corners , L , Delta );
std::vector<double> Patch(const std::vector<double>& Corners, double L, int number_of_left_patches, int number_of_right_patches) {

    std::vector<double> NL;
    
    // Left patch
    for (int i = number_of_left_patches; i >= 1; --i) {
        double node = Corners[0] - i * L;
        NL.push_back(node);
    }
    // Right patch
    for (int i = 1; i <= number_of_right_patches; ++i) {
        double node = Corners[1] + i * L;
        NL.push_back(node);
    }

    return NL;
}

// 4th function: [ PL ] = Topology( NL , L , Delta );
std::vector<Point> Topology(const std::vector<double>& NL, double L, double Delta) {
    int NoNs = NL.size();    
    int NoP = 0; // point number
    
    std::vector<int> NPL(NoNs);
    std::vector<Point> PL;
    
    for (int n = 0; n < NoNs; n++) {
        double X = NL[n];
        Point newPoint(NoP, X);
        newPoint.Nr = n;
        PL.emplace_back(newPoint);    
        NPL[n] = NoP;
        NoP = NoP + 1;
    }
    
    return PL;
}

// 5th function: [ PL ] = AssignNgbrs( PL , L , Delta );
std::vector<Point> AssignNgbrs(std::vector<Point> PL, double L, double Delta) {
    int NoPs = PL.size();    
    double tol = 1e-8;
    
    int NmaxNgbr = 0;
    
    int Del_by_L = static_cast<int>(floor(Delta / L));
    
    for (signed i = -Del_by_L; i <= Del_by_L; i++) {
        if ((sqrt(i * i) * L < Delta) && (i != 0)) {
            NmaxNgbr = NmaxNgbr + 1;
        }
    }
    
    std::vector<int> neighbors;
    std::vector<double> neighborsX;
    std::vector<double> neighborsx;    
    
    for (int p = 0; p < NoPs; p++) {
        
        neighbors.clear();
        neighborsX.clear();
        neighborsx.clear();
        
        for (int q = 0; q < NoPs; q++) {
            if ((q != p) && (std::abs((PL[p].X - PL[q].X)) <= Delta)) {
                neighbors.emplace_back(q);
                neighborsX.emplace_back(PL[q].X);
                neighborsx.emplace_back(PL[q].x);
            }
        }
        
        PL[p].neighbors = neighbors;
        PL[p].neighborsx = neighborsx;
        PL[p].neighborsX = neighborsX;
                
        int NNgbr = neighbors.size();
                
        double Amax = 2 * Delta;
        double AV = static_cast<double>(NNgbr + 1) / (NmaxNgbr + 1) * Amax;
        
        PL[p].NI = NNgbr;
        PL[p].AV = AV;
    }
    
    return PL;
}

// 6th function: [ PL ] = AssignVols( Corners , PL , L , MATpars );
std::vector<Point> AssignVols(const std::vector<double>& Corners, std::vector<Point> PL, double L) {
    int NoPs = PL.size();
    int PD = PL[0].PD;
    
    double tol = 1e-4;
    
    double A = Corners[0]; // bottom left
    double B = Corners[1]; // bottom right
    
    for (int p = 0; p < NoPs; p++) {
        double X = PL[p].X;
        
        double alpha;
        
        if ((X - A) < (-tol) || (X - B) > (tol)) {
            alpha = 0;
        } else if (abs(X - A) < tol || abs(X - B) < tol) {
            alpha = 1.0 / 2.0;
        } else {
            alpha = 1;
        }
        
        double V = alpha * L;
        
        PL[p].Vol = V;
    }
    
    return PL;
}

// 7th function: [ PL ] = SetMaterial( PL , L , Delta , MATpars );
std::vector<Point> SetMaterial(const std::vector<Point>& inp, double L, double Delta, double& MatPars) {
    std::vector<Point> PL = inp;
    
    int NoPs = PL.size();
    for (int p = 0; p < NoPs; p++) {
        int mat = 1; // Matlab uses 1-based indexing
        
        PL[p].L = L;
        PL[p].Delta = Delta;
        PL[p].Mat = mat;
        PL[p].MatPars = MatPars;
    }
    
    return PL;
}

// 8th function: FF = Compute_FF(PD,d,DEFflag);
double Compute_FF(int PD, double d, const std::string& DEFflag) {
    return (1.0 + d);
}

// Helper function: FreeAllPoints
std::vector<Point> FreeAllPoints(std::vector<Point> PL) {    
    // free all points with no force ... prescribe homogeneous Neumann BC on all points
    int NoPs = PL.size();
    for (int i = 0; i < NoPs; i++) {
        PL[i].BCflg = 1;
        PL[i].BCval = 0.0;
    }
    
    return PL;
}

// Helper function: AssignGlobalDOF
std::pair<std::vector<Point>, int> AssignGlobalDOF(std::vector<Point> PL, int& DOCs_out) {
    int DOFs = 0;
    int DOCs = 0;

    for (auto& point : PL) {
        if (point.BCflg == 1) {
            DOFs++;
            point.DOF = DOFs;
            point.DOC = 0;
        } else {
            DOCs++;
            point.DOC = DOCs;
            point.DOF = 0;
        }
    }

    DOCs_out = DOCs;
    return std::make_pair(PL, DOFs);
}

// 9th function: [ PL , DOFs ] = AssignBCs( Corners , PL , FF );
std::pair<std::vector<Point>, int> AssignBCs(const std::vector<double>& Corners, std::vector<Point> PL, const double& d) {
    int NoPs = PL.size();
    int PD = PL[0].PD;
    double FF = 1.0 + d;
    double A = Corners[0]; // bottom left
    double B = Corners[1]; // bottom right
    int dummy_DOCs = 0;
    PL = FreeAllPoints(PL);

    double tol = 1e-6;
    int idx = 0;
    for (int i = 0; i < NoPs; i++) {
        double X = PL[i].X;

        if ((X < 0.0)) {
            int BCflg = 0;
            double BCval = 0.0;
                        
            PL[i].BCflg = BCflg;
            PL[i].BCval = BCval;
            PL[i].Flag = "Patch";
        }
        else if ((X > 1.0))
        {
            int BCflg = 0;
            double BCval = (FF * X) - X;

            PL[i].BCflg = BCflg;
            PL[i].BCval = BCval;
            PL[i].Flag = "Right Patch";
            idx++;
        }
        else if ((X > 0.0 || X < 1.0))
        {
            //PL[i].BCflg = BCflg;
            PL[i].BCval = 0.0;
            PL[i].Flag = "Point";
        }
    }
    auto result = AssignGlobalDOF(PL, dummy_DOCs);
    return result;
}

void calculate_rk(std::vector<Point>& PL, double C1, double delta, double nn)
{
    double Vh = 2 * delta;
    int NoPs = PL.size();

    for (int i = 0; i < NoPs; i++)
    {
        PL[i].residual = 0.0;
        PL[i].psi = 0.0;
        double JI = Vh / PL[i].NI;

        std::vector<int> neighborsE = PL[i].neighbors;
        std::vector<double> neighborsEx = PL[i].neighborsx;
        std::vector<double> neighborsEX = PL[i].neighborsX;

        neighborsE.push_back(PL[i].Nr);
        neighborsEx.push_back(PL[i].x);
        neighborsEX.push_back(PL[i].X);

        const int NNgbrE = neighborsE.size();
        PL[i].stiffness.clear();
        PL[i].stiffness.resize(NNgbrE, 0.0);

        for (size_t j = 0; j < PL[i].NI; j++) {
            double XiI = PL[i].neighborsX[j] - PL[i].X;
            double xiI = PL[i].neighborsx[j] - PL[i].x;
            double LL = std::abs(XiI);

            if (LL > 1e-8) {
                hyperdual xiI_HD(xiI, 1.0, 1.0, 0.0);
                hyperdual ll = fabs(xiI_HD);

                // Lambda power-law stretch: s = (1/nn) * [ (l/LL)^nn - 1 ]
                hyperdual s = (1.0 / nn) * (pow(ll / LL, nn) - 1.0);

                hyperdual psi = 0.5 * C1 * LL * s * s;

                PL[i].psi += psi.real();
                PL[i].residual += psi.eps1() * JI;

                for (int b = 0; b < NNgbrE; b++) {
                    double K_factor = 0.0;
                    if (PL[i].neighbors[j] == neighborsE[b]) K_factor += 1.0;
                    if (PL[i].Nr == neighborsE[b]) K_factor -= 1.0;
                    PL[i].stiffness[b] += psi.eps1eps2() * JI * K_factor;
                }
            }
        }
    }
}



// Assemble residual or stiffness matrix
void assembly(const std::vector<Point>& point_list, int DOFs, int DOCs, Eigen::VectorXd& R, Eigen::SparseMatrix<double>& K
    , Eigen::SparseMatrix<double>& Kuu, Eigen::SparseMatrix<double>& Kpu, Eigen::SparseMatrix<double>& Kpp, const std::string& flag)
{
    if (flag == "residual") {
        // Reset residual vector
        R.setZero();

        // Assemble residual
        for (const auto& point : point_list) {
            double R_P = point.residual;
            double BCflg = point.BCflg;
            int DOF = point.DOF;
            if (BCflg == 1) {
                R(DOF - 1) += R_P;
            }
        }

        //std::cout << "Size of the residual vector is: " << R.size() << "\n";
        //std::cout << "\nResidual Vector R:\n" << R << std::endl;
    }
    else if (flag == "stiffness") {
        // Reset stiffness matrix
        K.setZero();
        std::vector<Eigen::Triplet<double>> triplets;

        for (const auto& point : point_list) {
            std::vector<int> neighborsE = point.neighbors;
            neighborsE.push_back(point.Nr);

            for (size_t q = 0; q < neighborsE.size(); ++q) {
                int nbr_idx = neighborsE[q];
                double Kval = point.stiffness[q];

                // Use actual point index (i.e., row = point.Nr)
                triplets.emplace_back(point.Nr, nbr_idx, Kval);
            }
        }

        int N = point_list.size();
        K.resize(N, N);
        K.setFromTriplets(triplets.begin(), triplets.end());

        Eigen::MatrixXd A = Eigen::MatrixXd(K);

        //std::cout << "\nStiffness matrix size: " << A.rows() << " x " << K.cols() << std::endl;
        //std::cout << "\nStiffness Matrix K:\n" << A << std::endl;
        // === Now extract Kuu, Kpu, Kpp ===
        std::vector<int> unknown_indices;     // free DOFs
        std::vector<int> prescribed_indices;  // constrained DOFs

        for (int i = 0; i < point_list.size(); ++i) {
            if (point_list[i].BCflg == 1)
                unknown_indices.push_back(i);         // global matrix row = i
            else if (point_list[i].BCflg == 0)
                prescribed_indices.push_back(i);      // global matrix row = i
        }

        int Nu = unknown_indices.size();
        int Np = prescribed_indices.size();

        // Prepare triplets for Kuu, Kpu, Kpp
        std::vector<Eigen::Triplet<double>> Kuu_trips, Kpu_trips, Kpp_trips;

        for (int i = 0; i < Nu; ++i)
            for (int j = 0; j < Nu; ++j)
                Kuu_trips.emplace_back(i, j, A(unknown_indices[i], unknown_indices[j]));

        for (int i = 0; i < Np; ++i)
            for (int j = 0; j < Nu; ++j)
                Kpu_trips.emplace_back(i, j, A(prescribed_indices[i], unknown_indices[j]));

        for (int i = 0; i < Np; ++i)
            for (int j = 0; j < Np; ++j)
                Kpp_trips.emplace_back(i, j, A(prescribed_indices[i], prescribed_indices[j]));

        // Resize and assign triplets
        Kuu.resize(Nu, Nu);
        Kpu.resize(Np, Nu);
        Kpp.resize(Np, Np);

        Kuu.setFromTriplets(Kuu_trips.begin(), Kuu_trips.end());
        Kpu.setFromTriplets(Kpu_trips.begin(), Kpu_trips.end());
        Kpp.setFromTriplets(Kpp_trips.begin(), Kpp_trips.end());

        //std::cout << "\nStiffness Matrix Kpu:\n" << Kpu << std::endl;
        //std::cout << "\nStiffness Matrix Kpp:\n" << Kpp << std::endl;
    }
}
// Update points based on displacement or prescribed values
void update_points(std::vector<Point>& PL, double LF, Eigen::VectorXd& dx, const std::string& Update_flag)
{
    int NoPs = PL.size();
    if (Update_flag == "Prescribed") {
        for (int i = 0; i < NoPs; i++) {
            if (PL[i].BCflg == 0) {
                PL[i].x = PL[i].X + (LF * PL[i].BCval);
            }
        }
    }
    else if (Update_flag == "Displacement") {
        for (int i = 0; i < NoPs; i++) {
            if (PL[i].BCflg == 1 && PL[i].DOF > 0) {
                PL[i].x += dx(PL[i].DOF - 1);
            }
        }
    }

    // Update neighbor coordinates by directly accessing updated coordinates
    for (int i = 0; i < NoPs; i++) {
        for (size_t n = 0; n < PL[i].NI; n++) {
            int nbr_idx = PL[i].neighbors[n];
            PL[i].neighborsx[n] = PL[nbr_idx].x;
        }
    }
}