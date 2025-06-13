//
// Created by srini on 22/04/2025.
//

#ifndef POINTS_H
#define POINTS_H

#pragma once
#include <vector>
#include <cmath>
#include <algorithm>
#include <string>
#include <iostream>

// Point class definition to match Matlab structure
class Point {
public:
    int Nr;
    int PD;
    double X;
    double x;
    std::vector<int> neighbors;
    std::vector<double> neighborsx;
    std::vector<double> neighborsX;
    int NI;
    int NInII;
    double AV;
    double Vol;
    double L;
    double Delta;
    int Mat;
    double MatPars;
    int BCflg;
    double BCval;
    int DOF;
    std::string Flag;
    double psi;
    double residual;
    std::vector<double> stiffness;

    
    // Constructor
    Point(int id, const double& position) : Nr(id), X(position) {
        PD = 1;
        x = X; // assuming x is same as X initially
        NI = 0;
        NInII = 0;
        AV = 0.0;
        Vol = 0.0;
        L = 0.0;
        Delta = 0.0;
        Mat = 0;
    }
    
    Point() : Nr(0), PD(0), NI(0), NInII(0), AV(0.0), Vol(0.0), L(0.0), Delta(0.0), Mat(0), Flag("") {}
};

std::vector<double> Compute_Corners(double SiZe);
std::vector<double> Mesh(const std::vector<double>& Corners, double L) ;
bool PatchNode(const double& node, const std::vector<double>& Corners) ;
std::vector<double> Patch(const std::vector<double>& Corners, double L, double Delta) ;
std::vector<Point> Topology(const std::vector<double>& NL, double L, double Delta) ;
std::vector<Point> AssignNgbrs(std::vector<Point> PL, double L, double Delta) ;
std::vector<Point> AssignVols(const std::vector<double>& Corners, std::vector<Point> PL, double L) ;
std::vector<Point> SetMaterial(const std::vector<Point>& inp, double L, double Delta, double& MatPars) ;
std::pair<std::vector<Point>, int> AssignGlobalDOF(std::vector<Point> PL) ;
std::pair<std::vector<Point>, int> AssignBCs(const std::vector<double>& Corners, std::vector<Point> PL, const double& FF) ;
void calculate_rk(std::vector<Point>& point_list, double C1, double delta, double nn);
void assembly(const std::vector<Point>& point_list, int DOFs, Eigen::VectorXd& R, Eigen::SparseMatrix<double>& K, const std::string& flag);
void update_points(std::vector<Point>& point_list, double LF, Eigen::VectorXd& dx, const std::string& Update_flag);


#endif //POINTS_H
