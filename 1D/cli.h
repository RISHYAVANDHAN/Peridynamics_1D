#ifndef CLI_H
#define CLI_H

#include <string>

struct CLIOptions {
    double domain_size = 10.0;
    double Delta = 0.00301;
    double L = 0.001;
    int number_of_patches = 3;
    int number_of_right_patches = 1;
    double C1 = 0.5;
    double nn = 2.0;
    double d = 0.1;
    double F_prescribed = 1.0;
    std::string Prescribed_Flag = "Force";  // or "Displacement"
    int steps = 10000;
    double tol = 1e-10;
    std::string DEFflag = "EXT";
};

CLIOptions parseArguments(int argc, char* argv[]);

#endif // CLI_H
