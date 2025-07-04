#include "cli.h"
#include <iostream>
#include <cstdlib>
#include <cstring>

CLIOptions parseArguments(int argc, char* argv[]) {
    CLIOptions options;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        auto next = [&]() -> char* {
            if (i + 1 < argc) return argv[++i];
            std::cerr << "Missing value for " << arg << "\n"; std::exit(1);
        };

        if (arg == "--domain") options.domain_size = std::atof(next());
        else if (arg == "--delta") options.Delta = std::atof(next());
        else if (arg == "--spacing") options.L = std::atof(next());
        else if (arg == "--patches") options.number_of_patches = std::atoi(next());
        else if (arg == "--rpatches") options.number_of_right_patches = std::atoi(next());
        else if (arg == "--C1") options.C1 = std::atof(next());
        else if (arg == "--nn") options.nn = std::atof(next());
        else if (arg == "--d") options.d = std::atof(next());
        else if (arg == "--force") options.F_prescribed = std::atof(next());
        else if (arg == "--flag") options.Prescribed_Flag = next();
        else if (arg == "--steps") options.steps = std::atoi(next());
        else if (arg == "--tol") options.tol = std::atof(next());
        else if (arg == "--DEFflag") options.DEFflag = next();
        else {
            std::cerr << "Unknown option: " << arg << "\n";
            std::exit(1);
        }
    }

    return options;
}
