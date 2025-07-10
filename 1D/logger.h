#pragma once
#include <fstream>
#include <iomanip>
#include <string>
#include <filesystem>
#include <chrono>
#include <ctime>

class Logger {
    std::ofstream logfile;

public:
    Logger(const std::string& file_name) {
        std::string path = "log_files/" + file_name + ".log";
        logfile.open(path);
        if (!logfile.is_open()) {
            std::cerr << "Failed to open log file: " << path << std::endl;
        }
    }

    void writeHeader(const std::string& sim_id) {
        logfile << "============================================================\n";
        logfile << "1D PERIDYNAMICS SIMULATION LOG\n";
        logfile << "Simulation ID     : " << sim_id << "\n";
        logfile << "Timestamp         : " << timestamp() << "\n";
        logfile << "============================================================\n\n";
        logfile.flush(); // Ensure immediate write
    }

    void writeParameters(double domain, double L, double Delta, int numPoints, int steps,
                         double C1, double nn, const std::string& flag, double force) {
        logfile << "--- Parameters ---\n";
        logfile << std::fixed << std::setprecision(5);
        logfile << "Domain size       : " << domain << "\n";
        logfile << "Grid spacing (L)  : " << L << "\n";
        logfile << "Horizon (Delta)   : " << Delta << "\n";
        logfile << "# Points          : " << numPoints << "\n";
        logfile << "C1 (material)     : " << C1 << "\n";
        logfile << "Power (nn)        : " << nn << "\n";
        logfile << "Prescribed flag   : " << flag << "\n";
        logfile << "Applied force     : " << force << "\n";
        logfile << "Steps             : " << steps << "\n\n";
        logfile.flush(); // Ensure immediate write
    }

    void writeLoadFactor(double LF) {
        logfile << "\nLoad Factor: " << LF << "\n";
        logfile.flush(); // Ensure immediate write
    }

    void writeConvergence(int iter, double res, double rel) {
        logfile << "  Iter " << iter
                << "  : Residual = " << std::scientific << res
                << ", Relative = " << rel << "\n";
        logfile.flush(); // Ensure immediate write
    }

    void writeConverged(int count) {
        logfile << "  Converged after " << count << " iterations.\n\n";
        logfile.flush(); // Ensure immediate write
    }

    void writeTiming(double sim, double total) {
        logfile << "\n--- Timing ---\n";
        logfile << std::fixed << std::setprecision(6);
        logfile << "Simulation time   : " << sim << " seconds\n";
        logfile << "Total runtime     : " << total << " seconds\n";
        logfile.flush(); // Ensure immediate write
    }

    void writeReactoinForce(double LF, double F_rec_right_patch, double F_rec_patch){
        logfile<<"\t Reaction Force on the RIGHT PATCH at Load Factor    : "<< LF << " is : "<< F_rec_right_patch <<std::endl;
        logfile<<"\t Reaction Force on the PATCH at Load Factor          : "<< LF << " is : "<< F_rec_patch <<std::endl;
        logfile<<"\t Total Reaction force = Rightpatch - Patch = " << (F_rec_right_patch - F_rec_patch)<< std::endl<< std::endl;
 
    }


    void close() {
        logfile.close();
    }

private:
    std::string timestamp() {
        auto now = std::chrono::system_clock::now();
        std::time_t t = std::chrono::system_clock::to_time_t(now);
        char buffer[64];
        std::strftime(buffer, sizeof(buffer), "%Y-%m-%d %H:%M:%S", std::localtime(&t));
        return buffer;
    }
};