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

    void writeParameters(double domain, double L, double Delta, int numPoints, int steps, double C1, double nn, const std::string& flag, double force, double displacement, double patches, double right_patches) {
        logfile << "--- Parameters ---\n";
        logfile << std::fixed << std::setprecision(5);
        logfile << "Domain size          : " << domain << "\n";
        logfile << "Grid spacing (L)     : " << L << "\n";
        logfile << "Horizon (Delta)      : " << Delta << "\n";
        logfile << "No. of Patches       : " << patches << "\n";
        logfile << "No. of Right Patches : " << right_patches << "\n";
        logfile << "# Points             : " << numPoints << "\n";
        logfile << "C1 (material)        : " << C1 << "\n";
        logfile << "Power (nn)           : " << nn << "\n";
        logfile << "Prescribed flag      : " << flag << "\n";
        logfile << "Applied force        : " << force << "\n";
        logfile << "Applied Disp         : " << displacement << "\n";
        logfile << "Steps                : " << steps << "\n\n";
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

    void writeReactoinForce(double LF, double F_rec_right_patch, double F_rec_patch, int H, double nn){
        logfile<<"Reaction Force on the RIGHT PATCH at Load Factor    : "<< LF << " is : "<< F_rec_right_patch <<std::endl;
        logfile<<"Reaction Force on the PATCH at Load Factor          : "<< LF << " is : "<< F_rec_patch <<std::endl;
        logfile<<"Total Reaction force = Rightpatch - Patch = " << (F_rec_right_patch - F_rec_patch)<< std::endl<< std::endl;
        double Force_Diff = (F_rec_right_patch - F_rec_patch);
        double Force_Diff_error = (std::abs(Force_Diff) / (F_rec_right_patch)) * 100;
        logfile << "Force Difference = Right − Left total = " << Force_Diff << "\n";
        logfile << "Force Difference Error % = " << Force_Diff_error << "\n";
        std::ofstream error_csv("csv_files/force_error.csv", std::ios::app);
        if (error_csv.is_open()) {
            error_csv << H << "," << nn << "," << Force_Diff_error << "\n";
        }
        error_csv.close();
    }

    void writePatchForces(int H, double nn, const std::vector<double>& left_residuals, double right_total) {
        logfile << "\n--- Patch Forces ---\n";
        //logfile << "H = " << H << ", NN = " << nn << "\n";
        logfile << "RightPatch total = " << right_total << "\n";

        double left_total = 0.0;
        for (int k = 0; k < left_residuals.size(); ++k) {
            int Xpos = -(k+1);
            double val = left_residuals[k];
            left_total += val;
            logfile << "X=" << Xpos
                    << " Left=" << val 
                    << " Diff=" << (right_total - val) << "\n";
        }        
        logfile << "LeftPatch total = " << left_total << "\n";
        logfile.flush();
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