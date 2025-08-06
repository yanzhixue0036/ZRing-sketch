#include <fstream>
#include <string>
#include <vector>
#include <random>
#include <cmath>
#include <iostream>
#include <filesystem>
#include <getopt.h> // For long options
#include "cnpy.h"
#include "ZRing.h"
#include "qsketch.h"
#include "qsketch_dyn.h"

using namespace std;
namespace fs = std::filesystem;

enum MethodType {
    ZRing_Method,
    QSketch_Method,
    QSketchDyn_Method
};

std::pair<double, double> qs_proc(double sketch_size, double* data, int data_num, int register_size, int size_a, int z_size, vector<double>& proportions, MethodType method) {
    vector<PackedVector> g;
    g.reserve(sketch_size);
    for (int i = 0; i < sketch_size; i++) g.emplace_back(z_size, size_a);

    double aare;
    double actual, estimated, update_time, estimated_time;

    if (method == ZRing_Method) {
        ZRing alg(sketch_size, register_size, z_size);
        alg.Update(data, data_num, g, 0);
        aare = alg.Count();
        alg.Update(data, data_num, g, 0, false);
        aare = alg.Count();
        actual = alg.Actual();
        estimated = alg.Estimated_card();
        update_time = alg.update_time;
        estimated_time = alg.estimation_time;
    } else if (method == QSketch_Method) {
        QSketch alg(sketch_size*z_size, register_size);
        alg.Update(data, data_num, g, 0);
        aare = alg.Count();
        alg.Update(data, data_num, g, 0, false);
        aare = alg.Count();
        actual = alg.Actual();
        estimated = alg.Estimated_card();
        update_time = alg.update_time;
        estimated_time = alg.estimation_time;
    } else if (method == QSketchDyn_Method) {
        QDyn alg(sketch_size*z_size, register_size, rand());
        alg.Update(data, data_num, g, 0);
        aare = alg.Count();
        alg.Update(data, data_num, g, 0, false);
        aare = alg.Count();
        actual = alg.Actual();
        estimated = alg.Estimated_card();
        update_time = alg.update_time;
        estimated_time = alg.estimation_time;
    }

    for (int j = 0; j < size_a; j++) {
        int zero_count = 0;
        for (int i = 0; i < sketch_size; i++) {
            if (g[i].get(j) == 0) {
                zero_count++;
            }
        }
        double zero_ratio = static_cast<double>(zero_count) / sketch_size;
        proportions[j] = std::max(0.0, (zero_ratio * pow(2, z_size) - 1) / (pow(2, z_size) - 1));
    }

    proportions.push_back(std::log(sketch_size));
    proportions.push_back(z_size);
    proportions.push_back(actual);
    proportions.push_back(estimated);
    proportions.push_back(update_time);
    proportions.push_back(estimated_time);

    return {estimated, actual};
}

vector<double> process_file(const string& file_name, int sketch_size, int register_size, int z_size, int size_a, MethodType method) {
    ifstream file(file_name);
    if (!file) {
        cerr << "Failed to open file: " << file_name << endl;
        return {};
    }

    vector<double> data;
    double value;
    while (file >> value) {
        data.push_back(value);
    }
    file.close();

    vector<double> proportions(size_a, 0.0);
    qs_proc(sketch_size, data.data(), data.size(), register_size, size_a, z_size, proportions, method);
    return proportions;
}

void print_usage() {
    cout << "Usage: ./main [--method ZRing|QSketch|QSketchDyn] [--data FILE] [--repeat R] [--memory KB]" << endl;
}

int main(int argc, char* argv[]) {
    string dataset_path = "./data.txt";
    string output_file = "./ZRingset/test.npy";
    int R = 10;
    int register_size = 8;
    int z_size = 3;
    int memory = 1 * 1024 * 8; // KB
    MethodType method = ZRing_Method;

    // Parse arguments
    const struct option long_opts[] = {
        {"method", required_argument, nullptr, 'm'},
        {"data", required_argument, nullptr, 'd'},
        {"repeat", required_argument, nullptr, 'r'},
        {"memory", required_argument, nullptr, 'k'},
        {nullptr, 0, nullptr, 0}
    };

    int opt;
    while ((opt = getopt_long(argc, argv, "m:d:r:s:g:k:", long_opts, nullptr)) != -1) {
        switch (opt) {
            case 'm':
                if (string(optarg) == "ZRingDME") method = ZRing_Method;
                else if (string(optarg) == "QSketch") method = QSketch_Method;
                else if (string(optarg) == "QSketchDME") method = QSketchDyn_Method;
                else {
                    cerr << "Unknown method: " << optarg << endl;
                    return 1;
                }
                break;
            case 'd': dataset_path = optarg; break;
            case 'r': R = stoi(optarg); break;
            case 'k': memory = stoi(optarg) * 1024 * 8; break;
            default: print_usage(); return 1;
        }
    }

    int size_a = pow(2, register_size);
    int sketch_size = memory / z_size / size_a;

    vector<vector<double>> results;

    for (int i = 0; i < R; ++i) {
        auto result = process_file(dataset_path, sketch_size, register_size, z_size, size_a, method);
        if (!result.empty()) {
            results.push_back(result);
        }
        cout << i + 1 << "/" << R << " finished" << endl;
    }

    if (!results.empty()) {
        size_t rows = results.size();
        size_t cols = results[0].size();
        vector<double> combined_data;
        for (const auto& row : results) {
            combined_data.insert(combined_data.end(), row.begin(), row.end());
        }
        vector<unsigned long> shape = {rows, cols};
        if(method == ZRing_Method) cnpy::npy_save(output_file, combined_data.data(), shape, "w");

        double total_relative_error = 0.0, total_time = 0.0, total_esti_time = 0.0;
        int valid_count = 0;
        for (const auto& row : results) {
            double actual = row[row.size() - 4];
            double estimate = row[row.size() - 3];
            double time = row[row.size() - 2];
            double esti_time = row[row.size() - 1];

            if (actual > 1e-6) {
                total_relative_error += std::abs(estimate - actual) / actual;
                total_time += time;
                total_esti_time += esti_time;
                valid_count++;
            }
        }

        double AARE = total_relative_error / valid_count;
        cout << "AARE: " << AARE << endl;
        cout << "Update Time: " << total_time / valid_count << endl;
        cout << "Estimation Time: " << total_esti_time / valid_count << endl;
    }

    cout << "Done. Saved to: " << output_file << endl;
    return 0;
}
