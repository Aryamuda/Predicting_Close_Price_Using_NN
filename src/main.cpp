#include <iostream>
#include <vector>
#include <string>
#include <stdexcept>
#include "Utils.h"

// Vector
void print_vector(const std::string& name, const std::vector<double>& vec) {
    std::cout << name << ": [ ";
    for (size_t i = 0; i < vec.size(); ++i) {
        std::cout << vec[i] << (i == vec.size() - 1 ? "" : ", ");
    }
    std::cout << " ]" << std::endl;
}

int main() {
    std::cout << "Initializing main..." << std::endl;
    std::cout << "Performing Utils functions..." << std::endl;

    // Utils::random_double
    std::cout << "\n--- Utils::random_double ---" << std::endl;
    double min_r = 1.0;
    double max_r = 10.0;
    std::cout << "Generating 5 random numbers between " << min_r << " and " << max_r << ":" << std::endl;
    for (int i = 0; i < 5; ++i) {
        double r_val = Utils::random_double(min_r, max_r);
        std::cout << "Random value " << i + 1 << ": " << r_val << std::endl;
    }


    // Utils::dot
    std::cout << "\n---Utils::dot ---" << std::endl;
    std::vector<std::vector<double>> matrix1 = {{1.0, 2.0, 3.0}, {4.0, 5.0, 6.0}};
    std::vector<double> vector1 = {1.0, 2.0, 3.0};
    std::cout << "Matrix1: {{1,2,3},{4,5,6}}" << std::endl;
    std::cout << "Vector1: {1,2,3}" << std::endl;
    try {
        std::vector<double> result1 = Utils::dot(matrix1, vector1);
        print_vector("Result", result1);
    } catch (const std::exception& e) {
        std::cerr << "Error during Utils::dot call: " << e.what() << std::endl;
    }


    std::cout << "\nUtils Completed." << std::endl;

    return 0;
}