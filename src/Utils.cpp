#include "Utils.h"
#include <stdexcept>

namespace Utils {

    double random_double(double min, double max) {
        static std::mt19937 random_engine(std::random_device{}()); // Mersenne Twister engine seeded with a random device
        std::uniform_real_distribution<double> distribution(min, max);
        return distribution(random_engine);
    }

    std::vector<double> dot(const std::vector<std::vector<double>>& matrix,
                            const std::vector<double>& vector) {
        if (matrix.empty()) {
            return {}; // Empty if matrix is empty
        }

        size_t num_rows = matrix.size();
        size_t num_cols = matrix[0].size();

        if (num_cols != vector.size()) {
            throw std::invalid_argument("Matrix inner dimension (" + std::to_string(num_cols) +
                                        ") must match vector size (" + std::to_string(vector.size()) + ").");
        }

        std::vector<double> result(num_rows, 0.0);

        for (size_t i = 0; i < num_rows; ++i) {
            if (matrix[i].size() != num_cols) {
                throw std::invalid_argument("Matrix has inconsistent column count at row " + std::to_string(i));
            }
            for (size_t j = 0; j < num_cols; ++j) {
                result[i] += matrix[i][j] * vector[j];
            }
        }
        return result;
    }

}