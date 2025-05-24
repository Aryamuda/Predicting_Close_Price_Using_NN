#ifndef UTILS_HPP
#define UTILS_HPP

#include <vector>

namespace Utils {

    double random_double(double min, double max);
    std::vector<double> dot(const std::vector<std::vector<double>>& matrix,
                            const std::vector<double>& vector);


}
#endif