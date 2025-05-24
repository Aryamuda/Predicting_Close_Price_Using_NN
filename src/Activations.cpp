#include "Activations.h"
#include <cmath>

namespace Activations {

    double relu(double x) {
        return (x > 0.0) ? x : 0.0;
    }

    double relu_derivative(double x) {
        // Note: The derivative at x=0 is undefined.
        return (x > 0.0) ? 1.0 : 0.0;
    }

    double sigmoid(double x) {
        return 1.0 / (1.0 + std::exp(-x));
    }

    double sigmoid_derivative(double activated_output) {
        // Assumes activated_output = sigmoid(x)
        return activated_output * (1.0 - activated_output);
    }

    double linear(double x) {
        return x;
    }

    double linear_derivative(double x) {
        // It's kept for a consistent function signature with other derivative functions.
        (void)x; // Suppress unused parameter warning if compiler flags are strict
        return 1.0;
    }

}
