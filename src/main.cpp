#include <iostream>
#include <vector>
#include <string>
#include <stdexcept>
#include "Utils.h"
#include "Activations.h"
#include <cmath>
#include <iomanip>
#include "Loss.h"

// Vector
void print_vector(const std::string& name, const std::vector<double>& vec) {
    std::cout << name << ": [ ";
    for (size_t i = 0; i < vec.size(); ++i) {
        std::cout << vec[i] << (i == vec.size() - 1 ? "" : ", ");
    }
    std::cout << " ]" << std::endl;
}

//Simple assertion function for temporary testing.
template<typename T>
void assertTest(bool condition, const std::string& test_name, T expected, T actual) {
    std::cout << std::fixed << std::setprecision(8); // Consistent float output
    if (!condition) {
        std::cerr << "ASSERTION FAILED: " << test_name
                  << " | Expected: " << expected
                  << " | Actual: " << actual << std::endl;
    } else {
        std::cout << "Assertion PASSED: " << test_name
                  << " | Value: " << actual << std::endl;
    }
}

// --- Temporary Test Functions for Activations ---
void temporary_test_activations() {
    std::cout << "\nRunning Activations Tests" << std::endl;

    double val_pos = 2.0;
    double val_neg = -3.0;
    double val_zero = 0.0;
    double epsilon = 1e-9; // Comparing doubles

    // Test ReLU
    assertTest(std::abs(Activations::relu(val_pos) - 2.0) < epsilon, "Activations::relu(positive)", 2.0, Activations::relu(val_pos));
    assertTest(std::abs(Activations::relu(val_neg) - 0.0) < epsilon, "Activations::relu(negative)", 0.0, Activations::relu(val_neg));
    assertTest(std::abs(Activations::relu(val_zero) - 0.0) < epsilon, "Activations::relu(zero)", 0.0, Activations::relu(val_zero));

    // Test ReLU Derivative (assuming input is pre-activation value)
    assertTest(std::abs(Activations::relu_derivative(val_pos) - 1.0) < epsilon, "Activations::relu_derivative(positive input)", 1.0, Activations::relu_derivative(val_pos));
    assertTest(std::abs(Activations::relu_derivative(val_neg) - 0.0) < epsilon, "Activations::relu_derivative(negative input)", 0.0, Activations::relu_derivative(val_neg));
    assertTest(std::abs(Activations::relu_derivative(val_zero) - 0.0) < epsilon, "Activations::relu_derivative(zero input)", 0.0, Activations::relu_derivative(val_zero));

    // Test Sigmoid
    double sig_val_pos = Activations::sigmoid(val_pos);
    double sig_val_neg = Activations::sigmoid(val_neg);
    double sig_val_zero = Activations::sigmoid(val_zero);
    assertTest(sig_val_pos > 0.5 && sig_val_pos < 1.0, "Activations::sigmoid(positive) range", true, sig_val_pos > 0.5 && sig_val_pos < 1.0);
    assertTest(std::abs(sig_val_pos - (1.0 / (1.0 + std::exp(-val_pos)))) < epsilon, "Activations::sigmoid(positive) value", (1.0 / (1.0 + std::exp(-val_pos))), sig_val_pos);
    assertTest(sig_val_neg > 0.0 && sig_val_neg < 0.5, "Activations::sigmoid(negative) range", true, sig_val_neg > 0.0 && sig_val_neg < 0.5);
    assertTest(std::abs(sig_val_neg - (1.0 / (1.0 + std::exp(-val_neg)))) < epsilon, "Activations::sigmoid(negative) value", (1.0 / (1.0 + std::exp(-val_neg))), sig_val_neg);
    assertTest(std::abs(sig_val_zero - 0.5) < epsilon, "Activations::sigmoid(zero)", 0.5, sig_val_zero);

    // Test Sigmoid Derivative (takes sigmoid's output as input)
    double sig_deriv_val_pos = Activations::sigmoid_derivative(sig_val_pos);
    double sig_deriv_val_neg = Activations::sigmoid_derivative(sig_val_neg);
    double sig_deriv_val_zero = Activations::sigmoid_derivative(sig_val_zero);
    assertTest(std::abs(sig_deriv_val_pos - (sig_val_pos * (1.0 - sig_val_pos))) < epsilon, "Activations::sigmoid_derivative(from positive output)", (sig_val_pos * (1.0 - sig_val_pos)), sig_deriv_val_pos);
    assertTest(std::abs(sig_deriv_val_neg - (sig_val_neg * (1.0 - sig_val_neg))) < epsilon, "Activations::sigmoid_derivative(from negative output)", (sig_val_neg * (1.0 - sig_val_neg)), sig_deriv_val_neg);
    assertTest(std::abs(sig_deriv_val_zero - 0.25) < epsilon, "Activations::sigmoid_derivative(from zero output)", 0.25, sig_deriv_val_zero);


    // Test Linear
    assertTest(std::abs(Activations::linear(val_pos) - val_pos) < epsilon, "Activations::linear(positive)", val_pos, Activations::linear(val_pos));
    assertTest(std::abs(Activations::linear(val_neg) - val_neg) < epsilon, "Activations::linear(negative)", val_neg, Activations::linear(val_neg));
    assertTest(std::abs(Activations::linear(val_zero) - val_zero) < epsilon, "Activations::linear(zero)", val_zero, Activations::linear(val_zero));

    // Test Linear Derivative
    assertTest(std::abs(Activations::linear_derivative(val_pos) - 1.0) < epsilon, "Activations::linear_derivative(positive input)", 1.0, Activations::linear_derivative(val_pos));
    assertTest(std::abs(Activations::linear_derivative(val_neg) - 1.0) < epsilon, "Activations::linear_derivative(negative input)", 1.0, Activations::linear_derivative(val_neg));
    assertTest(std::abs(Activations::linear_derivative(val_zero) - 1.0) < epsilon, "Activations::linear_derivative(zero input)", 1.0, Activations::linear_derivative(val_zero));

}
// --- End of Activations Test Functions ---

// --- Temporary Test Functions for Loss ---
void temporary_test_loss() {
    std::cout << "\n--- Running Temporary Loss Tests ---" << std::endl;
    double epsilon = 1e-9; // Comparing doubles

    // Test Case 1: Perfect prediction
    double y_true1 = 10.0;
    double y_pred1 = 10.0;
    double expected_mse1 = 0.0;
    double actual_mse1 = Loss::mean_squared_error(y_true1, y_pred1);
    assertTest(std::abs(actual_mse1 - expected_mse1) < epsilon, "Loss::mean_squared_error (perfect prediction)", expected_mse1, actual_mse1);

    double expected_mse_deriv1 = 0.0;
    double actual_mse_deriv1 = Loss::mean_squared_error_derivative(y_true1, y_pred1);
    assertTest(std::abs(actual_mse_deriv1 - expected_mse_deriv1) < epsilon, "Loss::mean_squared_error_derivative (perfect prediction)", expected_mse_deriv1, actual_mse_deriv1);

    // Test Case 2: Prediction is higher
    double y_true2 = 5.0;
    double y_pred2 = 7.0;
    double expected_mse2 = 0.5 * std::pow(7.0 - 5.0, 2);
    double actual_mse2 = Loss::mean_squared_error(y_true2, y_pred2);
    assertTest(std::abs(actual_mse2 - expected_mse2) < epsilon, "Loss::mean_squared_error (prediction higher)", expected_mse2, actual_mse2);

    double expected_mse_deriv2 = 2.0;
    double actual_mse_deriv2 = Loss::mean_squared_error_derivative(y_true2, y_pred2);
    assertTest(std::abs(actual_mse_deriv2 - expected_mse_deriv2) < epsilon, "Loss::mean_squared_error_derivative (prediction higher)", expected_mse_deriv2, actual_mse_deriv2);

    // Test Case 3: Prediction is lower
    double y_true3 = 12.0;
    double y_pred3 = 8.0;
    double expected_mse3 = 0.5 * std::pow(8.0 - 12.0, 2);
    double actual_mse3 = Loss::mean_squared_error(y_true3, y_pred3);
    assertTest(std::abs(actual_mse3 - expected_mse3) < epsilon, "Loss::mean_squared_error (prediction lower)", expected_mse3, actual_mse3);

    double expected_mse_deriv3 = -4.0;
    double actual_mse_deriv3 = Loss::mean_squared_error_derivative(y_true3, y_pred3);
    assertTest(std::abs(actual_mse_deriv3 - expected_mse_deriv3) < epsilon, "Loss::mean_squared_error_derivative (prediction lower)", expected_mse_deriv3, actual_mse_deriv3);

}
// --- End of Loss Test Functions ---


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

    // --- Temporary Tests for Activations ---
    std::cout << "\nTesting Activations" << std::endl;
    temporary_test_activations();
    std::cout << "\nActivations Tests Completed" << std::endl << std::endl;
    // --- End of Activations Test Calls ---

    // -- Temporary Tests for Loss
    std::cout << "\nTesting Loss" << std::endl;
    temporary_test_loss();
    std::cout << "\nLoss Tests Complete" << std::endl << std::endl;
    // --- End of Temporary Test Loss ---



    return 0;
}