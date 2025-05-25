#include <iostream>
#include <vector>
#include <string>
#include <cmath>
#include <iomanip>
#include <stdexcept>
#include "Layer.h"
#include "NeuralNetwork.h"


template<typename T, typename U>
void assertTest(bool condition, const std::string& test_name, const T& expected, const U& actual) {
    std::cout << std::fixed << std::setprecision(5);
    if (!condition) {
        std::cerr << "ASSERTION FAILED: " << test_name
                  << " | Expected: " << expected
                  << " | Actual: " << actual << std::endl;
    } else {
        std::cout << "Assertion PASSED: " << test_name
                  << " | Expected: " << expected
                  << " | Actual: " << actual << std::endl;
    }
}

void assertTest(bool condition, const std::string& test_name) {
    if (!condition) {
        std::cerr << "ASSERTION FAILED: " << test_name << std::endl;
    } else {
        std::cout << "Assertion PASSED: " << test_name << std::endl;
    }
}

void print_vector_main(const std::string& name, const std::vector<double>& vec) {
    std::cout << name << ": [ ";
    for (size_t i = 0; i < vec.size(); ++i) {
        std::cout << std::fixed << std::setprecision(5) << vec[i] << (i == vec.size() - 1 ? "" : ", ");
    }
    std::cout << " ]" << std::endl;
}
void print_matrix_main(const std::string& name, const std::vector<std::vector<double>>& matrix) {
    std::cout << name << ": [" << std::endl;
    for (const auto& row : matrix) {
        std::cout << "  [ ";
        for (size_t i = 0; i < row.size(); ++i) {
            std::cout << std::fixed << std::setprecision(5) << row[i] << (i == row.size() - 1 ? "" : ", ");
        }
        std::cout << " ]" << std::endl;
    }
    std::cout << "]" << std::endl;
}



// --- Temporary Test for NeuralNetwork train_one_sample ---
void temporary_test_neural_network_train_one_sample() {
    std::cout << "\n--- Running Temporary NeuralNetwork::train_one_sample Test ---" << std::endl;
    double epsilon = 1e-5; // Increased epsilon slightly for complex calculations

    // Network: 1 input -> 1 hidden neuron (ReLU) -> 1 output neuron (Linear)
    // Input: x = [2.0]
    // Target: y_true = 0.5
    // Learning rate: lr = 0.1

    // Layer 0 (Hidden, ReLU): 1 input, 1 neuron
    // Initial W0 = [0.5], B0 = [0.1]
    // Layer 1 (Output, Linear): 1 input (from L0), 1 neuron
    // Initial W1 = [0.3], B1 = [-0.1]

    // --- Manual Calculation for one step ---
    // **Forward Pass:**
    // L0 Input: x0 = 2.0
    // L0 Z: z0 = W0*x0 + B0 = 0.5*2.0 + 0.1 = 1.0 + 0.1 = 1.1
    // L0 A: a0 = relu(z0) = relu(1.1) = 1.1 (This is input to L1)

    // L1 Input: x1 = a0 = 1.1
    // L1 Z: z1 = W1*x1 + B1 = 0.3*1.1 - 0.1 = 0.33 - 0.1 = 0.23
    // L1 A: a1 = linear(z1) = 0.23 (This is y_pred)
    // y_pred = 0.23

    // **Loss Derivative:**
    // y_true = 0.5
    // dError/dy_pred = y_pred - y_true = 0.23 - 0.5 = -0.27

    // **Backward Pass:**
    // --- Layer 1 (Output, Linear) ---
    // error_signal_from_loss (dError/da1) = -0.27
    // L1 delta (dError/dz1) = (dError/da1) * linear_derivative(z1) = -0.27 * 1.0 = -0.27
    // L1 dW1 = delta1 * x1_cached (a0) = -0.27 * 1.1 = -0.297
    // L1 db1 = delta1 = -0.27
    // L1 W1_new = W1_old - lr * dW1 = 0.3 - 0.1*(-0.297) = 0.3 + 0.0297 = 0.3297
    // L1 B1_new = B1_old - lr * db1 = -0.1 - 0.1*(-0.27) = -0.1 + 0.027 = -0.073
    // L1 error_to_L0 (dError/da0) = W1_old^T * delta1 = 0.3 * (-0.27) = -0.081

    // --- Layer 0 (Hidden, ReLU) ---
    // error_signal_from_L1 (dError/da0) = -0.081
    // L0 delta (dError/dz0) = (dError/da0) * relu_derivative(z0) = -0.081 * relu_derivative(1.1) = -0.081 * 1.0 = -0.081
    // L0 dW0 = delta0 * x0_cached = -0.081 * 2.0 = -0.162
    // L0 db0 = delta0 = -0.081
    // L0 W0_new = W0_old - lr * dW0 = 0.5 - 0.1*(-0.162) = 0.5 + 0.0162 = 0.5162
    // L0 B0_new = B0_old - lr * db0 = 0.1 - 0.1*(-0.081) = 0.1 + 0.0081 = 0.1081

    std::cout << "\n-- Test Case: 1 -> 1 (ReLU) -> 1 (Linear) Network, 1 Sample Train --" << std::endl;
    try {
        std::vector<int> layer_sizes = {1, 1, 1}; // Input:1, Hidden:1, Output:1
        std::vector<std::string> activations = {"relu", "linear"};
        double lr = 0.1;
        double dropout = 0.0;

        Predicting_Close_Price_Using_NN::NeuralNetwork nn(layer_sizes, activations, lr, dropout);

        // Manually set initial weights and biases
        // Layer 0 (Hidden, ReLU)
        nn.layers_[0].weights_ = {{0.5}};
        nn.layers_[0].biases_  = {0.1};
        // Layer 1 (Output, Linear)
        nn.layers_[1].weights_ = {{0.3}};
        nn.layers_[1].biases_  = {-0.1};

        std::cout << "Initial NN state:" << std::endl;
        std::cout << " L0 W: " << nn.layers_[0].weights_[0][0] << ", B: " << nn.layers_[0].biases_[0] << std::endl;
        std::cout << " L1 W: " << nn.layers_[1].weights_[0][0] << ", B: " << nn.layers_[1].biases_[0] << std::endl;

        std::vector<double> x_sample = {2.0};
        double y_true_sample = 0.5;

        // Perform one training step
        nn.train_one_sample(x_sample, y_true_sample);

        std::cout << "NN state after one train_one_sample call:" << std::endl;
        std::cout << " L0 W: " << nn.layers_[0].weights_[0][0] << ", B: " << nn.layers_[0].biases_[0] << std::endl;
        std::cout << " L1 W: " << nn.layers_[1].weights_[0][0] << ", B: " << nn.layers_[1].biases_[0] << std::endl;

        // Expected new weights and biases after one step
        double expected_L0_W_new = 0.5162;
        double expected_L0_B_new = 0.1081;
        double expected_L1_W_new = 0.3297;
        double expected_L1_B_new = -0.073;

        // Verify Layer 0 (Hidden)
        assertTest(std::abs(nn.layers_[0].weights_[0][0] - expected_L0_W_new) < epsilon,
                   "train_one_sample: L0 Updated Weight W[0][0]", expected_L0_W_new, nn.layers_[0].weights_[0][0]);
        assertTest(std::abs(nn.layers_[0].biases_[0] - expected_L0_B_new) < epsilon,
                   "train_one_sample: L0 Updated Bias B[0]", expected_L0_B_new, nn.layers_[0].biases_[0]);

        // Verify Layer 1 (Output)
        assertTest(std::abs(nn.layers_[1].weights_[0][0] - expected_L1_W_new) < epsilon,
                   "train_one_sample: L1 Updated Weight W[0][0]", expected_L1_W_new, nn.layers_[1].weights_[0][0]);
        assertTest(std::abs(nn.layers_[1].biases_[0] - expected_L1_B_new) < epsilon,
                   "train_one_sample: L1 Updated Bias B[0]", expected_L1_B_new, nn.layers_[1].biases_[0]);

    } catch (const std::exception& e) {
        std::cerr << "NeuralNetwork train_one_sample Test failed with exception: " << e.what() << std::endl;
        assertTest(false, "NN train_one_sample Test: Execution without exceptions");
    }

}
// --- End of NeuralNetwork train_one_sample Test ---


int main() {
    std::cout << "Initializing main..." << std::endl;

    std::cout << "\nRunning NeuralNetwork train_one_sample Tests " << std::endl;
    temporary_test_neural_network_train_one_sample();
    std::cout << "\nTemporary NeuralNetwork train_one_sample Tests Complete" << std::endl << std::endl;
    // --- End of Temporary Test Calls ---


    return 0;
}

