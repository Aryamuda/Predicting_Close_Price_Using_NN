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

// Helper to print vectors for debugging
void print_vector_main(const std::string& name, const std::vector<double>& vec) {
    std::cout << name << ": [ ";
    for (size_t i = 0; i < vec.size(); ++i) {
        std::cout << std::fixed << std::setprecision(5) << vec[i] << (i == vec.size() - 1 ? "" : ", ");
    }
    std::cout << " ]" << std::endl;
}



// --- Temporary Test for NeuralNetwork Predict ---
void temporary_test_neural_network_predict() {
    std::cout << "\n--- Running Temporary NeuralNetwork::predict Test ---" << std::endl;
    double epsilon = 1e-7;

    // Test Case 1: Simple 2-layer network (Input -> Output)
    // Input (2) -> Layer 0 (2 neurons, ReLU) -> Layer 1 (1 neuron, Linear)
    std::cout << "\n-- Test Case 1: 2 -> 2 (ReLU) -> 1 (Linear) Network --" << std::endl;
    try {
        std::vector<int> layer_sizes = {2, 2, 1}; // Input:2, Hidden:2, Output:1
        std::vector<std::string> activations = {"relu", "linear"};
        double lr = 0.01;
        double dropout = 0.0;

        Predicting_Close_Price_Using_NN::NeuralNetwork nn(layer_sizes, activations, lr, dropout);

        // Manually set weights and biases for predictable output
        // Layer 0 (Hidden, ReLU, 2 inputs, 2 neurons)
        if (nn.get_num_layers() > 0) {
            nn.layers_[0].weights_ = {{0.1, 0.2},  // Neuron 0 weights (from input 0, input 1)
                                     {-0.3, 0.4}}; // Neuron 1 weights
            nn.layers_[0].biases_ = {0.05, -0.1};  // Neuron 0 bias, Neuron 1 bias
        }
        // Layer 1 (Output, Linear, 2 inputs from Layer 0, 1 neuron)
        if (nn.get_num_layers() > 1) {
            nn.layers_[1].weights_ = {{0.5, -0.2}}; // Neuron 0 weights (from L0_N0, L0_N1)
            nn.layers_[1].biases_ = {0.15};         // Neuron 0 bias
        }

        std::vector<double> input_data = {1.0, 2.0};
        std::vector<double> predicted_output = nn.predict(input_data, false);


        std::vector<double> expected_final_output = {0.345};
        print_vector_main("Input Data", input_data);
        print_vector_main("NN Predicted Output", predicted_output);
        print_vector_main("Expected NN Output", expected_final_output);

        assertTest(predicted_output.size() == 1, "NN Predict: Output vector size", (size_t)1, predicted_output.size());
        if (predicted_output.size() == 1) {
            assertTest(std::abs(predicted_output[0] - expected_final_output[0]) < epsilon,
                       "NN Predict: Final output value", expected_final_output[0], predicted_output[0]);
        }

    } catch (const std::exception& e) {
        std::cerr << "NeuralNetwork predict Test Case 1 failed with exception: " << e.what() << std::endl;
        assertTest(false, "NN Predict Test Case 1: Execution without exceptions");
    }

    // Test Case 2: Input size mismatch (should throw)
    std::cout << "\n-- Test Case 2: Input Size Mismatch --" << std::endl;
    try {
        std::vector<int> layer_sizes = {3, 1}; // Expects 3 inputs
        std::vector<std::string> activations = {"linear"};
        Predicting_Close_Price_Using_NN::NeuralNetwork nn_mismatch(layer_sizes, activations, 0.01);
        std::vector<double> wrong_input = {1.0, 2.0}; // Only 2 inputs provided
        nn_mismatch.predict(wrong_input); // This should throw
        assertTest(false, "NN Predict: Input size mismatch did not throw"); // Should not reach here
    } catch (const std::invalid_argument& e) {
        std::cout << "Caught expected std::invalid_argument for input size mismatch: " << e.what() << std::endl;
        assertTest(true, "NN Predict: Input size mismatch throws std::invalid_argument");
    } catch (const std::exception& e) {
        std::cerr << "NN Predict Test Case 2 (mismatch) caught unexpected exception: " << e.what() << std::endl;
        assertTest(false, "NN Predict Test Case 2: Correct exception type for mismatch");
    }

}
// --- End of NeuralNetwork Predict Test ---


int main() {
    std::cout << "Initializing main..." << std::endl;

    std::cout << "\nRunning NeuralNetwork Predict Tests" << std::endl;
    temporary_test_neural_network_predict();
    std::cout << "\nNeuralNetwork Predict Tests Complete" << std::endl << std::endl;


    return 0;
}

