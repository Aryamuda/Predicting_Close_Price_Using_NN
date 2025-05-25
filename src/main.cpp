// src/main.cpp
// Main entry point for the PricePredictorNN application.
// Includes temporary test calls/instantiations.

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



// --- Temporary Test for NeuralNetwork Initialization ---
void temporary_test_neural_network_initialization() {
    std::cout << "\n--- Running Temporary NeuralNetwork Initialization Test ---" << std::endl;
    double epsilon = 1e-7;

    // Test Case 1: Simple 2-layer network (input -> output)
    std::cout << "\n-- Test Case 1: Input -> Output Network --" << std::endl;
    try {
        std::vector<int> layer_sizes1 = {5, 1}; // 5 inputs, 1 output neuron
        std::vector<std::string> activations1 = {"linear"};
        double lr1 = 0.01;
        double dropout1 = 0.0; // No dropout

        Predicting_Close_Price_Using_NN::NeuralNetwork nn1(layer_sizes1, activations1, lr1, dropout1);
        std::cout << "NeuralNetwork 1 (5_in -> 1_linear_out, lr=" << lr1 << ") instantiated successfully." << std::endl;

        assertTest(nn1.get_num_layers() == 1, "NN1: Number of layers", (size_t)1, nn1.get_num_layers());
        assertTest(std::abs(nn1.learning_rate_ - lr1) < epsilon, "NN1: Learning rate", lr1, nn1.learning_rate_);
        if (nn1.get_num_layers() == 1) {
            assertTest(nn1.layers_[0].input_size_ == 5, "NN1: Layer 0 input size", 5, nn1.layers_[0].input_size_);
            assertTest(nn1.layers_[0].output_size_ == 1, "NN1: Layer 0 output size", 1, nn1.layers_[0].output_size_);
            assertTest(nn1.layers_[0].activation_type_ == "linear", "NN1: Layer 0 activation", std::string("linear"), nn1.layers_[0].activation_type_);
            assertTest(std::abs(nn1.layers_[0].dropout_rate_ - 0.0) < epsilon, "NN1: Layer 0 dropout rate", 0.0, nn1.layers_[0].dropout_rate_);
        }

    } catch (const std::exception& e) {
        std::cerr << "NeuralNetwork Test Case 1 failed with exception: " << e.what() << std::endl;
        assertTest(false, "NN Test Case 1: Instantiation without exceptions");
    }

    // Test Case 2: Network with one hidden layer and dropout
    std::cout << "\n-- Test Case 2: Input -> Hidden (ReLU) -> Output (Linear) Network with Dropout --" << std::endl;
    try {
        std::vector<int> layer_sizes2 = {10, 5, 1}; // 10 in, 5 hidden (ReLU), 1 out (Linear)
        std::vector<std::string> activations2 = {"relu", "linear"};
        double lr2 = 0.005;
        double dropout2 = 0.2; // 20% dropout for hidden layers

        Predicting_Close_Price_Using_NN::NeuralNetwork nn2(layer_sizes2, activations2, lr2, dropout2);
        std::cout << "NeuralNetwork 2 (10_in -> 5_relu_hidden -> 1_linear_out, lr=" << lr2 << ", dropout=" << dropout2 << ") instantiated." << std::endl;

        assertTest(nn2.get_num_layers() == 2, "NN2: Number of layers", (size_t)2, nn2.get_num_layers());
        assertTest(std::abs(nn2.learning_rate_ - lr2) < epsilon, "NN2: Learning rate", lr2, nn2.learning_rate_);

        // Check Layer 0 (Hidden Layer)
        if (nn2.get_num_layers() > 0) {
            assertTest(nn2.layers_[0].input_size_ == 10, "NN2: Layer 0 (Hidden) input size", 10, nn2.layers_[0].input_size_);
            assertTest(nn2.layers_[0].output_size_ == 5, "NN2: Layer 0 (Hidden) output size", 5, nn2.layers_[0].output_size_);
            assertTest(nn2.layers_[0].activation_type_ == "relu", "NN2: Layer 0 (Hidden) activation", std::string("relu"), nn2.layers_[0].activation_type_);
            assertTest(std::abs(nn2.layers_[0].dropout_rate_ - dropout2) < epsilon, "NN2: Layer 0 (Hidden) dropout rate", dropout2, nn2.layers_[0].dropout_rate_);
        }
        // Check Layer 1 (Output Layer)
        if (nn2.get_num_layers() > 1) {
            assertTest(nn2.layers_[1].input_size_ == 5, "NN2: Layer 1 (Output) input size", 5, nn2.layers_[1].input_size_);
            assertTest(nn2.layers_[1].output_size_ == 1, "NN2: Layer 1 (Output) output size", 1, nn2.layers_[1].output_size_);
            assertTest(nn2.layers_[1].activation_type_ == "linear", "NN2: Layer 1 (Output) activation", std::string("linear"), nn2.layers_[1].activation_type_);
            assertTest(std::abs(nn2.layers_[1].dropout_rate_ - 0.0) < epsilon, "NN2: Layer 1 (Output) dropout rate (should be 0)", 0.0, nn2.layers_[1].dropout_rate_);
        }

    } catch (const std::exception& e) {
        std::cerr << "NeuralNetwork Test Case 2 failed with exception: " << e.what() << std::endl;
        assertTest(false, "NN Test Case 2: Instantiation without exceptions");
    }

    // Test Case 3: Invalid configurations (should throw)
    std::cout << "\n-- Test Case 3: Invalid Configurations --" << std::endl;
    bool thrown = false;
    try {
        std::vector<int> ls = {5}; std::vector<std::string> ac = {};
        Predicting_Close_Price_Using_NN::NeuralNetwork nn_invalid(ls, ac, 0.01);
    } catch (const std::invalid_argument& e) {
        thrown = true;
        std::cout << "Caught expected invalid_argument for too few layer_sizes: " << e.what() << std::endl;
    }
    assertTest(thrown, "NN Invalid Config: Too few layer_sizes throws");

    thrown = false;
    try {
        std::vector<int> ls = {5, 3, 1}; std::vector<std::string> ac = {"relu"}; // Mismatch
        Predicting_Close_Price_Using_NN::NeuralNetwork nn_invalid(ls, ac, 0.01);
    } catch (const std::invalid_argument& e) {
        thrown = true;
        std::cout << "Caught expected invalid_argument for activations/layers mismatch: " << e.what() << std::endl;
    }
    assertTest(thrown, "NN Invalid Config: Activations/layers mismatch throws");


}
// --- End of NeuralNetwork Initialization Test ---


int main() {
    std::cout << "Initializing main..." << std::endl;

    std::cout << "\nRunning NeuralNetwork Init Tests" << std::endl;
    temporary_test_neural_network_initialization();
    std::cout << "\nNeuralNetwork Init Tests Complete" << std::endl << std::endl;

    return 0;
}

