#include <iostream>
#include <vector>
#include <string>
#include <cmath>
#include <iomanip>
#include <stdexcept>
#include "Layer.h"

template<typename T, typename U>
void assertTest(bool condition, const std::string& test_name, const T& expected, const U& actual) {
    std::cout << std::fixed << std::setprecision(8); // Consistent float output
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


// --- Test for Layer Initialization ---
void temporary_test_layer_initialization() {
    std::cout << "\nRunning Layer Test" << std::endl;
    try {
        Predicting_Close_Price_Using_NN::Layer test_layer1(10, 5, "relu", 0.1);
        std::cout << "Layer 1 (10 in, 5 out, relu, dropout 0.1) instantiated successfully." << std::endl;
        assertTest(test_layer1.input_size_ == 10, "Layer 1: input_size_ check", 10, test_layer1.input_size_);
        assertTest(test_layer1.output_size_ == 5, "Layer 1: output_size_ check", 5, test_layer1.output_size_);
        assertTest(test_layer1.activation_type_ == "relu", "Layer 1: activation_type_ check", std::string("relu"), test_layer1.activation_type_);
        assertTest(std::abs(test_layer1.dropout_rate_ - 0.1) < 1e-9, "Layer 1: dropout_rate_ check");
        assertTest(test_layer1.weights_.size() == 5, "Layer 1: weights_ rows check", (size_t)5, test_layer1.weights_.size());
        if (!test_layer1.weights_.empty()) {
            assertTest(test_layer1.weights_[0].size() == 10, "Layer 1: weights_ cols check", (size_t)10, test_layer1.weights_[0].size());
        }
        assertTest(test_layer1.biases_.size() == 5, "Layer 1: biases_ size check", (size_t)5, test_layer1.biases_.size());
        
        // Print a few initial weights/biases to see they are not all zero
        if (test_layer1.output_size_ > 0 && test_layer1.input_size_ > 0 && !test_layer1.weights_.empty() && !test_layer1.weights_[0].empty()) {
            std::cout << "Layer 1: Sample initial weight w[0][0]: " << test_layer1.weights_[0][0] << std::endl;
        }
        if (test_layer1.output_size_ > 0 && !test_layer1.biases_.empty()) {
             std::cout << "Layer 1: Sample initial bias b[0]: " << test_layer1.biases_[0] << std::endl;
        }


        Predicting_Close_Price_Using_NN::Layer test_layer2(5, 1, "linear"); // 5 inputs, 1 output neuron, Linear activation
        std::cout << "Layer 2 (5 in, 1 out, linear) instantiated successfully." << std::endl;
        assertTest(test_layer2.input_size_ == 5, "Layer 2: input_size_ check", 5, test_layer2.input_size_);
        assertTest(test_layer2.output_size_ == 1, "Layer 2: output_size_ check", 1, test_layer2.output_size_);
        assertTest(test_layer2.activation_type_ == "linear", "Layer 2: activation_type_ check", std::string("linear"), test_layer2.activation_type_);

    } catch (const std::exception& e) {
        std::cerr << "Layer initialization test failed with exception: " << e.what() << std::endl;
        assertTest(false, "Layer instantiation without exceptions");
    }
}
// --- End of Layer Initialization Test ---

int main() {
    std::cout << "Initializing main..." << std::endl;

    std::cout << "\nLayer Tests" << std::endl;
    temporary_test_layer_initialization();
    std::cout << "\nLayer Tests Complete" << std::endl << std::endl;

    return 0;
}

