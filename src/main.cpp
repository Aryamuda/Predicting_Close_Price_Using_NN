#include <iostream>
#include <vector>
#include <string>
#include <cmath>
#include <iomanip>
#include <stdexcept>
#include <numeric>
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

// Helper to print vectors for debugging
void print_vector_main(const std::string& name, const std::vector<double>& vec) {
    std::cout << name << ": [ ";
    for (size_t i = 0; i < vec.size(); ++i) {
        std::cout << std::fixed << std::setprecision(5) << vec[i] << (i == vec.size() - 1 ? "" : ", ");
    }
    std::cout << " ]" << std::endl;
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

// --- Temporary Test for Layer Forward Pass ---
void temporary_test_layer_forward() {
    std::cout << "\nRunning Temporary Layer Forward Pass Test" << std::endl;
    double epsilon = 1e-7; // Tolerance for floating point comparisons

    // Test Case 1: Linear Activation
    std::cout << "\n-- Test Case 1: Linear Activation --" << std::endl;
    try {
        Predicting_Close_Price_Using_NN::Layer linear_layer(2, 2, "linear"); // 2 inputs, 2 outputs

        // Manually set weights and biases for predictable output
        linear_layer.weights_ = {{0.5, 0.2},  // Neuron 0 weights
                                 {0.1, -0.3}}; // Neuron 1 weights
        linear_layer.biases_ = {0.1, -0.05};   // Neuron 0 bias, Neuron 1 bias

        std::vector<double> input1 = {1.0, 2.0};
        std::vector<double> output1 = linear_layer.forward(input1);
        std::vector<double> expected_output1 = {1.0, -0.55};
        print_vector_main("Input 1", input1);
        print_vector_main("Linear Layer Output 1", output1);
        print_vector_main("Expected Linear Output 1", expected_output1);

        assertTest(output1.size() == 2, "Linear Layer: Output vector size", (size_t)2, output1.size());
        if (output1.size() == 2) {
            assertTest(std::abs(output1[0] - expected_output1[0]) < epsilon, "Linear Layer: Neuron 0 output", expected_output1[0], output1[0]);
            assertTest(std::abs(output1[1] - expected_output1[1]) < epsilon, "Linear Layer: Neuron 1 output", expected_output1[1], output1[1]);
        }

        // Check caches
        assertTest(linear_layer.input_cache_ == input1, "Linear Layer: Input cache check");
        assertTest(std::abs(linear_layer.z_cache_[0] - expected_output1[0]) < epsilon &&
                   std::abs(linear_layer.z_cache_[1] - expected_output1[1]) < epsilon,
                   "Linear Layer: Z cache check");
        assertTest(linear_layer.activation_cache_ == output1, "Linear Layer: Activation cache check");


    } catch (const std::exception& e) {
        std::cerr << "Linear Layer forward test failed with exception: " << e.what() << std::endl;
        assertTest(false, "Linear Layer forward test without exceptions");
    }

    // Test Case 2: ReLU Activation
    std::cout << "\n-- Test Case 2: ReLU Activation --" << std::endl;
    try {
        Predicting_Close_Price_Using_NN::Layer relu_layer(3, 2, "relu"); // 3 inputs, 2 outputs

        // Manually set weights and biases
        relu_layer.weights_ = {{0.1, -0.2, 0.3},   // Neuron 0
                               {-0.4, 0.5, -0.1}};  // Neuron 1
        relu_layer.biases_ = {0.05, -0.1};

        std::vector<double> input2 = {2.0, 1.0, 3.0};
        std::vector<double> output2 = relu_layer.forward(input2);
        std::vector<double> expected_output2 = {0.95, 0.0};
        print_vector_main("Input 2", input2);
        print_vector_main("ReLU Layer Output 2", output2);
        print_vector_main("Expected ReLU Output 2", expected_output2);

        assertTest(output2.size() == 2, "ReLU Layer: Output vector size", (size_t)2, output2.size());
        if (output2.size() == 2) {
            assertTest(std::abs(output2[0] - expected_output2[0]) < epsilon, "ReLU Layer: Neuron 0 output", expected_output2[0], output2[0]);
            assertTest(std::abs(output2[1] - expected_output2[1]) < epsilon, "ReLU Layer: Neuron 1 output", expected_output2[1], output2[1]);
        }
         // Check Z cache
        std::vector<double> expected_z_cache2 = {0.95, -0.7};
        assertTest(std::abs(relu_layer.z_cache_[0] - expected_z_cache2[0]) < epsilon &&
                   std::abs(relu_layer.z_cache_[1] - expected_z_cache2[1]) < epsilon,
                   "ReLU Layer: Z cache check");


    } catch (const std::exception& e) {
        std::cerr << "ReLU Layer forward test failed with exception: " << e.what() << std::endl;
        assertTest(false, "ReLU Layer forward test without exceptions");
    }

    // Test Case 3: Sigmoid Activation (Optional, similar structure)
    std::cout << "\n-- Test Case 3: Sigmoid Activation (Brief) --" << std::endl;
    try {
        Predicting_Close_Price_Using_NN::Layer sigmoid_layer(1, 1, "sigmoid");
        sigmoid_layer.weights_ = {{0.5}};
        sigmoid_layer.biases_ = {0.1};
        std::vector<double> input3 = {1.0};
        std::vector<double> output3 = sigmoid_layer.forward(input3);
        double expected_a0_sigmoid = 1.0 / (1.0 + std::exp(-0.6));
        print_vector_main("Sigmoid Layer Output 3", output3);
        assertTest(output3.size() == 1, "Sigmoid Layer: Output size", (size_t)1, output3.size());
        if(output3.size() == 1) {
            assertTest(std::abs(output3[0] - expected_a0_sigmoid) < epsilon, "Sigmoid Layer: Neuron 0 output", expected_a0_sigmoid, output3[0]);
        }

    } catch (const std::exception& e) {
        std::cerr << "Sigmoid Layer forward test failed with exception: " << e.what() << std::endl;
    }

}
// --- End of Layer Forward Pass Test ---

// --- Temporary Test for Layer Backward Pass ---
void temporary_test_layer_backward() {
    std::cout << "\nRunning Temporary Layer Backward Pass Test" << std::endl;
    double epsilon = 1e-7;
    double learning_rate = 0.1;

    std::cout << "\n-- Test Case 1: 1-Input, 1-Output Linear Layer --" << std::endl;
    try {
        Predicting_Close_Price_Using_NN::Layer linear_layer(1, 1, "linear");
        linear_layer.weights_ = {{0.5}}; // W_old
        linear_layer.biases_ = {0.1};    // B_old
        std::vector<double> input_x = {2.0};

        // Perform forward pass to populate caches
        std::vector<double> activation_a = linear_layer.forward(input_x);
        print_vector_main("Test 1: Forward Activation A", activation_a);
        assertTest(std::abs(activation_a[0] - 1.1) < epsilon, "Test 1: Forward pass A[0]", 1.1, activation_a[0]);

        // Define incoming error and learning rate
        std::vector<double> error_from_next = {0.2}; // dError/dA

        // Store old weights and biases for checking update
        std::vector<std::vector<double>> w_old = linear_layer.weights_;
        std::vector<double> b_old = linear_layer.biases_;

        // Perform backward pass
        std::vector<double> error_to_prev = linear_layer.backward(error_from_next, learning_rate);
        print_vector_main("Test 1: Backward error_to_prev", error_to_prev);

        // Check delta_
        assertTest(std::abs(linear_layer.delta_[0] - 0.2) < epsilon, "Test 1: delta_[0]", 0.2, linear_layer.delta_[0]);

        // Check error_to_prev_layer
        assertTest(error_to_prev.size() == 1, "Test 1: error_to_prev size", (size_t)1, error_to_prev.size());
        if(error_to_prev.size() == 1) {
            assertTest(std::abs(error_to_prev[0] - 0.1) < epsilon, "Test 1: error_to_prev[0]", 0.1, error_to_prev[0]);
        }

        // Check updated weights and biases
        double expected_w_new = 0.46;
        double expected_b_new = 0.08;
        assertTest(std::abs(linear_layer.weights_[0][0] - expected_w_new) < epsilon, "Test 1: Updated weight w[0][0]", expected_w_new, linear_layer.weights_[0][0]);
        assertTest(std::abs(linear_layer.biases_[0] - expected_b_new) < epsilon, "Test 1: Updated bias b[0]", expected_b_new, linear_layer.biases_[0]);

    } catch (const std::exception& e) {
        std::cerr << "Linear Layer backward test (Test 1) failed: " << e.what() << std::endl;
        assertTest(false, "Test 1: Linear Layer backward test without exceptions");
    }


    std::cout << "\n-- Test Case 2: 2-Inputs, 1-Output ReLU Layer --" << std::endl;
    try {
        Predicting_Close_Price_Using_NN::Layer relu_layer(2, 1, "relu");
        relu_layer.weights_ = {{0.3, -0.2}};
        relu_layer.biases_ = {0.1};
        std::vector<double> input_x2 = {2.0, 1.0};

        std::vector<double> activation_a2 = relu_layer.forward(input_x2);
        print_vector_main("Test 2: Forward Activation A", activation_a2);
        assertTest(std::abs(activation_a2[0] - 0.5) < epsilon, "Test 2: Forward pass A[0]", 0.5, activation_a2[0]);
        assertTest(std::abs(relu_layer.z_cache_[0] - 0.5) < epsilon, "Test 2: Forward pass Z[0]", 0.5, relu_layer.z_cache_[0]);


        std::vector<double> error_from_next2 = {-0.4};
        std::vector<double> error_to_prev2 = relu_layer.backward(error_from_next2, learning_rate);
        print_vector_main("Test 2: Backward error_to_prev", error_to_prev2);

        assertTest(std::abs(relu_layer.delta_[0] - (-0.4)) < epsilon, "Test 2: delta_[0]", -0.4, relu_layer.delta_[0]);

        assertTest(error_to_prev2.size() == 2, "Test 2: error_to_prev size", (size_t)2, error_to_prev2.size());
        if(error_to_prev2.size() == 2) {
            assertTest(std::abs(error_to_prev2[0] - (-0.12)) < epsilon, "Test 2: error_to_prev[0]", -0.12, error_to_prev2[0]);
            assertTest(std::abs(error_to_prev2[1] - 0.08) < epsilon, "Test 2: error_to_prev[1]", 0.08, error_to_prev2[1]);
        }

        assertTest(std::abs(relu_layer.weights_[0][0] - 0.38) < epsilon, "Test 2: Updated weight w[0][0]", 0.38, relu_layer.weights_[0][0]);
        assertTest(std::abs(relu_layer.weights_[0][1] - (-0.16)) < epsilon, "Test 2: Updated weight w[0][1]", -0.16, relu_layer.weights_[0][1]);
        assertTest(std::abs(relu_layer.biases_[0] - 0.14) < epsilon, "Test 2: Updated bias b[0]", 0.14, relu_layer.biases_[0]);

    } catch (const std::exception& e) {
        std::cerr << "ReLU Layer backward test (Test 2) failed: " << e.what() << std::endl;
        assertTest(false, "Test 2: ReLU Layer backward test without exceptions");
    }

    std::cout << "\n-- Test Case 3: 1-Input, 1-Output Sigmoid Layer --" << std::endl;
    try {
        Predicting_Close_Price_Using_NN::Layer sigmoid_layer(1, 1, "sigmoid");
        sigmoid_layer.weights_ = {{0.8}};
        sigmoid_layer.biases_ = {-0.2};
        std::vector<double> input_x3 = {0.5};

        std::vector<double> activation_a3 = sigmoid_layer.forward(input_x3);
        double expected_A3 = 1.0 / (1.0 + std::exp(-0.2)); // approx 0.549834
        print_vector_main("Test 3: Forward Activation A", activation_a3);
        assertTest(std::abs(activation_a3[0] - expected_A3) < epsilon, "Test 3: Forward pass A[0]", expected_A3, activation_a3[0]);

        std::vector<double> error_from_next3 = {0.1};
        std::vector<double> error_to_prev3 = sigmoid_layer.backward(error_from_next3, learning_rate);
        print_vector_main("Test 3: Backward error_to_prev", error_to_prev3);

        double expected_delta3 = 0.1 * (expected_A3 * (1.0 - expected_A3)); // approx 0.024751
        assertTest(std::abs(sigmoid_layer.delta_[0] - expected_delta3) < epsilon, "Test 3: delta_[0]", expected_delta3, sigmoid_layer.delta_[0]);

        double expected_error_to_prev3 = 0.8 * expected_delta3; // approx 0.0198008
        assertTest(std::abs(error_to_prev3[0] - expected_error_to_prev3) < epsilon, "Test 3: error_to_prev[0]", expected_error_to_prev3, error_to_prev3[0]);

        double expected_w_new3 = 0.8 - learning_rate * (expected_delta3 * 0.5); // approx 0.79876245
        double expected_b_new3 = -0.2 - learning_rate * expected_delta3;      // approx -0.2024751
        assertTest(std::abs(sigmoid_layer.weights_[0][0] - expected_w_new3) < epsilon, "Test 3: Updated weight w[0][0]", expected_w_new3, sigmoid_layer.weights_[0][0]);
        assertTest(std::abs(sigmoid_layer.biases_[0] - expected_b_new3) < epsilon, "Test 3: Updated bias b[0]", expected_b_new3, sigmoid_layer.biases_[0]);

    } catch (const std::exception& e) {
        std::cerr << "Sigmoid Layer backward test (Test 3) failed: " << e.what() << std::endl;
        assertTest(false, "Test 3: Sigmoid Layer backward test without exceptions");
    }


}
// --- End of Layer Backward Pass Test ---

int main() {
    std::cout << "Initializing main..." << std::endl;

    std::cout << "\nLayer Tests" << std::endl;
    temporary_test_layer_initialization();
    std::cout << "\nLayer Tests Complete" << std::endl << std::endl;

    std::cout << "\nRunning Layer Forward Pass Tests" << std::endl;
    temporary_test_layer_forward();
    std::cout << "Layer Forward Pass Tests Complete" << std::endl << std::endl;

    std::cout << "\nRunning Layer Backward Pass Tests" << std::endl;
    temporary_test_layer_backward();
    std::cout << "\nBackward Pass Tests Complete" << std::endl << std::endl;
    // --- End of Temporary Test Calls ---

    return 0;
}

