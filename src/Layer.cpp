// src/Layer.cpp
#include "Layer.h"
#include "Utils.h"
#include <cmath>
#include <stdexcept>

namespace Predicting_Close_Price_Using_NN {

    Layer::Layer(int input_size, int output_size, const std::string& activation_type, double dropout_rate)
        : input_size_(input_size),
          output_size_(output_size),
          activation_type_(activation_type),
          dropout_rate_(dropout_rate) {

        if (input_size <= 0) {
            throw std::invalid_argument("Layer input size must be positive.");
        }
        if (output_size <= 0) {
            throw std::invalid_argument("Layer output size must be positive.");
        }
        if (dropout_rate < 0.0 || dropout_rate >= 1.0) { // Dropout of 1.0 would zero out everything
            throw std::invalid_argument("Dropout rate must be in [0.0, 1.0).");
        }

        // Validate activation type
        if (activation_type_ != "relu" && activation_type_ != "sigmoid" && activation_type_ != "linear") {
            throw std::invalid_argument("Unsupported activation type: " + activation_type_ +
                                        ". Supported types are 'relu', 'sigmoid', 'linear'.");
        }

        initialize_parameters();

        // Initialize cache vectors with correct sizes, filled with 0.0
        // These will be overwritten during forward/backward passes.
        input_cache_.resize(input_size_, 0.0);
        z_cache_.resize(output_size_, 0.0);
        activation_cache_.resize(output_size_, 0.0);
        delta_.resize(output_size_, 0.0);
        // dropout_mask_.resize(output_size_, 1.0); // When dropout is implemented
    }

    void Layer::initialize_parameters() {
        // Initialize weights
        weights_.resize(output_size_, std::vector<double>(input_size_));

        // He initialization for ReLU
        // Xavier/Glorot initialization for Sigmoid
        // Small random values for Linear output layer
        double scale = 1.0;
        if (activation_type_ == "relu") {
            scale = std::sqrt(2.0 / input_size_); // He initialization factor
        } else if (activation_type_ == "sigmoid") {
            // Xavier/Glorot initialization factor for sigmoid/tanh
            scale = std::sqrt(1.0 / input_size_);
        } else if (activation_type_ == "linear") {


        for (int i = 0; i < output_size_; ++i) {
            for (int j = 0; j < input_size_; ++j) {
                // Generate number from a standard normal distribution (mean 0, variance 1)
                // then scale it
                weights_[i][j] = Utils::random_double(-1.0, 1.0) * scale;
            }
        }

        // Initialize biases to zero
        biases_.assign(output_size_, 0.0);
    }

}
