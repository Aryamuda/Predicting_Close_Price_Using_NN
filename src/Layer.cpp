// src/Layer.cpp
#include "Layer.h"
#include "Utils.h"
#include "Activations.h"
#include <cmath>
#include <stdexcept>
#include <numeric>

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
        weights_.resize(output_size_, std::vector<double>(input_size_));
        biases_.assign(output_size_, 0.0); // Initialize biases to zero

        double scale = 1.0;
        if (activation_type_ == "relu") {
            scale = std::sqrt(2.0 / input_size_); // He initialization
        } else if (activation_type_ == "sigmoid" || activation_type_ == "linear") {
            // Xavier/Glorot initialization for sigmoid/linear (can be tuned for linear)
            scale = std::sqrt(1.0 / input_size_);
        }

        for (int i = 0; i < output_size_; ++i) {
            for (int j = 0; j < input_size_; ++j) {
                weights_[i][j] = Utils::random_double(-1.0, 1.0) * scale;
            }
        }
    }

    std::vector<double> Layer::forward(const std::vector<double>& input_data, bool training_mode) {
        if (static_cast<int>(input_data.size()) != input_size_) {
            throw std::invalid_argument("Layer::forward - Input data size (" + std::to_string(input_data.size()) +
                                        ") does not match layer input size (" + std::to_string(input_size_) + ").");
        }

        // Cache the input data
        input_cache_ = input_data;

        // Suppress unused parameter warning for training_mode if dropout is not yet implemented
        (void)training_mode;

        // Calculate Z = W*X + B (pre-activation values)
        // z_cache_ has already been resized in the constructor.
        for (int i = 0; i < output_size_; ++i) {
            double z_neuron_i = biases_[i]; // Start with the bias
            for (int j = 0; j < input_size_; ++j) {
                z_neuron_i += weights_[i][j] * input_cache_[j];
            }
            z_cache_[i] = z_neuron_i;
        }

        // Apply activation function to Z to get A (activated_output)
        // activation_cache_ has already been resized.
        if (activation_type_ == "relu") {
            for (int i = 0; i < output_size_; ++i) {
                activation_cache_[i] = Activations::relu(z_cache_[i]);
            }
        } else if (activation_type_ == "sigmoid") {
            for (int i = 0; i < output_size_; ++i) {
                activation_cache_[i] = Activations::sigmoid(z_cache_[i]);
            }
        } else if (activation_type_ == "linear") {
            for (int i = 0; i < output_size_; ++i) {
                activation_cache_[i] = Activations::linear(z_cache_[i]);
            }
        }

        return activation_cache_;
    }

}
