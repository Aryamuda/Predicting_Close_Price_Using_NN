// src/Layer.cpp
#include "Layer.h"
#include "Utils.h"
#include "Activations.h"
#include <cmath>
#include <stdexcept>
#include <numeric>
#include <iostream>

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
        if (dropout_rate < 0.0 || dropout_rate >= 1.0) {
            throw std::invalid_argument("Dropout rate must be in [0.0, 1.0).");
        }

        // Validate activation type
        if (activation_type_ != "relu" && activation_type_ != "sigmoid" && activation_type_ != "linear") {
            throw std::invalid_argument("Unsupported activation type: " + activation_type_ +
                                        ". Supported types are 'relu', 'sigmoid', 'linear'.");
        }

        initialize_parameters();

        // Initialize cache vectors with correct sizes, filled with 0.0
        input_cache_.resize(input_size_, 0.0);
        z_cache_.resize(output_size_, 0.0);
        activation_cache_.resize(output_size_, 0.0);
        delta_.assign(output_size_, 0.0); // Ensure delta_ is also initialized properly
    }

    void Layer::initialize_parameters() {
        weights_.resize(output_size_, std::vector<double>(input_size_));
        biases_.assign(output_size_, 0.0);

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
        (void)training_mode;

        for (int i = 0; i < output_size_; ++i) {
            double z_neuron_i = biases_[i];
            for (int j = 0; j < input_size_; ++j) {
                z_neuron_i += weights_[i][j] * input_cache_[j];
            }
            z_cache_[i] = z_neuron_i;
        }

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

    std::vector<double> Layer::backward(const std::vector<double>& error_from_next_layer, double learning_rate) {
        if (static_cast<int>(error_from_next_layer.size()) != output_size_) {
            throw std::invalid_argument("Layer::backward - error_from_next_layer size (" + std::to_string(error_from_next_layer.size()) +
                                        ") does not match layer output size (" + std::to_string(output_size_) + ").");
        }

        // Step 1: Calculate delta_ for this layer (dError/dZ_this_layer)

        if (delta_.size() != static_cast<size_t>(output_size_)) {
            delta_.resize(output_size_);
        }

        if (activation_type_ == "relu") {
            for (int i = 0; i < output_size_; ++i) {
                delta_[i] = error_from_next_layer[i] * Activations::relu_derivative(z_cache_[i]);
            }
        } else if (activation_type_ == "sigmoid") {
            for (int i = 0; i < output_size_; ++i) {
                // Sigmoid derivative uses the activated output (A)
                delta_[i] = error_from_next_layer[i] * Activations::sigmoid_derivative(activation_cache_[i]);
            }
        } else if (activation_type_ == "linear") {
            for (int i = 0; i < output_size_; ++i) {
                delta_[i] = error_from_next_layer[i] * Activations::linear_derivative(z_cache_[i]); // which is just error_from_next_layer[i] * 1.0
            }
        }


        // Step 2: Calculate error to propagate to the previous layer (dError/dA_previous_layer)
        // error_to_prev_layer = W^T * delta_
        std::vector<double> error_to_prev_layer(input_size_, 0.0);
        for (int j = 0; j < input_size_; ++j) { // For each neuron in the previous layer (or input feature)
            double sum_error_for_input_j = 0.0;
            for (int i = 0; i < output_size_; ++i) { // Sum over neurons in this current layer
                sum_error_for_input_j += weights_[i][j] * delta_[i];
            }
            error_to_prev_layer[j] = sum_error_for_input_j;
        }

        // Step 3: Calculate gradients for weights and biases
        // Update weights (dW_ij = delta_i * input_cache_j)
        for (int i = 0; i < output_size_; ++i) { // For each neuron in this layer
            for (int j = 0; j < input_size_; ++j) { // For each input to this neuron
                double dW_ij = delta_[i] * input_cache_[j];
                weights_[i][j] -= learning_rate * dW_ij;
            }
            // Update bias (db_i = delta_i)
            double db_i = delta_[i];
            biases_[i] -= learning_rate * db_i;
        }

        return error_to_prev_layer;
    }

}
