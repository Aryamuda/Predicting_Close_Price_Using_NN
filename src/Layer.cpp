// src/Layer.cpp
#include "Layer.h"
#include "Utils.h"
#include "Activations.h"
#include <cmath>
#include <stdexcept>
#include <numeric>
#include <iostream>
#include <algorithm>

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
        if (dropout_rate < 0.0 || dropout_rate >= 1.0) { // dropout_rate = 1.0 would zero everything
            throw std::invalid_argument("Dropout rate must be in [0.0, 1.0).");
        }

        if (activation_type_ != "relu" && activation_type_ != "sigmoid" && activation_type_ != "linear") {
            throw std::invalid_argument("Unsupported activation type: " + activation_type_ +
                                        ". Supported types are 'relu', 'sigmoid', 'linear'.");
        }

        initialize_parameters();

        input_cache_.resize(input_size_, 0.0);
        z_cache_.resize(output_size_, 0.0);
        activation_cache_.resize(output_size_, 0.0);
        delta_.assign(output_size_, 0.0);
        dropout_mask_.resize(output_size_, 1.0); // Initialize mask (will be regenerated)
    }

    void Layer::initialize_parameters() {
        weights_.resize(output_size_, std::vector<double>(input_size_));
        biases_.assign(output_size_, 0.0);

        double scale = 1.0;
        if (activation_type_ == "relu") {
            scale = std::sqrt(2.0 / input_size_);
        } else if (activation_type_ == "sigmoid" || activation_type_ == "linear") {
            scale = std::sqrt(1.0 / input_size_);
        }

        for (int i = 0; i < output_size_; ++i) {
            for (int j = 0; j < input_size_; ++j) {
                weights_[i][j] = Utils::random_double(-1.0, 1.0) * scale;
            }
        }
    }

    void Layer::generate_dropout_mask(bool training_mode) {
        if (!training_mode || dropout_rate_ == 0.0) {
             std::fill(dropout_mask_.begin(), dropout_mask_.end(), 1.0);
            return;
        }
        double scale_factor = 1.0 / (1.0 - dropout_rate_);
        static std::mt19937 random_engine(std::random_device{}());
        std::uniform_real_distribution<double> dist(0.0, 1.0);

        for (int i = 0; i < output_size_; ++i) {
            if (dist(random_engine) < dropout_rate_) {
                dropout_mask_[i] = 0.0;
            } else {
                dropout_mask_[i] = scale_factor;
            }
        }
    }

    std::vector<double> Layer::forward(const std::vector<double>& input_data, bool training_mode) {
        if (static_cast<int>(input_data.size()) != input_size_) {
            throw std::invalid_argument("Layer::forward - Input data size (" + std::to_string(input_data.size()) +
                                        ") does not match layer input size (" + std::to_string(input_size_) + ").");
        }
        input_cache_ = input_data;

        for (int i = 0; i < output_size_; ++i) {
            double z_neuron_i = biases_[i];
            for (int j = 0; j < input_size_; ++j) {
                z_neuron_i += weights_[i][j] * input_cache_[j];
            }
            z_cache_[i] = z_neuron_i;
        }

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

        if (dropout_rate_ > 0.0) {
            generate_dropout_mask(training_mode);
            if (training_mode) {
                for (int i = 0; i < output_size_; ++i) {
                    activation_cache_[i] *= dropout_mask_[i];
                }
            }
        }

        return activation_cache_;
    }

    std::vector<double> Layer::backward(const std::vector<double>& error_from_next_layer) {
        if (static_cast<int>(error_from_next_layer.size()) != output_size_) {
            throw std::invalid_argument("Layer::backward - error_from_next_layer size (" + std::to_string(error_from_next_layer.size()) +
                                        ") does not match layer output size (" + std::to_string(output_size_) + ").");
        }

        if (delta_.size() != static_cast<size_t>(output_size_)) {
            delta_.resize(output_size_);
        }

        if (activation_type_ == "relu") {
            for (int i = 0; i < output_size_; ++i) {
                delta_[i] = error_from_next_layer[i] * Activations::relu_derivative(z_cache_[i]);
            }
        } else if (activation_type_ == "sigmoid") {
            for (int i = 0; i < output_size_; ++i) {
                delta_[i] = error_from_next_layer[i] * Activations::sigmoid_derivative(activation_cache_[i]);
            }
        } else if (activation_type_ == "linear") {
            for (int i = 0; i < output_size_; ++i) {
                delta_[i] = error_from_next_layer[i] * Activations::linear_derivative(z_cache_[i]);
            }
        }

        // Apply dropout mask to deltas
        if (dropout_rate_ > 0.0) {
            // The dropout_mask_ must be the one from the corresponding forward pass.
            // It's a member variable, so it should persist correctly between forward and backward for a given sample.
            for (int i = 0; i < output_size_; ++i) {
                delta_[i] *= dropout_mask_[i];
            }
        }

        // error_to_prev_layer = W^T * delta_
        std::vector<double> error_to_prev_layer(input_size_, 0.0);
        for (int j = 0; j < input_size_; ++j) {
            double sum_error_for_input_j = 0.0;
            for (int i = 0; i < output_size_; ++i) {
                sum_error_for_input_j += weights_[i][j] * delta_[i];
            }
            error_to_prev_layer[j] = sum_error_for_input_j;
        }


        return error_to_prev_layer;
    }

} // namespace PricePredictorNN
