// src/NeuralNetwork.cpp
#include "NeuralNetwork.h"
#include "Loss.h"
#include <iostream>
#include <stdexcept>
#include <numeric>
#include <algorithm>
#include <random>
#include <iomanip>

namespace Predicting_Close_Price_Using_NN {

    NeuralNetwork::NeuralNetwork(const std::vector<int>& layer_sizes,
                                 const std::vector<std::string>& activations,
                                 double learning_rate,
                                 double dropout_rate,
                                 double momentum_coeff,      // New parameter
                                 double weight_decay_coeff)  // New parameter
        : learning_rate_(learning_rate),
          momentum_coeff_(momentum_coeff),
          weight_decay_coeff_(weight_decay_coeff) {

        // --- Validations ---
        if (layer_sizes.size() < 2) {
            throw std::invalid_argument("NeuralNetwork requires at least an input and an output layer (layer_sizes.size() must be >= 2).");
        }
        if (layer_sizes.size() - 1 != activations.size()) {
            throw std::invalid_argument("Number of activation functions must be one less than the number of layer sizes.");
        }
        if (learning_rate <= 0.0) {
            throw std::invalid_argument("Learning rate must be positive.");
        }
        if (dropout_rate < 0.0 || dropout_rate >= 1.0) {
            throw std::invalid_argument("Dropout rate must be in [0.0, 1.0).");
        }
        if (momentum_coeff < 0.0 || momentum_coeff >= 1.0) { // Momentum typically [0, 0.99]
            throw std::invalid_argument("Momentum coefficient must be in [0.0, 1.0).");
        }
        if (weight_decay_coeff < 0.0) {
            throw std::invalid_argument("Weight decay coefficient must be non-negative.");
        }

        // --- Create Layers ---
        for (size_t i = 0; i < activations.size(); ++i) {
            int input_dim_for_layer = layer_sizes[i];
            int output_dim_for_layer = layer_sizes[i + 1];
            std::string activation_for_layer = activations[i];

            double current_layer_dropout_rate = 0.0;
            if (i < activations.size() - 1) { // Apply dropout only to hidden layers
                current_layer_dropout_rate = dropout_rate;
            }

            try {
                layers_.emplace_back(input_dim_for_layer,
                                     output_dim_for_layer,
                                     activation_for_layer,
                                     current_layer_dropout_rate);
            } catch (const std::exception& e) {
                throw std::runtime_error("Failed to create layer " + std::to_string(i) + ": " + e.what());
            }
        }

        // --- Initialize Momentum Velocities ---
        // Only if momentum is to be used.
        if (momentum_coeff_ > 0.0 || layers_.empty()) { // Check layers_.empty() to prevent access violation if no layers
             if (!layers_.empty()) initialize_momentum_velocities();
        }
    }

    void NeuralNetwork::initialize_momentum_velocities() {
        if (layers_.empty()) return; // Should not happen if constructor validation is correct

        velocity_weights_.resize(layers_.size());
        velocity_biases_.resize(layers_.size());

        for (size_t l = 0; l < layers_.size(); ++l) {
            // For weights: layers_[l].output_size_ neurons, each with layers_[l].input_size_ weights
            velocity_weights_[l].resize(layers_[l].output_size_,
                                       std::vector<double>(layers_[l].input_size_, 0.0));
            // For biases: layers_[l].output_size_ biases
            velocity_biases_[l].resize(layers_[l].output_size_, 0.0);
        }
    }


    std::vector<double> NeuralNetwork::predict(const std::vector<double>& input_data, bool training_mode) {
        if (layers_.empty()) {
            throw std::runtime_error("NeuralNetwork::predict - Network has no layers.");
        }
        if (static_cast<int>(input_data.size()) != layers_[0].input_size_) {
            throw std::invalid_argument("NeuralNetwork::predict - Input data size (" + std::to_string(input_data.size()) +
                                        ") does not match network's input layer size (" + std::to_string(layers_[0].input_size_) + ").");
        }
        std::vector<double> current_output = input_data;
        for (size_t i = 0; i < layers_.size(); ++i) {
            current_output = layers_[i].forward(current_output, training_mode);
        }
        return current_output;
    }

    void NeuralNetwork::train_one_sample(const std::vector<double>& x_input, double y_true_price) {
        if (layers_.empty()) {
            throw std::runtime_error("NeuralNetwork::train_one_sample - Network has no layers.");
        }

        // 1. Forward pass (populates caches in layers, including input_cache_ and delta_ during backprop)
        std::vector<double> y_pred_vector = predict(x_input, true); // training_mode = true
        if (y_pred_vector.size() != 1) {
            throw std::runtime_error("NeuralNetwork::train_one_sample - Expected single output for regression.");
        }
        double y_pred_price = y_pred_vector[0];

        // 2. Calculate loss derivative for the output layer
        double loss_derivative = Loss::mean_squared_error_derivative(y_true_price, y_pred_price);
        std::vector<double> current_error_signal = {loss_derivative};

        // 3. Backward pass through layers to compute deltas (dError/dZ) for each layer
        // Layer::backward now only computes deltas and error_to_propagate
        for (int i = static_cast<int>(layers_.size()) - 1; i >= 0; --i) {
            current_error_signal = layers_[i].backward(current_error_signal); // CORRECTED: Removed learning_rate_
        }

        // 4. Update weights and biases for each layer using gradients, momentum, and L2 decay
        for (size_t l = 0; l < layers_.size(); ++l) {
            Layer& current_layer = layers_[l]; // Get a reference to modify

            for (int i = 0; i < current_layer.output_size_; ++i) { // For each neuron in the current layer
                // Update Biases
                double grad_b = current_layer.delta_[i]; // db = delta

                if (momentum_coeff_ > 0.0) {
                    // Ensure velocity_biases_ is properly sized
                    if (l < velocity_biases_.size() && i < static_cast<int>(velocity_biases_[l].size())) {
                       velocity_biases_[l][i] = momentum_coeff_ * velocity_biases_[l][i] - learning_rate_ * grad_b;
                       current_layer.biases_[i] += velocity_biases_[l][i];
                    } else {
                        // This case indicates an issue with velocity_biases_ initialization or access logic
                        current_layer.biases_[i] -= learning_rate_ * grad_b; // Fallback to no momentum
                    }
                } else {
                    current_layer.biases_[i] -= learning_rate_ * grad_b;
                }

                // Update Weights
                for (int j = 0; j < current_layer.input_size_; ++j) { // For each weight connecting to this neuron
                    double grad_w = current_layer.delta_[i] * current_layer.input_cache_[j]; // dW = delta * X_prev^T

                    if (weight_decay_coeff_ > 0.0) {
                        grad_w += weight_decay_coeff_ * current_layer.weights_[i][j];
                    }

                    if (momentum_coeff_ > 0.0) {
                        // Ensure velocity_weights_ is properly sized
                        if (l < velocity_weights_.size() &&
                            i < static_cast<int>(velocity_weights_[l].size()) &&
                            j < static_cast<int>(velocity_weights_[l][i].size())) {
                            velocity_weights_[l][i][j] = momentum_coeff_ * velocity_weights_[l][i][j] - learning_rate_ * grad_w;
                            current_layer.weights_[i][j] += velocity_weights_[l][i][j];
                        } else {
                            // Fallback for safety, indicates initialization issue
                            current_layer.weights_[i][j] -= learning_rate_ * grad_w;
                        }
                    } else {
                        current_layer.weights_[i][j] -= learning_rate_ * grad_w;
                    }
                }
            }
        }
    }

    void NeuralNetwork::train(
        const std::vector<std::vector<double>>& X_train,
        const std::vector<double>& y_train,
        int epochs,
        int print_every_n_epochs,
        const std::vector<std::vector<double>>& X_val,
        const std::vector<double>& y_val) {
        if (X_train.size() != y_train.size()) {
            throw std::invalid_argument("NeuralNetwork::train - X_train and y_train must have the same number of samples.");
        }
        if (X_train.empty()) {
            std::cout << "Warning: NeuralNetwork::train called with empty training dataset." << std::endl;
            return;
        }
        if (epochs <= 0) {
            std::cout << "Warning: NeuralNetwork::train called with non-positive epochs. No training will occur." << std::endl;
            return;
        }
        bool has_validation_data = !X_val.empty() && !y_val.empty();
        if (has_validation_data && (X_val.size() != y_val.size())) {
             throw std::invalid_argument("NeuralNetwork::train - X_val and y_val must have the same number of samples if validation data is provided.");
        }

        std::cout << "Starting Neural Network training..." << std::endl;
        std::cout << " - Training samples: " << X_train.size() << std::endl;
        if(has_validation_data) {
            std::cout << " - Validation samples: " << X_val.size() << std::endl;
        }
        std::cout << " - Epochs: " << epochs << std::endl;
        std::cout << " - Learning rate: " << learning_rate_ << std::endl;
        if (momentum_coeff_ > 0.0) std::cout << " - Momentum: " << momentum_coeff_ << std::endl;
        if (weight_decay_coeff_ > 0.0) std::cout << " - Weight Decay (L2): " << weight_decay_coeff_ << std::endl;


        for (int epoch = 0; epoch < epochs; ++epoch) {
            for (size_t i = 0; i < X_train.size(); ++i) {
                size_t current_idx = i;
                if (X_train[current_idx].empty() || static_cast<int>(X_train[current_idx].size()) != layers_[0].input_size_) {
                     std::cerr << "Warning: NeuralNetwork::train - Skipping training sample " << current_idx
                               << " in epoch " << epoch + 1 << " due to incorrect feature size." << std::endl;
                    continue;
                }
                train_one_sample(X_train[current_idx], y_train[current_idx]);
            }

            if (print_every_n_epochs > 0 && ((epoch + 1) % print_every_n_epochs == 0 || epoch == 0 || epoch == epochs - 1)) {
                double train_mse = evaluate_regression(X_train, y_train);
                std::cout << "Epoch " << std::setw(4) << (epoch + 1) << "/" << epochs
                          << " | Train MSE (Norm): " << std::fixed << std::setprecision(8) << train_mse;

                if (has_validation_data && !X_val.empty()) {
                    double val_mse = evaluate_regression(X_val, y_val);
                    std::cout << " | Val MSE (Norm): " << std::fixed << std::setprecision(8) << val_mse;
                }
                std::cout << std::endl;
            }
        }
        std::cout << "Neural Network training complete." << std::endl;
    }


    double NeuralNetwork::evaluate_regression(
        const std::vector<std::vector<double>>& X_data,
        const std::vector<double>& y_true_targets) {
        if (X_data.size() != y_true_targets.size()) {
            throw std::invalid_argument("evaluate_regression: X_data and y_true_targets must have the same number of samples.");
        }
        if (X_data.empty()) {
            return 0.0;
        }
        if (layers_.empty()) {
            throw std::runtime_error("evaluate_regression: Network has no layers to make predictions.");
        }
        double total_squared_error = 0.0;
        int valid_predictions = 0;
        for (size_t i = 0; i < X_data.size(); ++i) {
            if (X_data[i].empty() || static_cast<int>(X_data[i].size()) != layers_[0].input_size_) {
                continue;
            }
            std::vector<double> y_pred_vec = predict(X_data[i], false);
            if (y_pred_vec.size() != 1) {
                continue;
            }
            double y_pred = y_pred_vec[0];
            double error = y_pred - y_true_targets[i];
            total_squared_error += error * error;
            valid_predictions++;
        }
        if (valid_predictions == 0) {
            return std::numeric_limits<double>::quiet_NaN();
        }
        return total_squared_error / static_cast<double>(valid_predictions);
    }

}