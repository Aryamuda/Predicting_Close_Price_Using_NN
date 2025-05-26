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
                                 double dropout_rate)
        : learning_rate_(learning_rate) {

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

        for (size_t i = 0; i < activations.size(); ++i) {
            int input_dim_for_layer = layer_sizes[i];
            int output_dim_for_layer = layer_sizes[i + 1];
            std::string activation_for_layer = activations[i];

            double current_layer_dropout_rate = 0.0;
            if (i < activations.size() - 1) {
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

        // 1. Forward pass to get the prediction and populate caches in layers
        std::vector<double> y_pred_vector = predict(x_input, true);

        // Assuming the network has a single output neuron for regression
        if (y_pred_vector.size() != 1) {
            throw std::runtime_error("NeuralNetwork::train_one_sample - Expected a single output value from network for regression.");
        }
        double y_pred_price = y_pred_vector[0];
        double loss_derivative = Loss::mean_squared_error_derivative(y_true_price, y_pred_price);
        std::vector<double> error_signal_from_loss = {loss_derivative};
        std::vector<double> current_error_signal = error_signal_from_loss;
        for (int i = static_cast<int>(layers_.size()) - 1; i >= 0; --i) {
            current_error_signal = layers_[i].backward(current_error_signal, learning_rate_);
        }
    }

    void NeuralNetwork::train(
        const std::vector<std::vector<double>>& X_train,
        const std::vector<double>& y_train,
        int epochs,
        int print_every_n_epochs,
        const std::vector<std::vector<double>>& X_val, // Optional validation data
        const std::vector<double>& y_val) {           // Optional validation data

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


        for (int epoch = 0; epoch < epochs; ++epoch) {

            // Iterate through training data sequentially (important for time series)
            for (size_t i = 0; i < X_train.size(); ++i) {
                // size_t current_idx = shuffle_each_epoch ? indices[i] : i;
                size_t current_idx = i; // No shuffling for time series by default

                if (X_train[current_idx].empty() || static_cast<int>(X_train[current_idx].size()) != layers_[0].input_size_) {
                     std::cerr << "Warning: NeuralNetwork::train - Skipping training sample " << current_idx
                               << " in epoch " << epoch + 1 << " due to incorrect feature size." << std::endl;
                    continue;
                }
                train_one_sample(X_train[current_idx], y_train[current_idx]);
            }

            // Print progress
            if (print_every_n_epochs > 0 && ((epoch + 1) % print_every_n_epochs == 0 || epoch == 0 || epoch == epochs - 1)) {
                double train_mse = evaluate_regression(X_train, y_train);
                std::cout << "Epoch " << std::setw(4) << (epoch + 1) << "/" << epochs
                          << " | Train MSE (Norm): " << std::fixed << std::setprecision(8) << train_mse;

                if (has_validation_data) {
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
