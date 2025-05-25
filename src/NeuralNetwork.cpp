#include "NeuralNetwork.h"
#include "Loss.h"
#include <iostream>
#include <stdexcept>

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

        // 2. Calculate the derivative of the loss function with respect to the network's output (prediction)
        // For MSE = 0.5 * (y_pred - y_true)^2, derivative w.r.t y_pred is (y_pred - y_true)
        double loss_derivative = Loss::mean_squared_error_derivative(y_true_price, y_pred_price);

        // This loss_derivative is dError/dActivation_output_layer.
        // It needs to be a vector to pass to the last layer's backward method.
        std::vector<double> error_signal_from_loss = {loss_derivative};

        // 3. Backward pass through all layers
        // Start with the error signal from the loss function for the output layer
        std::vector<double> current_error_signal = error_signal_from_loss;

        for (int i = static_cast<int>(layers_.size()) - 1; i >= 0; --i) {
            // The backward method of a layer returns the error signal for the *previous* layer (its input)
            current_error_signal = layers_[i].backward(current_error_signal, learning_rate_);
        }

    }



}
