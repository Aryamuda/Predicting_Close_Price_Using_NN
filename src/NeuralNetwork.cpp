#include "NeuralNetwork.h"
#include <iostream>
#include <stdexcept>

namespace Predicting_Close_Price_Using_NN {

    NeuralNetwork::NeuralNetwork(const std::vector<int>& layer_sizes,
                                 const std::vector<std::string>& activations,
                                 double learning_rate,
                                 double dropout_rate)
        : learning_rate_(learning_rate) {

        // Validate inputs
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


        // Create the layers
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

        // Validate input_data size against the first layer's input size
        if (static_cast<int>(input_data.size()) != layers_[0].input_size_) {
            throw std::invalid_argument("NeuralNetwork::predict - Input data size (" + std::to_string(input_data.size()) +
                                        ") does not match network's input layer size (" + std::to_string(layers_[0].input_size_) + ").");
        }

        std::vector<double> current_output = input_data;

        // Propagate through each layer
        for (size_t i = 0; i < layers_.size(); ++i) {
            current_output = layers_[i].forward(current_output, training_mode);
        }

        return current_output; // This is the final output of the network
    }


}
