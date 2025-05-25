#include "NeuralNetwork.h"
#include <iostream>

namespace Predicting_Close_Price_Using_NN {

    NeuralNetwork::NeuralNetwork(const std::vector<int>& layer_sizes,
                                 const std::vector<std::string>& activations,
                                 double learning_rate,
                                 double dropout_rate) // Added dropout_rate
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

            // Apply dropout only to hidden layers.
            double current_layer_dropout_rate = 0.0;
            if (i < activations.size() - 1) { // If it's not the output layer
                current_layer_dropout_rate = dropout_rate;
            }

            try {
                layers_.emplace_back(input_dim_for_layer,
                                     output_dim_for_layer,
                                     activation_for_layer,
                                     current_layer_dropout_rate); // Pass dropout rate
            } catch (const std::exception& e) {
                throw std::runtime_error("Failed to create layer " + std::to_string(i) + ": " + e.what());
            }
        }
    }


}