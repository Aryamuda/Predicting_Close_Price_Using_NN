#ifndef NEURAL_NETWORK_HPP
#define NEURAL_NETWORK_HPP

#include <vector>
#include <string>
#include "Layer.h"

namespace Predicting_Close_Price_Using_NN {

    class NeuralNetwork {
    public:
        // Variables
        std::vector<Layer> layers_; // Stores all layers in the network
        double learning_rate_;

        // --- Constructor ---
        NeuralNetwork(const std::vector<int>& layer_sizes,
                      const std::vector<std::string>& activations,
                      double learning_rate,
                      double dropout_rate = 0.0);

        // --- Methods ---
        std::vector<double> predict(const std::vector<double>& input_data, bool training_mode = false);
                       const std::vector<double>& y_test_prices);

        // Helper to get number of layers
        size_t get_num_layers() const { return layers_.size(); }

    };

}

#endif
