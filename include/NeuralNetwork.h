// include/NeuralNetwork.h
#ifndef NEURAL_NETWORK_HPP
#define NEURAL_NETWORK_HPP

#include <vector>
#include <string>
#include <stdexcept>
#include "Layer.h"
#include "Loss.h"

namespace Predicting_Close_Price_Using_NN {

    class NeuralNetwork {
    public:
        // --- Member Variables ---
        std::vector<Layer> layers_;
        double learning_rate_;
        double momentum_coeff_;      // Coefficient for momentum (e.g., 0.9)
        double weight_decay_coeff_;  // Coefficient for L2 weight decay (e.g., 1e-4)

        // Velocities for momentum update
        // Structure: [layer_index][neuron_index_in_layer][weight_index_for_neuron]
        std::vector<std::vector<std::vector<double>>> velocity_weights_;
        // Structure: [layer_index][neuron_index_in_layer]
        std::vector<std::vector<double>> velocity_biases_;


        // --- Constructor ---
        NeuralNetwork(const std::vector<int>& layer_sizes,
                      const std::vector<std::string>& activations,
                      double learning_rate,
                      double dropout_rate = 0.0,
                      double momentum_coeff = 0.0,      // Default to no momentum
                      double weight_decay_coeff = 0.0); // Default to no weight decay


        // --- Methods ---
        std::vector<double> predict(const std::vector<double>& input_data, bool training_mode = false);

        void train_one_sample(const std::vector<double>& x_input, double y_true_price);

        void train(const std::vector<std::vector<double>>& X_train,
                   const std::vector<double>& y_train,
                   int epochs,
                   int print_every_n_epochs = 10,
                   const std::vector<std::vector<double>>& X_val = {},
                   const std::vector<double>& y_val = {});


        double evaluate_regression(const std::vector<std::vector<double>>& X_data,
                                   const std::vector<double>& y_true_targets);

        // Helper to get number of layers
        size_t get_num_layers() const { return layers_.size(); }

    private:
        void initialize_momentum_velocities();
    };

}

#endif //NEURAL_NETWORK_HPP
