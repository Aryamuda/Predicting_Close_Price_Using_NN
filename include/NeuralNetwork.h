#ifndef NEURAL_NETWORK_HPP
#define NEURAL_NETWORK_HPP

#include <vector>
#include <string>
#include <stdexcept> // For std::invalid_argument
#include "Layer.h"   // We are composing Layer objects
#include "Loss.h"    // For Loss functions
#include "BatchNormLayer.h" // Include BatchNormLayer header

namespace Predicting_Close_Price_Using_NN {

    class NeuralNetwork {
    public:
        //Variables
        std::vector<Layer> layers_;
        std::vector<BatchNormLayer> bn_layers_; // BatchNorm layers, one per corresponding Layer
        std::vector<bool> use_bn_for_layer_;   // Flags to indicate if BN is used for a layer

        double learning_rate_;
        double momentum_coeff_;
        double weight_decay_coeff_;

        std::vector<std::vector<std::vector<double>>> velocity_weights_;
        std::vector<std::vector<double>> velocity_biases_;


        // --- Constructor ---
        NeuralNetwork(const std::vector<int>& layer_sizes,
                      const std::vector<std::string>& activations,
                      double learning_rate,
                      double dropout_rate = 0.0,
                      double momentum_coeff = 0.0,
                      double weight_decay_coeff = 0.0,
                      bool apply_batch_norm = false); // New parameter for BN


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
