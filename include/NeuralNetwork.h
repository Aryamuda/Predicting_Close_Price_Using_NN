#ifndef NEURAL_NETWORK_HPP
#define NEURAL_NETWORK_HPP

#include <vector>
#include <string>
#include <stdexcept>
#include "Layer.h"
#include "Loss.h"
#include "BatchNormLayer.h"

namespace Predicting_Close_Price_Using_NN {

    class NeuralNetwork {
    public:
        //Variables
        std::vector<Layer> layers_;
        std::vector<BatchNormLayer> bn_layers_;
        std::vector<bool> use_bn_for_layer_;

        double learning_rate_;
        double momentum_coeff_;
        double weight_decay_coeff_;

        // Velocities for momentum update
        std::vector<std::vector<std::vector<double>>> velocity_weights_;
        std::vector<std::vector<double>> velocity_biases_;

        //Gradient Accumulation
        std::vector<std::vector<std::vector<double>>> accumulated_weight_gradients_;
        std::vector<std::vector<double>> accumulated_bias_gradients_;


        //Constructor
        NeuralNetwork(const std::vector<int>& layer_sizes,
                      const std::vector<std::string>& activations,
                      double learning_rate,
                      double dropout_rate = 0.0,
                      double momentum_coeff = 0.0,
                      double weight_decay_coeff = 0.0,
                      bool apply_batch_norm = false);


        // --- Methods ---
        std::vector<double> predict(const std::vector<double>& input_data, bool training_mode = false);

        //Processes a single input sample: performs forward and backward pass
        void process_sample_and_accumulate_gradients(const std::vector<double>& x_input, double y_true_price);

        //Applies the accumulated gradients (averaged by batch size) to update
        void apply_accumulated_gradients(size_t batch_size);


        void train(const std::vector<std::vector<double>>& X_train,
                   const std::vector<double>& y_train,
                   int epochs,
                   size_t batch_size, // <-- NEW: Add batch_size parameter
                   int print_every_n_epochs = 10,
                   const std::vector<std::vector<double>>& X_val = {},
                   const std::vector<double>& y_val = {});


        double evaluate_regression(const std::vector<std::vector<double>>& X_data,
                                   const std::vector<double>& y_true_targets);

        size_t get_num_layers() const { return layers_.size(); }

    private:
        void initialize_momentum_velocities();
        void initialize_accumulated_gradients(); //Set up gradient accumulators
        void reset_accumulated_gradients();      //Clear accumulators after batch update
    };

}

#endif
