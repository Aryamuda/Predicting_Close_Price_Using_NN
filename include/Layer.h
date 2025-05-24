#ifndef LAYER_HPP
#define LAYER_HPP

#include <vector>
#include <string>

namespace Predicting_Close_Price_Using_NN {

    class Layer {
    public:
        //Variables
        int input_size_;  // Number of input neurons (or features from the previous layer)
        int output_size_; // Number of output neurons in this layer

        std::string activation_type_; // Type of activation: "relu", "sigmoid", "linear"

        // Weights: output_size_ rows, input_size_ columns
        std::vector<std::vector<double>> weights_;
        std::vector<double> biases_; // One bias per output neuron

        // Input to this layer during the last forward pass (size: input_size_)
        std::vector<double> input_cache_;

        // Pre-activation output (Z = W*X + B) of this layer (size: output_size_)
        std::vector<double> z_cache_;

        // Post-activation output (A = activation(Z)) of this layer (size: output_size_)
        std::vector<double> activation_cache_;

        // Error signal (delta) for this layer's outputs (dError/dZ) (size: output_size_)

        std::vector<double> delta_;

        double dropout_rate_; // Dropout rate (0.0 means no dropout)
        // std::vector<double> dropout_mask_; // Will be added when dropout is implemented

        Layer(int input_size, int output_size, const std::string& activation_type, double dropout_rate = 0.0);

        std::vector<double> forward(const std::vector<double>& input_data, bool training_mode = false);

        std::vector<double> backward(const std::vector<double>& error_from_next_layer, double learning_rate);

    private:
        void initialize_parameters();
    };

}

#endif
