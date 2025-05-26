#ifndef LAYER_HPP
#define LAYER_HPP

#include <vector>
#include <string>
#include <stdexcept>
#include <random>

namespace Predicting_Close_Price_Using_NN {

    class Layer {
    public:
        //Variables
        int input_size_;
        int output_size_;
        std::string activation_type_;
        std::vector<std::vector<double>> weights_;
        std::vector<double> biases_;

        // --- Cache for Forward and Backward Pass ---
        std::vector<double> input_cache_; // Stores X for this layer
        std::vector<double> z_cache_;     // Stores Z = WX + B
        std::vector<double> activation_cache_; // Stores A = activation(Z)

        // dError/dZ for this layer's outputs (calculated in backward pass)
        std::vector<double> delta_;

        double dropout_rate_;
        std::vector<double> dropout_mask_;

        // --- Constructor ---
        Layer(int input_size, int output_size, const std::string& activation_type, double dropout_rate = 0.0);

        // --- Methods ---
        std::vector<double> forward(const std::vector<double>& input_data, bool training_mode = false);

        std::vector<double> backward(const std::vector<double>& error_from_next_layer);

    private:
        void initialize_parameters();
        void generate_dropout_mask(bool training_mode);
    };

}

#endif //LAYER_HPP
