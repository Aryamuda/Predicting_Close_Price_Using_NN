#ifndef NEURAL_NETWORK_HPP
#define NEURAL_NETWORK_HPP

#include <vector>
#include <string>
#include <stdexcept>
#include "Layer.h"
#include "Loss.h"
#include "BatchNormLayer.h"

namespace Predicting_Close_Price_Using_NN {

    // NEW: Struct to hold multiple regression metrics
    struct RegressionMetrics {
        double mse = 0.0; // Mean Squared Error
        double rmse = 0.0; // Root Mean Squared Error
        double mae = 0.0; // Mean Absolute Error
        double r2 = 0.0;  // R-squared (Coefficient of Determination)
    };

    class NeuralNetwork {
    public:
        // --- Member Variables ---
        std::vector<Layer> layers_;
        std::vector<BatchNormLayer> bn_layers_;
        std::vector<bool> use_bn_for_layer_;

        double learning_rate_;
        double momentum_coeff_;
        double weight_decay_coeff_;

        std::vector<std::vector<std::vector<double>>> velocity_weights_;
        std::vector<std::vector<double>> velocity_biases_;

        std::vector<std::vector<std::vector<double>>> accumulated_weight_gradients_;
        std::vector<std::vector<double>> accumulated_bias_gradients_;


        // --- Constructor ---
        NeuralNetwork(const std::vector<int>& layer_sizes,
                      const std::vector<std::string>& activations,
                      double learning_rate,
                      double dropout_rate = 0.0,
                      double momentum_coeff = 0.0,
                      double weight_decay_coeff = 0.0,
                      bool apply_batch_norm = false);


        // --- Methods ---
        std::vector<double> predict(const std::vector<double>& input_data, bool training_mode = false);

        void process_sample_and_accumulate_gradients(const std::vector<double>& x_input, double y_true_price);

        void apply_accumulated_gradients(size_t batch_size);


        void train(const std::vector<std::vector<double>>& X_train,
                   const std::vector<double>& y_train,
                   int epochs,
                   size_t batch_size,
                   int print_every_n_epochs = 10,
                   const std::vector<std::vector<double>>& X_val = {},
                   const std::vector<double>& y_val = {});

        /**
         * @brief Evaluates the network's performance on a given dataset.
         * Note: Metrics are calculated on the provided values (e.g., normalized if inputs are normalized).
         * @param X_data The input features for the test/validation set.
         * @param y_true_targets The true target values for the test/validation set.
         * @return A RegressionMetrics struct containing MSE, RMSE, MAE, and R2.
         */
        RegressionMetrics evaluate_regression( // <-- CHANGED RETURN TYPE
            const std::vector<std::vector<double>>& X_data,
            const std::vector<double>& y_true_targets);

        size_t get_num_layers() const { return layers_.size(); }

    private:
        void initialize_momentum_velocities();
        void initialize_accumulated_gradients();
        void reset_accumulated_gradients();
    };

}

#endif //NEURAL_NETWORK_HPP
