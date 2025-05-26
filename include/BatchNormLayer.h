#ifndef BATCH_NORM_LAYER_HPP
#define BATCH_NORM_LAYER_HPP

#include <vector>
#include <string>
#include <cmath>
#include <numeric>   // For std::accumulate
#include <stdexcept> // For std::invalid_argument

namespace Predicting_Close_Price_Using_NN {

    class BatchNormLayer {
    public:
        int num_features_; // Number of features/neurons in the input to this layer
        double epsilon_;   // Small constant for numerical stability (to avoid division by zero)
        double momentum_;  // Momentum for updating running mean and variance

        // Learnable parameters
        std::vector<double> gamma_; // Scale parameters (one per feature)
        std::vector<double> beta_;  // Shift parameters (one per feature)

        // Statistics for inference mode (exponential moving averages)
        std::vector<double> running_mean_;
        std::vector<double> running_var_;

        // Cache for backward pass (for a single instance/sample)
        std::vector<double> x_input_cache_;      // Input to this BN layer (A from previous layer)
        std::vector<double> x_normalized_cache_; // x_input_cache_ after normalization, before gamma/beta
        double mean_cache_;          // Mean of x_input_cache_ for the current instance
        double variance_cache_;      // Variance of x_input_cache_ for the current instance
                                     // Note: For true Batch Norm, mean/var would be vectors (per feature across batch)
                                     // Here, for LayerNorm style, they are scalars (across features of one instance)

        bool initialized_running_stats_; // Flag to track if running stats have been initialized from first batch


        BatchNormLayer(int num_features, double epsilon = 1e-5, double momentum = 0.9);

        std::vector<double> forward(const std::vector<double>& x_input_sample, bool training_mode);


        std::vector<double> backward(const std::vector<double>& dout_sample, double learning_rate);
    };

}

#endif
