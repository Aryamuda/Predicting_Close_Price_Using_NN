#include "BatchNormLayer.h"
#include <iostream>

namespace Predicting_Close_Price_Using_NN {

    BatchNormLayer::BatchNormLayer(int num_features, double epsilon, double momentum)
        : num_features_(num_features),
          epsilon_(epsilon),
          momentum_(momentum),
          gamma_(num_features, 1.0),      // Initialize gamma (scale) to 1
          beta_(num_features, 0.0),       // Initialize beta (shift) to 0
          running_mean_(num_features, 0.0),
          running_var_(num_features, 1.0),  // Initialize running variance to 1 for stability
          x_input_cache_(num_features),
          x_normalized_cache_(num_features),
          mean_cache_(0.0),
          variance_cache_(0.0),
          initialized_running_stats_(false) {

        if (num_features <= 0) {
            throw std::invalid_argument("BatchNormLayer: num_features must be positive.");
        }
    }

    std::vector<double> BatchNormLayer::forward(const std::vector<double>& x_input_sample, bool training_mode) {
        if (static_cast<int>(x_input_sample.size()) != num_features_) {
            throw std::invalid_argument("BatchNormLayer::forward - Input sample size does not match num_features.");
        }

        x_input_cache_ = x_input_sample; // Cache input for backward pass

        double current_mean;
        double current_variance;
        std::vector<double> x_hat(num_features_); // x_normalized before gamma/beta
        std::vector<double> output(num_features_);

        if (training_mode) {
            // Calculate mean and variance for the current input sample (across its features)
            // This is Layer Normalization style, not canonical Batch Normalization
            current_mean = std::accumulate(x_input_sample.begin(), x_input_sample.end(), 0.0) / num_features_;

            double sum_sq_diff = 0.0;
            for (int j = 0; j < num_features_; ++j) {
                sum_sq_diff += std::pow(x_input_sample[j] - current_mean, 2);
            }
            current_variance = sum_sq_diff / num_features_;

            // Cache these for backward pass
            mean_cache_ = current_mean;
            variance_cache_ = current_variance;

            // Normalize
            double inv_stddev = 1.0 / std::sqrt(current_variance + epsilon_);
            for (int j = 0; j < num_features_; ++j) {
                x_hat[j] = (x_input_sample[j] - current_mean) * inv_stddev;
            }
            x_normalized_cache_ = x_hat; // Cache normalized x

            if (!initialized_running_stats_) {
                 for(int j=0; j<num_features_; ++j) {
                    running_mean_[j] = current_mean;
                    running_var_[j] = current_variance;
                 }
                initialized_running_stats_ = true;
            } else {
                for(int j=0; j<num_features_; ++j) { // All features get same running_mean/var if mean/var are scalar over features
                    running_mean_[j] = momentum_ * running_mean_[j] + (1.0 - momentum_) * current_mean;
                    running_var_[j] = momentum_ * running_var_[j] + (1.0 - momentum_) * current_variance;
                }
            }

        } else { // Inference mode
            // Use running mean and variance
            for (int j = 0; j < num_features_; ++j) {
                double inv_stddev = 1.0 / std::sqrt(running_var_[j] + epsilon_);
                x_hat[j] = (x_input_sample[j] - running_mean_[j]) * inv_stddev;
            }

        }

        // Scale and shift: y = gamma * x_hat + beta
        for (int j = 0; j < num_features_; ++j) {
            output[j] = gamma_[j] * x_hat[j] + beta_[j];
        }

        return output;
    }


    std::vector<double> BatchNormLayer::backward(const std::vector<double>& dout_sample, double learning_rate) {
        if (static_cast<int>(dout_sample.size()) != num_features_) {
            throw std::invalid_argument("BatchNormLayer::backward - dout_sample size does not match num_features.");
        }

        // Gradients for learnable parameters gamma and beta
        std::vector<double> dgamma(num_features_);
        std::vector<double> dbeta(num_features_);

        // Gradient of loss w.r.t. normalized input (dx_hat)
        std::vector<double> dx_hat(num_features_);

        for (int j = 0; j < num_features_; ++j) {
            dbeta[j] = dout_sample[j];                           // dL/dbeta_j = dL/dy_j * dy_j/dbeta_j = dout_j * 1
            dgamma[j] = dout_sample[j] * x_normalized_cache_[j]; // dL/dgamma_j = dL/dy_j * dy_j/dgamma_j = dout_j * x_hat_j
            dx_hat[j] = dout_sample[j] * gamma_[j];              // dL/dx_hat_j = dL/dy_j * dy_j/dx_hat_j = dout_j * gamma_j
        }

        // Gradients w.r.t. variance and mean
        // dx_hat_j = (x_j - mean) * inv_stddev
        // inv_stddev = (variance + epsilon)^(-0.5)
        double inv_stddev = 1.0 / std::sqrt(variance_cache_ + epsilon_);
        double dvar = 0.0;
        for (int j = 0; j < num_features_; ++j) {
            dvar += dx_hat[j] * (x_input_cache_[j] - mean_cache_) * (-0.5) * std::pow(variance_cache_ + epsilon_, -1.5);
        }

        double dmean = 0.0;
        for (int j = 0; j < num_features_; ++j) {
            dmean += dx_hat[j] * (-inv_stddev);
        }
        // Contribution to dmean from dvar: dL/dmean = dL/dvar * dvar/dmean
        // dvar/dmean = d/dmean (1/N * sum(x_j - mean)^2) = 1/N * sum(2 * (x_j - mean) * (-1)) = -2/N * sum(x_j - mean)
        // Since sum(x_j - mean) = 0, this term is 0 for dvar/dmean for a single instance's variance derivative w.r.t its own mean.
        // However, the formula for dmean for Batch Norm is typically:
        // dmean += dvar * (sum over i: -2 * (x_i - mean_batch)) / N
        // For layer norm, this simplifies due to sum(x_i - mean_cache) = 0 for this instance.

        // Re-evaluating dmean for layer norm (simplifies from standard BN formulas):
        // dL/dmean = (sum_j dL/dx_hat_j * (-1/std)) + (dL/dvar * dvar/dmean)
        // dvar/dmean for instance variance: dvar/dmean = (1/N) * sum_j 2*(x_j-mean)*(-1) = (-2/N) * sum_j(x_j-mean) = 0.
        // So for Layer Norm, dmean is simpler.
        // The effect of d(x_j - mean)/dmean is -1. Sum of dx_hat_j * (-1/stddev) is one part.
        // The effect of d(x_j - mean)/dmean on dVariance means sum(x_j - mean_cache) becomes zero.
        // The part of dmean from dVariance component (from dL/dvar * dvar/dmean) is indeed 0.
        // So, dmean = sum_j (dx_hat[j] * (-1/inv_stddev)) is correct for this part.
        // The other part of dmean comes from d(x_input_cache_[j] - mean_cache_) / dmean_cache_ from the dVariance chain, which sums to 0.


        // Gradient of loss w.r.t. input to BN layer (dx)
        std::vector<double> dx_input_sample(num_features_);
        for (int j = 0; j < num_features_; ++j) {
            // dL/dx_j = dL/dx_hat_j * dx_hat_j/dx_j + dL/dmean * dmean/dx_j + dL/dvar * dvar/dx_j
            // dx_hat_j/dx_j = inv_stddev
            // dmean/dx_j = 1/N
            // dvar/dx_j = 2 * (x_j - mean_cache_) / N
            dx_input_sample[j] = dx_hat[j] * inv_stddev +
                                 dmean * (1.0 / num_features_) +
                                 dvar * (2.0 * (x_input_cache_[j] - mean_cache_)) / num_features_;
        }

        // Update learnable parameters gamma and beta
        for (int j = 0; j < num_features_; ++j) {
            gamma_[j] -= learning_rate * dgamma[j];
            beta_[j] -= learning_rate * dbeta[j];
        }

        return dx_input_sample;
    }

}
