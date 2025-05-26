#include "NeuralNetwork.h"
#include "Loss.h"
#include <iostream>
#include <stdexcept>
#include <numeric>
#include <algorithm>
#include <random>
#include <iomanip>

namespace Predicting_Close_Price_Using_NN {

    NeuralNetwork::NeuralNetwork(const std::vector<int>& layer_sizes,
                                 const std::vector<std::string>& activations,
                                 double learning_rate,
                                 double dropout_rate,
                                 double momentum_coeff,
                                 double weight_decay_coeff,
                                 bool apply_batch_norm) // New parameter
        : learning_rate_(learning_rate),
          momentum_coeff_(momentum_coeff),
          weight_decay_coeff_(weight_decay_coeff) {

        // --- Validations ---
        if (layer_sizes.size() < 2) {
            throw std::invalid_argument("NeuralNetwork requires at least an input and an output layer (layer_sizes.size() must be >= 2).");
        }
        if (layer_sizes.size() - 1 != activations.size()) {
            throw std::invalid_argument("Number of activation functions must be one less than the number of layer sizes.");
        }
        // ... (other validations for lr, dropout, momentum, weight_decay) ...
        if (learning_rate <= 0.0) {
            throw std::invalid_argument("Learning rate must be positive.");
        }
        if (dropout_rate < 0.0 || dropout_rate >= 1.0) {
            throw std::invalid_argument("Dropout rate must be in [0.0, 1.0).");
        }
        if (momentum_coeff < 0.0 || momentum_coeff >= 1.0) {
            throw std::invalid_argument("Momentum coefficient must be in [0.0, 1.0).");
        }
        if (weight_decay_coeff < 0.0) {
            throw std::invalid_argument("Weight decay coefficient must be non-negative.");
        }


        // --- Create Layers and BatchNormLayers ---
        size_t num_dense_layers = activations.size();
        use_bn_for_layer_.resize(num_dense_layers, false);

        for (size_t i = 0; i < num_dense_layers; ++i) {
            int input_dim_for_layer = layer_sizes[i];
            int output_dim_for_layer = layer_sizes[i + 1];
            std::string activation_for_layer = activations[i];

            double current_layer_dropout_rate = 0.0;
            // Apply dropout only to hidden layers (not the final output layer)
            if (i < num_dense_layers - 1) {
                current_layer_dropout_rate = dropout_rate;
            }

            try {
                layers_.emplace_back(input_dim_for_layer,
                                     output_dim_for_layer,
                                     activation_for_layer,
                                     current_layer_dropout_rate);

                // Add BatchNorm layer after hidden layers' activations if requested
                if (apply_batch_norm && i < num_dense_layers - 1) {
                    bn_layers_.emplace_back(output_dim_for_layer); // BN features = output of current dense layer
                    use_bn_for_layer_[i] = true;
                     std::cout << "Info: Added BatchNorm after Layer " << i << " (output_size: " << output_dim_for_layer << ")" << std::endl;
                } else {

                }

            } catch (const std::exception& e) {
                throw std::runtime_error("Failed to create layer " + std::to_string(i) + ": " + e.what());
            }
        }

        // Correctly size bn_layers_ to match layers_ and initialize dummy layers where BN is not used.
        // This makes indexing consistent.
        if (apply_batch_norm) {
            std::vector<BatchNormLayer> temp_bn_layers;
            for(size_t i = 0; i < num_dense_layers; ++i) {
                if (i < num_dense_layers - 1) { // Apply to hidden layers
                     temp_bn_layers.emplace_back(layers_[i].output_size_);
                     use_bn_for_layer_[i] = true; // Mark as used
                } else { // For output layer or if not applying BN
                     temp_bn_layers.emplace_back(layers_[i].output_size_); // Dummy, won't be used if use_bn_for_layer_[i] is false
                     use_bn_for_layer_[i] = false; // Ensure it's marked false for output layer
                }
            }
            bn_layers_ = std::move(temp_bn_layers);
        }


        if (momentum_coeff_ > 0.0 && !layers_.empty()) {
            initialize_momentum_velocities();
        }
    }

    void NeuralNetwork::initialize_momentum_velocities() {
        if (layers_.empty()) return;
        velocity_weights_.resize(layers_.size());
        velocity_biases_.resize(layers_.size());
        for (size_t l = 0; l < layers_.size(); ++l) {
            velocity_weights_[l].resize(layers_[l].output_size_, std::vector<double>(layers_[l].input_size_, 0.0));
            velocity_biases_[l].resize(layers_[l].output_size_, 0.0);
        }
    }


    std::vector<double> NeuralNetwork::predict(const std::vector<double>& input_data, bool training_mode) {
        if (layers_.empty()) {
            throw std::runtime_error("NeuralNetwork::predict - Network has no layers.");
        }
        if (static_cast<int>(input_data.size()) != layers_[0].input_size_) {
            throw std::invalid_argument("NeuralNetwork::predict - Input data size does not match network's input layer size.");
        }

        std::vector<double> current_output = input_data;
        for (size_t i = 0; i < layers_.size(); ++i) {
            current_output = layers_[i].forward(current_output, training_mode);
            // Apply BatchNorm if enabled for this layer
            if (use_bn_for_layer_[i]) { // Check the flag
                 // Ensure bn_layers_ has an entry for this layer if use_bn_for_layer_[i] is true
                if (i < bn_layers_.size()) { // Safety check, should always be true if initialized correctly
                    current_output = bn_layers_[i].forward(current_output, training_mode);
                } else {
                    // This indicates a logic error in constructor if use_bn_for_layer_ is true but no bn_layer exists
                     std::cerr << "Warning: use_bn_for_layer_[" << i << "] is true, but no corresponding bn_layer found. Skipping BN." << std::endl;
                }
            }
        }
        return current_output;
    }

    void NeuralNetwork::train_one_sample(const std::vector<double>& x_input, double y_true_price) {
        if (layers_.empty()) { /* ... */ }

        std::vector<double> y_pred_vector = predict(x_input, true);
        if (y_pred_vector.size() != 1) { /* ... */ }
        double y_pred_price = y_pred_vector[0];
        double loss_derivative = Loss::mean_squared_error_derivative(y_true_price, y_pred_price);
        std::vector<double> current_error_signal = {loss_derivative};

        // Backward pass through layers and BatchNorm layers
        for (int i = static_cast<int>(layers_.size()) - 1; i >= 0; --i) {
            // If BN was applied after this layer during forward pass, backprop through BN first
            if (use_bn_for_layer_[i]) {
                if (static_cast<size_t>(i) < bn_layers_.size()) { // Safety check
                    // current_error_signal is dError/d(Output of BN_i)
                    // We need dError/d(Input of BN_i), which is dError/d(Output of Layer_i)
                    current_error_signal = bn_layers_[i].backward(current_error_signal, learning_rate_); // BN updates its own gamma/beta
                }
            }
            // current_error_signal is now dError/d(Output of Layer_i)
            current_error_signal = layers_[i].backward(current_error_signal);
        }

        // Update weights and biases for each dense layer (Layer objects)
        for (size_t l = 0; l < layers_.size(); ++l) {
            Layer& current_layer = layers_[l];
            for (int i_neuron = 0; i_neuron < current_layer.output_size_; ++i_neuron) {
                double grad_b = current_layer.delta_[i_neuron];
                if (momentum_coeff_ > 0.0) {
                    if (l < velocity_biases_.size() && i_neuron < static_cast<int>(velocity_biases_[l].size())) {
                       velocity_biases_[l][i_neuron] = momentum_coeff_ * velocity_biases_[l][i_neuron] - learning_rate_ * grad_b;
                       current_layer.biases_[i_neuron] += velocity_biases_[l][i_neuron];
                    } else { current_layer.biases_[i_neuron] -= learning_rate_ * grad_b; }
                } else { current_layer.biases_[i_neuron] -= learning_rate_ * grad_b; }

                for (int j_input = 0; j_input < current_layer.input_size_; ++j_input) {
                    double grad_w = current_layer.delta_[i_neuron] * current_layer.input_cache_[j_input];
                    if (weight_decay_coeff_ > 0.0) {
                        grad_w += weight_decay_coeff_ * current_layer.weights_[i_neuron][j_input];
                    }
                    if (momentum_coeff_ > 0.0) {
                        if (l < velocity_weights_.size() && i_neuron < static_cast<int>(velocity_weights_[l].size()) && j_input < static_cast<int>(velocity_weights_[l][i_neuron].size())) {
                            velocity_weights_[l][i_neuron][j_input] = momentum_coeff_ * velocity_weights_[l][i_neuron][j_input] - learning_rate_ * grad_w;
                            current_layer.weights_[i_neuron][j_input] += velocity_weights_[l][i_neuron][j_input];
                        } else { current_layer.weights_[i_neuron][j_input] -= learning_rate_ * grad_w;}
                    } else { current_layer.weights_[i_neuron][j_input] -= learning_rate_ * grad_w; }
                }
            }
        }
    }

    void NeuralNetwork::train(
        const std::vector<std::vector<double>>& X_train,
        const std::vector<double>& y_train,
        int epochs,
        int print_every_n_epochs,
        const std::vector<std::vector<double>>& X_val,
        const std::vector<double>& y_val) {
        if (X_train.size() != y_train.size()) { throw std::invalid_argument("NN::train - X/y mismatch."); }
        if (X_train.empty()) { std::cout << "Warning: NN::train - empty training data." << std::endl; return; }
        if (epochs <= 0) { std::cout << "Warning: NN::train - non-positive epochs." << std::endl; return; }
        bool has_validation_data = !X_val.empty() && !y_val.empty();
        if (has_validation_data && (X_val.size() != y_val.size())) { throw std::invalid_argument("NN::train - X_val/y_val mismatch."); }

        std::cout << "Starting Neural Network training..." << std::endl;
        // Add BN status to printout
        bool bn_active_somewhere = false;
        for(bool use_bn : use_bn_for_layer_) if(use_bn) bn_active_somewhere = true;
        if(bn_active_somewhere) std::cout << " - Batch Normalization: Enabled for hidden layers" << std::endl;


        for (int epoch = 0; epoch < epochs; ++epoch) {
            for (size_t i = 0; i < X_train.size(); ++i) {
                if (X_train[i].empty() || static_cast<int>(X_train[i].size()) != layers_[0].input_size_) {
                    continue;
                }
                train_one_sample(X_train[i], y_train[i]);
            }
            if (print_every_n_epochs > 0 && ((epoch + 1) % print_every_n_epochs == 0 || epoch == 0 || epoch == epochs - 1)) {
                double train_mse = evaluate_regression(X_train, y_train);
                std::cout << "Epoch " << std::setw(4) << (epoch + 1) << "/" << epochs
                          << " | Train MSE (Norm): " << std::fixed << std::setprecision(8) << train_mse;
                if (has_validation_data && !X_val.empty()) {
                    double val_mse = evaluate_regression(X_val, y_val);
                    std::cout << " | Val MSE (Norm): " << std::fixed << std::setprecision(8) << val_mse;
                }
                std::cout << std::endl;
            }
        }
        std::cout << "Neural Network training complete." << std::endl;
    }


    double NeuralNetwork::evaluate_regression(
        const std::vector<std::vector<double>>& X_data,
        const std::vector<double>& y_true_targets) {
        if (X_data.size() != y_true_targets.size()) { throw std::invalid_argument("eval_reg: X/y mismatch."); }
        if (X_data.empty()) { return 0.0; }
        if (layers_.empty()) { throw std::runtime_error("eval_reg: No layers."); }
        double total_squared_error = 0.0;
        int valid_predictions = 0;
        for (size_t i = 0; i < X_data.size(); ++i) {
            if (X_data[i].empty() || static_cast<int>(X_data[i].size()) != layers_[0].input_size_) { continue; }
            std::vector<double> y_pred_vec = predict(X_data[i], false);
            if (y_pred_vec.size() != 1) { continue; }
            double y_pred = y_pred_vec[0];
            double error = y_pred - y_true_targets[i];
            total_squared_error += error * error;
            valid_predictions++;
        }
        if (valid_predictions == 0) { return std::numeric_limits<double>::quiet_NaN(); }
        return total_squared_error / static_cast<double>(valid_predictions);
    }

}
