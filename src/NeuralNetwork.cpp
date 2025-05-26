// src/NeuralNetwork.cpp
#include "NeuralNetwork.h"
#include "Loss.h"
#include "BatchDataLoader.h"
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
                                 bool apply_batch_norm)
        : learning_rate_(learning_rate),
          momentum_coeff_(momentum_coeff),
          weight_decay_coeff_(weight_decay_coeff) {

        //Validations
        if (layer_sizes.size() < 2) { throw std::invalid_argument("NN Ctor: Min 2 layer_sizes."); }
        if (layer_sizes.size() - 1 != activations.size()) { throw std::invalid_argument("NN Ctor: layer_sizes/activations mismatch."); }
        if (learning_rate <= 0.0) { throw std::invalid_argument("NN Ctor: LR must be > 0."); }
        if (dropout_rate < 0.0 || dropout_rate >= 1.0) { throw std::invalid_argument("NN Ctor: Dropout [0,1)."); }
        if (momentum_coeff < 0.0 || momentum_coeff >= 1.0) { throw std::invalid_argument("NN Ctor: Momentum [0,1)."); }
        if (weight_decay_coeff < 0.0) { throw std::invalid_argument("NN Ctor: Weight decay >= 0."); }

        //Create Layers and BatchNormLayers
        size_t num_dense_layers = activations.size();
        use_bn_for_layer_.resize(num_dense_layers, false);
        // Do not resize bn_layers_ here with default constructor. Reserve space instead.
        bn_layers_.reserve(num_dense_layers);

        for (size_t i = 0; i < num_dense_layers; ++i) {
            int input_dim = layer_sizes[i];
            int output_dim = layer_sizes[i + 1]; // Output dimension of the current dense layer
            std::string activation = activations[i];
            double current_dropout = (i < num_dense_layers - 1) ? dropout_rate : 0.0;

            try {
                // Create the dense layer
                layers_.emplace_back(input_dim, output_dim, activation, current_dropout);
                bn_layers_.emplace_back(output_dim); // Always construct with parameters

                if (apply_batch_norm && i < num_dense_layers - 1) { // Apply BN only after hidden layers
                    use_bn_for_layer_[i] = true;
                } else {
                    use_bn_for_layer_[i] = false; // No BN for output layer or if disabled globally
                }
            } catch (const std::exception& e) {
                throw std::runtime_error("NN Ctor: Layer " + std::to_string(i) + " or BNLayer creation failed: " + e.what());
            }
        }

        if (!layers_.empty()) {
            if (momentum_coeff_ > 0.0) {
                initialize_momentum_velocities();
            }
            initialize_accumulated_gradients();
        }
    }

    void NeuralNetwork::initialize_momentum_velocities() {
        velocity_weights_.resize(layers_.size());
        velocity_biases_.resize(layers_.size());
        for (size_t l = 0; l < layers_.size(); ++l) {
            velocity_weights_[l].resize(layers_[l].output_size_, std::vector<double>(layers_[l].input_size_, 0.0));
            velocity_biases_[l].resize(layers_[l].output_size_, 0.0);
        }
    }
    void NeuralNetwork::initialize_accumulated_gradients() {
        accumulated_weight_gradients_.resize(layers_.size());
        accumulated_bias_gradients_.resize(layers_.size());
        for (size_t l = 0; l < layers_.size(); ++l) {
            accumulated_weight_gradients_[l].resize(layers_[l].output_size_, std::vector<double>(layers_[l].input_size_, 0.0));
            accumulated_bias_gradients_[l].resize(layers_[l].output_size_, 0.0);
        }
    }
    void NeuralNetwork::reset_accumulated_gradients() {
        for (size_t l = 0; l < layers_.size(); ++l) {
            for (size_t i = 0; i < accumulated_weight_gradients_[l].size(); ++i) {
                std::fill(accumulated_weight_gradients_[l][i].begin(), accumulated_weight_gradients_[l][i].end(), 0.0);
            }
            std::fill(accumulated_bias_gradients_[l].begin(), accumulated_bias_gradients_[l].end(), 0.0);
        }
    }

    std::vector<double> NeuralNetwork::predict(const std::vector<double>& input_data, bool training_mode) {
        if (layers_.empty()) { throw std::runtime_error("NN predict: No layers."); }
        if (static_cast<int>(input_data.size()) != layers_[0].input_size_) { throw std::invalid_argument("NN predict: Input size mismatch.");}
        std::vector<double> current_output = input_data;
        for (size_t i = 0; i < layers_.size(); ++i) {
            current_output = layers_[i].forward(current_output, training_mode);
            if (use_bn_for_layer_[i]) {
                 // Ensure bn_layers_ has an entry for this layer if use_bn_for_layer_[i] is true
                if (i < bn_layers_.size()) {
                    current_output = bn_layers_[i].forward(current_output, training_mode);
                } else {
                     std::cerr << "Warning: use_bn_for_layer_[" << i << "] is true, but bn_layers_ size is " << bn_layers_.size() << ". Skipping BN." << std::endl;
                }
            }
        }
        return current_output;
    }

    void NeuralNetwork::process_sample_and_accumulate_gradients(const std::vector<double>& x_input, double y_true_price) {
        if (layers_.empty()) { throw std::runtime_error("NN process_sample: No layers."); }
        std::vector<double> y_pred_vector = predict(x_input, true);
        if (y_pred_vector.size() != 1) { throw std::runtime_error("NN process_sample: Expected single output."); }
        double y_pred_price = y_pred_vector[0];
        double loss_derivative = Loss::mean_squared_error_derivative(y_true_price, y_pred_price);
        std::vector<double> current_error_signal = {loss_derivative};
        for (int i = static_cast<int>(layers_.size()) - 1; i >= 0; --i) {
            if (use_bn_for_layer_[i]) {
                if (static_cast<size_t>(i) < bn_layers_.size()) {
                    current_error_signal = bn_layers_[i].backward(current_error_signal, learning_rate_);
                }
            }
            current_error_signal = layers_[i].backward(current_error_signal);
        }
        for (size_t l = 0; l < layers_.size(); ++l) {
            Layer& current_layer = layers_[l];
            for (int i_neuron = 0; i_neuron < current_layer.output_size_; ++i_neuron) {
                accumulated_bias_gradients_[l][i_neuron] += current_layer.delta_[i_neuron];
                for (int j_input = 0; j_input < current_layer.input_size_; ++j_input) {
                    accumulated_weight_gradients_[l][i_neuron][j_input] += current_layer.delta_[i_neuron] * current_layer.input_cache_[j_input];
                }
            }
        }
    }

    void NeuralNetwork::apply_accumulated_gradients(size_t batch_size) {
        if (batch_size == 0) { throw std::invalid_argument("Batch size cannot be zero when applying gradients."); }
        if (layers_.empty()) return;
        double avg_factor = 1.0 / static_cast<double>(batch_size);
        for (size_t l = 0; l < layers_.size(); ++l) {
            Layer& current_layer = layers_[l];
            for (int i = 0; i < current_layer.output_size_; ++i) {
                double avg_grad_b = accumulated_bias_gradients_[l][i] * avg_factor;
                if (momentum_coeff_ > 0.0) {
                    if (l < velocity_biases_.size() && i < static_cast<int>(velocity_biases_[l].size())) {
                       velocity_biases_[l][i] = momentum_coeff_ * velocity_biases_[l][i] - learning_rate_ * avg_grad_b;
                       current_layer.biases_[i] += velocity_biases_[l][i];
                    } else { current_layer.biases_[i] -= learning_rate_ * avg_grad_b; }
                } else { current_layer.biases_[i] -= learning_rate_ * avg_grad_b; }
                for (int j = 0; j < current_layer.input_size_; ++j) {
                    double avg_grad_w = accumulated_weight_gradients_[l][i][j] * avg_factor;
                    if (weight_decay_coeff_ > 0.0) {
                        avg_grad_w += weight_decay_coeff_ * current_layer.weights_[i][j];
                    }
                    if (momentum_coeff_ > 0.0) {
                         if (l < velocity_weights_.size() && i < static_cast<int>(velocity_weights_[l].size()) && j < static_cast<int>(velocity_weights_[l][i].size())) {
                            velocity_weights_[l][i][j] = momentum_coeff_ * velocity_weights_[l][i][j] - learning_rate_ * avg_grad_w;
                            current_layer.weights_[i][j] += velocity_weights_[l][i][j];
                        } else { current_layer.weights_[i][j] -= learning_rate_ * avg_grad_w;}
                    } else { current_layer.weights_[i][j] -= learning_rate_ * avg_grad_w; }
                }
            }
        }
        reset_accumulated_gradients();
    }

    void NeuralNetwork::train(
        const std::vector<std::vector<double>>& X_train,
        const std::vector<double>& y_train,
        int epochs,
        size_t batch_size,
        int print_every_n_epochs,
        const std::vector<std::vector<double>>& X_val,
        const std::vector<double>& y_val) {

        if (X_train.size() != y_train.size()) { throw std::invalid_argument("NN::train - X/y mismatch."); }
        if (X_train.empty()) { std::cout << "Warning: NN::train - empty training data." << std::endl; return; }
        if (epochs <= 0) { std::cout << "Warning: NN::train - non-positive epochs." << std::endl; return; }
        if (batch_size == 0) { throw std::invalid_argument("NN::train - batch_size cannot be zero.");}

        bool has_validation_data = !X_val.empty() && !y_val.empty();
        if (has_validation_data && (X_val.size() != y_val.size())) { throw std::invalid_argument("NN::train - X_val/y_val mismatch."); }

        std::cout << "Starting Neural Network training with mini-batches..." << std::endl;
        std::cout << " - Training samples: " << X_train.size() << std::endl;
        if(has_validation_data) { std::cout << " - Validation samples: " << X_val.size() << std::endl; }
        std::cout << " - Epochs: " << epochs << std::endl;
        std::cout << " - Batch Size: " << batch_size << std::endl;
        std::cout << " - Learning rate: " << learning_rate_ << std::endl;
        if (momentum_coeff_ > 0.0) std::cout << " - Momentum: " << momentum_coeff_ << std::endl;
        if (weight_decay_coeff_ > 0.0) std::cout << " - Weight Decay (L2): " << weight_decay_coeff_ << std::endl;
        bool bn_active = false; for(bool use_bn : use_bn_for_layer_) if(use_bn) bn_active = true;
        if(bn_active) std::cout << " - Batch Normalization: Enabled for hidden layers" << std::endl;

        BatchDataLoader data_loader(X_train, y_train, batch_size, false);

        for (int epoch = 0; epoch < epochs; ++epoch) {
            data_loader.reset();
            std::vector<std::vector<double>> current_X_batch;
            std::vector<double> current_y_batch;
            while(data_loader.next_batch(current_X_batch, current_y_batch)) {
                if (current_X_batch.empty()) continue;
                reset_accumulated_gradients();
                for (size_t i = 0; i < current_X_batch.size(); ++i) {
                    if (current_X_batch[i].empty() || static_cast<int>(current_X_batch[i].size()) != layers_[0].input_size_) {
                        std::cerr << "Warning: NN::train - Skipping sample in batch due to incorrect feature size. Epoch " << epoch + 1 << std::endl;
                        continue;
                    }
                    process_sample_and_accumulate_gradients(current_X_batch[i], current_y_batch[i]);
                }
                if (!current_X_batch.empty()) {
                    apply_accumulated_gradients(current_X_batch.size());
                }
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
