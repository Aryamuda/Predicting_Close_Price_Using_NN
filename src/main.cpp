// src/main.cpp
// Main entry point for the PricePredictorNN application.
// Current focus: Using NeuralNetwork with Momentum and L2 Weight Decay.

#include <iostream>
#include <vector>
#include <string>
#include <cmath>
#include <iomanip>
#include <stdexcept>
#include <numeric>
#include <algorithm>
#include <limits>
#include "Layer.h"
#include "NeuralNetwork.h"
#include "CSVReader.h"

//Helpers
void print_vector_main(const std::string& name, const std::vector<double>& vec, int limit = -1) {
    std::cout << name << ": [ ";
    int count = 0;
    for (size_t i = 0; i < vec.size(); ++i) {
        if (limit != -1 && count >= limit) {
            std::cout << "...";
            break;
        }
        std::cout << std::fixed << std::setprecision(5) << vec[i] << (i == vec.size() - 1 ? "" : ", ");
        count++;
    }
    std::cout << " ]" << std::endl;
}

void print_features_summary(const std::string& name, const std::vector<std::vector<double>>& features, int num_rows_to_print = 3, int num_cols_to_print = 5) {
    std::cout << name << " (Summary - first " << num_rows_to_print << " rows, first " << num_cols_to_print << " cols if available):" << std::endl;
    if (features.empty()) {
        std::cout << "  <No features loaded>" << std::endl;
        return;
    }
    std::cout << "  Total samples: " << features.size() << std::endl;
    if (!features.empty()) {
        std::cout << "  Features per sample: " << features[0].size() << std::endl;
    }
    for (int i = 0; i < std::min((int)features.size(), num_rows_to_print); ++i) {
        std::cout << "  Sample " << std::setw(3) << i << ": [ ";
        for(int j=0; j < std::min((int)features[i].size(), num_cols_to_print); ++j) {
            std::cout << std::fixed << std::setprecision(5) << features[i][j] << (j == std::min((int)features[i].size(), num_cols_to_print) - 1 ? "" : ", ");
        }
        if ((int)features[i].size() > num_cols_to_print) std::cout << "...";
        std::cout << " ]" << std::endl;
    }
}

// --- Normalization Functions
struct MinMaxVals {
    double min_val = std::numeric_limits<double>::max();
    double max_val = std::numeric_limits<double>::lowest();
};
std::vector<MinMaxVals> normalize_all_features_min_max(std::vector<std::vector<double>>& features) {
    if (features.empty() || features[0].empty()) { return {}; }
    size_t num_samples = features.size();
    size_t num_features = features[0].size();
    std::vector<MinMaxVals> feature_scaling_params(num_features);
    for (size_t j = 0; j < num_features; ++j) {
        for (size_t i = 0; i < num_samples; ++i) {
            if (features[i][j] < feature_scaling_params[j].min_val) feature_scaling_params[j].min_val = features[i][j];
            if (features[i][j] > feature_scaling_params[j].max_val) feature_scaling_params[j].max_val = features[i][j];
        }
    }
    for (size_t j = 0; j < num_features; ++j) {
        double min_v = feature_scaling_params[j].min_val;
        double max_v = feature_scaling_params[j].max_val;
        double range = max_v - min_v;
        if (std::abs(range) < 1e-9) {
            for (size_t i = 0; i < num_samples; ++i) features[i][j] = 0.0; // Or 0.5, or handle as constant
        } else {
            for (size_t i = 0; i < num_samples; ++i) features[i][j] = (features[i][j] - min_v) / range;
        }
    }
    return feature_scaling_params;
}
MinMaxVals normalize_target_variable_min_max(std::vector<double>& target_variable) {
    if (target_variable.empty()) { return {}; }
    MinMaxVals target_scaling_params;
    target_scaling_params.min_val = *std::min_element(target_variable.begin(), target_variable.end());
    target_scaling_params.max_val = *std::max_element(target_variable.begin(), target_variable.end());
    double min_v = target_scaling_params.min_val;
    double max_v = target_scaling_params.max_val;
    double range = max_v - min_v;
    if (std::abs(range) < 1e-9) {
        for (size_t i = 0; i < target_variable.size(); ++i) target_variable[i] = 0.0; // Or 0.5
    } else {
        for (size_t i = 0; i < target_variable.size(); ++i) target_variable[i] = (target_variable[i] - min_v) / range;
    }
    return target_scaling_params;
}
double denormalize_target_value_min_max(double normalized_value, const MinMaxVals& scaling_params) {
    double range = scaling_params.max_val - scaling_params.min_val;
    if (std::abs(range) < 1e-9) return scaling_params.min_val;
    return normalized_value * range + scaling_params.min_val;
}


// --- Time-Series Data Splitting Function ---
void time_series_split(
    const std::vector<std::vector<double>>& all_features,
    const std::vector<double>& all_targets,
    std::vector<std::vector<double>>& train_features,
    std::vector<double>& train_targets,
    std::vector<std::vector<double>>& val_features,
    std::vector<double>& val_targets,
    double validation_ratio = 0.2) {

    if (all_features.size() != all_targets.size()) {
        throw std::runtime_error("Time-series split: Features and targets must have the same number of samples.");
    }
    if (validation_ratio <= 0.0 || validation_ratio >= 1.0) {
        throw std::invalid_argument("Time-series split: Validation ratio must be between 0.0 and 1.0 (exclusive).");
    }
    if (all_features.empty()) {
        std::cout << "Warning: Time-series split called with empty dataset." << std::endl;
        train_features.clear(); train_targets.clear();
        val_features.clear(); val_targets.clear();
        return;
    }

    size_t total_samples = all_features.size();
    size_t validation_count = static_cast<size_t>(static_cast<double>(total_samples) * validation_ratio);
    size_t training_count = total_samples - validation_count;

    if (training_count == 0 || (validation_count > 0 && validation_count == total_samples) ) {
        std::cerr << "Warning: Time-series split resulted in an empty training set or all data as validation. Adjust ratio or dataset size." << std::endl;
        if (total_samples > 10) {
            validation_count = std::max((size_t)1, static_cast<size_t>(total_samples * 0.1));
            training_count = total_samples - validation_count;
            if (training_count == 0) {
                 training_count = total_samples; validation_count = 0;
            }
        } else {
            training_count = total_samples;
            validation_count = 0;
        }
         std::cout << "Adjusted split: Train=" << training_count << ", Val=" << validation_count << std::endl;
    }

    train_features.assign(all_features.begin(), all_features.begin() + training_count);
    train_targets.assign(all_targets.begin(), all_targets.begin() + training_count);

    if (validation_count > 0 && training_count > 0) {
        val_features.assign(all_features.begin() + training_count, all_features.end());
        val_targets.assign(all_targets.begin() + training_count, all_targets.end());
    } else {
        val_features.clear();
        val_targets.clear();
    }

    std::cout << "Time-series split: Total=" << total_samples
              << ", Train=" << train_features.size()
              << ", Validation=" << val_features.size() << std::endl;
}


// --- Main Training and Evaluation Orchestrator ---
void run_main_training_pipeline() {
    std::cout << "\n--- Neural Network Training Pipeline ---" << std::endl;

    std::string filename = "XAUUSD.csv";
    std::vector<std::vector<double>> features_all_original;
    std::vector<double> target_prices_all_original;
    std::vector<std::vector<double>> features_all_norm;
    std::vector<double> target_prices_all_norm;
    int target_column_idx = 3;

    try {
        Predicting_Close_Price_Using_NN::CSVReader::read_regression_data(filename, features_all_original, target_prices_all_original, target_column_idx);
        std::cout << "Successfully loaded " << features_all_original.size() << " original samples from " << filename << "." << std::endl;
        if (features_all_original.empty()) { std::cerr << "No data loaded. Exiting." << std::endl; return; }
        features_all_norm = features_all_original;
        target_prices_all_norm = target_prices_all_original;
    } catch (const std::exception& e) {
        std::cerr << "Error loading CSV data: " << e.what() << std::endl; return;
    }

    std::cout << "\nNormalizing features and target variable..." << std::endl;
    std::vector<MinMaxVals> feature_scaling_info = normalize_all_features_min_max(features_all_norm);
    MinMaxVals target_scaling_info = normalize_target_variable_min_max(target_prices_all_norm);

    std::cout << "Normalization complete." << std::endl;
    std::cout << "Target scaling (used for normalization): min=" << target_scaling_info.min_val << ", max=" << target_scaling_info.max_val << std::endl;

    std::vector<std::vector<double>> X_train_norm, X_val_norm;
    std::vector<double> y_train_norm, y_val_norm;
    double validation_split_ratio = 0.2;
    time_series_split(features_all_norm, target_prices_all_norm, X_train_norm, y_train_norm, X_val_norm, y_val_norm, validation_split_ratio);

    std::vector<std::vector<double>> X_train_orig, X_val_orig;
    std::vector<double> y_train_orig, y_val_orig;
    time_series_split(features_all_original, target_prices_all_original, X_train_orig, y_train_orig, X_val_orig, y_val_orig, validation_split_ratio);

    if (X_train_norm.empty()) { std::cerr << "Normalized training set is empty after split. Exiting." << std::endl; return; }

    int input_dim = static_cast<int>(X_train_norm[0].size());
    std::vector<int> layer_sizes = {input_dim, 64, 32, 1};
    std::vector<std::string> activations = {"relu", "relu", "linear"};

    // --- Hyperparameters for training ---
    double learning_rate = 0.001;
    double dropout_rate = 0.1;
    double momentum_coefficient = 0.9;      // Common value for momentum
    double weight_decay_coefficient = 1e-4; // Common value for L2 regularization

    std::cout << "\nInitializing Neural Network with:" << std::endl;
    std::cout << "  Learning Rate: " << learning_rate << std::endl;
    std::cout << "  Dropout Rate (for hidden layers): " << dropout_rate << std::endl;
    std::cout << "  Momentum Coefficient: " << momentum_coefficient << std::endl;
    std::cout << "  Weight Decay (L2) Coefficient: " << weight_decay_coefficient << std::endl;

    Predicting_Close_Price_Using_NN::NeuralNetwork nn(layer_sizes,
                                     activations,
                                     learning_rate,
                                     dropout_rate,
                                     momentum_coefficient,      // Pass momentum
                                     weight_decay_coefficient); // Pass weight decay

    int num_epochs = 200;
    int print_every_n_epochs = 20;

    nn.train(X_train_norm, y_train_norm, num_epochs, print_every_n_epochs, X_val_norm, y_val_norm);

    std::cout << "\n--- Final Evaluation Metrics (on Normalized Data) ---" << std::endl;
    double final_train_mse_norm = nn.evaluate_regression(X_train_norm, y_train_norm);
    std::cout << "Final Training MSE (Normalized): " << std::fixed << std::setprecision(8) << final_train_mse_norm << std::endl;

    if (!X_val_norm.empty()) {
        double final_val_mse_norm = nn.evaluate_regression(X_val_norm, y_val_norm);
        std::cout << "Final Validation MSE (Normalized): " << std::fixed << std::setprecision(8) << final_val_mse_norm << std::endl;
    }

    std::cout << "\nPredictions on first few validation samples (denormalized):" << std::endl;
    std::cout << std::setw(18) << "True (Original)" << std::setw(20) << "Predicted (DeNorm)" << std::setw(18) << "Abs Difference" << std::endl;
    std::cout << "------------------------------------------------------------------------------------" << std::endl;

    int eval_samples_count = 0;
    std::vector<std::vector<double>>* features_for_eval = &X_val_norm;
    std::vector<double>* original_targets_for_eval = &y_val_orig;

    if (X_val_norm.empty() && !X_train_norm.empty()) {
        std::cout << "(No validation set, showing predictions on first few training samples)" << std::endl;
        features_for_eval = &X_train_norm;
        original_targets_for_eval = &y_train_orig;
    }
    eval_samples_count = std::min((int)features_for_eval->size(), 5);

    for (int i = 0; i < eval_samples_count; ++i) {
        const auto& current_features_norm = (*features_for_eval)[i];
        if (current_features_norm.empty()) continue;

        std::vector<double> y_pred_vec_norm = nn.predict(current_features_norm, false);
        if (y_pred_vec_norm.empty()) continue;

        double y_pred_denorm = denormalize_target_value_min_max(y_pred_vec_norm[0], target_scaling_info);
        double y_true_original = (*original_targets_for_eval)[i];
        double diff_denorm = std::abs(y_pred_denorm - y_true_original);

        std::cout << std::fixed << std::setprecision(5)
                  << std::setw(18) << y_true_original
                  << std::setw(20) << y_pred_denorm
                  << std::setw(18) << diff_denorm
                  << std::endl;
    }
}


int main() {
    std::cout << "Initializing main..." << std::endl;

    std::cout << "\nRunning Main Training Pipeline" << std::endl;
    run_main_training_pipeline();
    std::cout << "Main Training Pipeline Complete" << std::endl << std::endl;

    return 0;
}

