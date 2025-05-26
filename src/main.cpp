#include <iostream>
#include <vector>
#include <string>
#include <cmath>
#include <iomanip>
#include <stdexcept>
#include <numeric>
#include <algorithm>
#include <limits>

#include "Loss.h"
#include "Layer.h"
#include "NeuralNetwork.h"
#include "CSVReader.h"

// --- Temporary Test Code / Helpers ---
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

//Normalization Functions
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
            for (size_t i = 0; i < num_samples; ++i) features[i][j] = 0.0;
        } else {
            for (size_t i = 0; i < num_samples; ++i) features[i][j] = (features[i][j] - min_v) / range;
        }
    }
    return feature_scaling_params;
}
MinMaxVals normalize_target_variable_min_max(std::vector<double>& target_variable) { /* ... (implementation from previous step) ... */
    if (target_variable.empty()) { return {}; }
    MinMaxVals target_scaling_params;
    target_scaling_params.min_val = *std::min_element(target_variable.begin(), target_variable.end());
    target_scaling_params.max_val = *std::max_element(target_variable.begin(), target_variable.end());
    double min_v = target_scaling_params.min_val;
    double max_v = target_scaling_params.max_val;
    double range = max_v - min_v;
    if (std::abs(range) < 1e-9) {
        for (size_t i = 0; i < target_variable.size(); ++i) target_variable[i] = 0.0;
    } else {
        for (size_t i = 0; i < target_variable.size(); ++i) target_variable[i] = (target_variable[i] - min_v) / range;
    }
    return target_scaling_params;
}
double denormalize_target_value_min_max(double normalized_value, const MinMaxVals& scaling_params) { /* ... (implementation from previous step) ... */
    double range = scaling_params.max_val - scaling_params.min_val;
    if (std::abs(range) < 1e-9) return scaling_params.min_val;
    return normalized_value * range + scaling_params.min_val;
}


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

    if (training_count == 0 || validation_count == 0) {
        std::cerr << "Warning: Time-series split resulted in an empty training or validation set. Adjust ratio or dataset size." << std::endl;
        // Default to using all for training if one set is empty due to small dataset/ratio
        if (training_count == 0 && total_samples > 0) {
             training_count = total_samples;
             validation_count = 0;
        } else if (validation_count == 0 && total_samples > 0) {
            // This case is less problematic
        }
    }

    train_features.assign(all_features.begin(), all_features.begin() + training_count);
    train_targets.assign(all_targets.begin(), all_targets.begin() + training_count);

    if (validation_count > 0) {
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


//Main Training Function
void run_training_and_evaluation() {
    std::cout << "\n--- Neural Network Training and Evaluation ---" << std::endl;

    std::string filename = "XAUUSD.csv"; // Filename
    std::vector<std::vector<double>> features_all;
    std::vector<double> target_prices_all;
    int target_column_idx = 3; // Close column index

    try {
        Predicting_Close_Price_Using_NN::CSVReader::read_regression_data(filename, features_all, target_prices_all, target_column_idx);
        std::cout << "Successfully loaded " << features_all.size() << " samples from " << filename << "." << std::endl;
        if (features_all.empty()) { std::cerr << "No data loaded. Exiting." << std::endl; return; }
    } catch (const std::exception& e) {
        std::cerr << "Error loading CSV data: " << e.what() << std::endl; return;
    }

    // Normalize features and target
    std::vector<MinMaxVals> feature_scaling_info = normalize_all_features_min_max(features_all);
    MinMaxVals target_scaling_info = normalize_target_variable_min_max(target_prices_all);

    //Split data into training and validation sets
    std::vector<std::vector<double>> X_train, X_val;
    std::vector<double> y_train, y_val;
    double validation_split_ratio = 0.2; // Use 20% for validation
    time_series_split(features_all, target_prices_all, X_train, y_train, X_val, y_val, validation_split_ratio);

    if (X_train.empty()) { std::cerr << "Training set is empty after split. Exiting." << std::endl; return; }

    int input_dim = static_cast<int>(X_train[0].size());
    std::vector<int> layer_sizes = {input_dim, 64, 32, 1};
    std::vector<std::string> activations = {"relu", "relu", "linear"};
    double learning_rate = 0.001;
    double dropout_rate = 0.1;

    Predicting_Close_Price_Using_NN::NeuralNetwork nn(layer_sizes, activations, learning_rate, dropout_rate);

    int num_epochs = 200;
    int print_every_n_epochs = 10;

    std::cout << "\nStarting training on " << X_train.size() << " training samples."
              << (X_val.empty() ? "" : " Validating on " + std::to_string(X_val.size()) + " samples.") << std::endl;

    for (int epoch = 0; epoch < num_epochs; ++epoch) {
        double train_epoch_loss_norm = 0.0;
        for (size_t i = 0; i < X_train.size(); ++i) {
            // (Error checking for X_train[i] as before)
            std::vector<double> y_pred_vec_norm = nn.predict(X_train[i], false);
            if(y_pred_vec_norm.empty()) continue;
            train_epoch_loss_norm += Predicting_Close_Price_Using_NN::Loss::mean_squared_error(y_train[i], y_pred_vec_norm[0]);
            nn.train_one_sample(X_train[i], y_train[i]);
        }
        train_epoch_loss_norm /= X_train.size();

        if ((epoch + 1) % print_every_n_epochs == 0 || epoch == 0 || epoch == num_epochs - 1) {
            std::cout << "Epoch " << std::setw(4) << (epoch + 1) << "/" << num_epochs
                      << " | Train MSE (Norm): " << std::fixed << std::setprecision(8) << train_epoch_loss_norm;

            // Basic validation loss
            if (!X_val.empty()) {
                double val_epoch_loss_norm = 0.0;
                for (size_t i = 0; i < X_val.size(); ++i) {
                     if (X_val[i].empty() || static_cast<int>(X_val[i].size()) != input_dim) continue;
                     std::vector<double> y_val_pred_vec_norm = nn.predict(X_val[i], false);
                     if(y_val_pred_vec_norm.empty()) continue;
                     val_epoch_loss_norm += Predicting_Close_Price_Using_NN::Loss::mean_squared_error(y_val[i], y_val_pred_vec_norm[0]);
                }
                val_epoch_loss_norm /= X_val.size();
                 std::cout << " | Val MSE (Norm): " << std::fixed << std::setprecision(8) << val_epoch_loss_norm;
            }
            std::cout << std::endl;
        }
    }

    // Simple evaluation on first few validation samples (denormalized)
    std::cout << "\nPredictions on first few validation samples (denormalized):" << std::endl;
    std::cout << std::setw(18) << "True (Original)" << std::setw(20) << "Predicted (DeNorm)" << std::endl;
    std::cout << "---------------------------------------------------------" << std::endl;

    // For denormalized comparison, we need original unnormalized validation targets
    std::vector<std::vector<double>> _, __; // dummy feature vectors
    std::vector<double> original_train_targets, original_val_targets;
    std::vector<double> original_all_targets_copy = target_prices_all;
    Predicting_Close_Price_Using_NN::CSVReader::read_regression_data(filename, _, original_all_targets_copy, target_column_idx); // Reload for original
    time_series_split(features_all, original_all_targets_copy, _, original_train_targets, __, original_val_targets, validation_split_ratio);


    int eval_samples = std::min((int)X_val.size(), 5);
    if (eval_samples == 0 && !X_train.empty()) { // If no val set, show some train preds
        eval_samples = std::min((int)X_train.size(), 5);
        std::cout << "(No validation set, showing predictions on first few training samples)" << std::endl;
        for (int i = 0; i < eval_samples; ++i) {
            if (X_train[i].empty()) continue;
            std::vector<double> y_pred_vec_norm = nn.predict(X_train[i], false);
            if (y_pred_vec_norm.empty()) continue;
            double y_pred_denorm = denormalize_target_value_min_max(y_pred_vec_norm[0], target_scaling_info);
            std::cout << std::fixed << std::setprecision(5)
                      << std::setw(18) << original_train_targets[i] // using original unnormalized
                      << std::setw(20) << y_pred_denorm << std::endl;
        }
    } else {
        for (int i = 0; i < eval_samples; ++i) {
            if (X_val[i].empty()) continue;
            std::vector<double> y_pred_vec_norm = nn.predict(X_val[i], false);
            if (y_pred_vec_norm.empty()) continue;
            double y_pred_denorm = denormalize_target_value_min_max(y_pred_vec_norm[0], target_scaling_info);
             std::cout << std::fixed << std::setprecision(5)
                      << std::setw(18) << original_val_targets[i] // using original unnormalized
                      << std::setw(20) << y_pred_denorm << std::endl;
        }
    }
}


int main() {
    std::cout << "Initializing main..." << std::endl;

    std::cout << "\nRunning Training and Evaluation on CSV Data" << std::endl;
    run_training_and_evaluation();
    std::cout << "\nTraining and Evaluation on CSV Data Complete" << std::endl << std::endl;

    return 0;
}

