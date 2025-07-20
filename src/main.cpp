#include <iostream>
#include <vector>
#include <string>
#include <cmath>
#include <iomanip>
#include <stdexcept>
#include <numeric>
#include <algorithm>
#include <limits>
#include <fstream>

// Project Includes
#include "Utils.h"
#include "Activations.h"
#include "Loss.h"
#include "Layer.h"
#include "NeuralNetwork.h"
#include "CSVReader.h"

// Helpers
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
void print_features_summary(const std::string& name, const std::vector<std::vector<double>>& features, int num_rows_to_print = 3, int num_cols_to_print = 5) { /* ... (as before) ... */
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

// --- Normalization Functions (from previous step) ---
struct MinMaxVals { /* ... (as before) ... */
    double min_val = std::numeric_limits<double>::max();
    double max_val = std::numeric_limits<double>::lowest();
};
std::vector<MinMaxVals> normalize_all_features_min_max(std::vector<std::vector<double>>& features) { /* ... (as before) ... */
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
MinMaxVals normalize_target_variable_min_max(std::vector<double>& target_variable) { /* ... (as before) ... */
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
double denormalize_target_value_min_max(double normalized_value, const MinMaxVals& scaling_params) { /* ... (as before) ... */
    double range = scaling_params.max_val - scaling_params.min_val;
    if (std::abs(range) < 1e-9) return scaling_params.min_val;
    return normalized_value * range + scaling_params.min_val;
}

// --- Time-Series Data Splitting Function ---
void time_series_split( const std::vector<std::vector<double>>& all_features, const std::vector<double>& all_targets, std::vector<std::vector<double>>& train_features, std::vector<double>& train_targets, std::vector<std::vector<double>>& val_features, std::vector<double>& val_targets, double validation_ratio = 0.2) { /* ... (as before) ... */
    if (all_features.size() != all_targets.size()) { throw std::runtime_error("TS split: F/T size mismatch."); }
    if (validation_ratio <= 0.0 || validation_ratio >= 1.0) { throw std::invalid_argument("TS split: Ratio out of bound."); }
    if (all_features.empty()) { train_features.clear(); train_targets.clear(); val_features.clear(); val_targets.clear(); return; }
    size_t total_samples = all_features.size();
    size_t validation_count = static_cast<size_t>(static_cast<double>(total_samples) * validation_ratio);
    size_t training_count = total_samples - validation_count;
    if (training_count == 0 || (validation_count > 0 && validation_count == total_samples) ) {
        if (total_samples > 10) { validation_count = std::max((size_t)1, static_cast<size_t>(total_samples * 0.1)); training_count = total_samples - validation_count;
            if (training_count == 0) { training_count = total_samples; validation_count = 0;}
        } else { training_count = total_samples; validation_count = 0; }
    }
    train_features.assign(all_features.begin(), all_features.begin() + training_count);
    train_targets.assign(all_targets.begin(), all_targets.begin() + training_count);
    if (validation_count > 0 && training_count > 0) {
        val_features.assign(all_features.begin() + training_count, all_features.end());
        val_targets.assign(all_targets.begin() + training_count, all_targets.end());
    } else { val_features.clear(); val_targets.clear(); }
    std::cout << "Time-series split: Total=" << total_samples << ", Train=" << train_features.size() << ", Validation=" << val_features.size() << std::endl;
}

// --- Main Training and Evaluation Orchestrator ---
void run_main_training_pipeline() {
    std::cout << "\n--- Neural Network Training Pipeline ---" << std::endl;

    std::string input_csv_filename = "EURUSD.csv";
    std::string output_val_predictions_csv = "validation_predictions.csv";

    std::vector<std::vector<double>> features_all_original;
    std::vector<double> target_prices_all_original;
    std::vector<std::vector<double>> features_all_norm;
    std::vector<double> target_prices_all_norm;
    int target_column_idx = 3;

    try {
        Predicting_Close_Price_Using_NN::CSVReader::read_regression_data(input_csv_filename, features_all_original, target_prices_all_original, target_column_idx);
        std::cout << "Successfully loaded " << features_all_original.size() << " original samples from " << input_csv_filename << "." << std::endl;
        if (features_all_original.empty()) { std::cerr << "No data loaded. Exiting." << std::endl; return; }
        features_all_norm = features_all_original;
        target_prices_all_norm = target_prices_all_original;
    } catch (const std::exception& e) {
        std::cerr << "Error loading CSV data: " << e.what() << std::endl; return;
    }

    std::cout << "\nNormalizing features and target variable..." << std::endl;
    std::vector<MinMaxVals> feature_scaling_info = normalize_all_features_min_max(features_all_norm); // Needed if you normalize features for prediction too
    MinMaxVals target_scaling_info = normalize_target_variable_min_max(target_prices_all_norm);

    std::cout << "Normalization complete." << std::endl;

    std::vector<std::vector<double>> X_train_norm, X_val_norm;
    std::vector<double> y_train_norm, y_val_norm;
    double validation_split_ratio = 0.2;
    time_series_split(features_all_norm, target_prices_all_norm, X_train_norm, y_train_norm, X_val_norm, y_val_norm, validation_split_ratio);

    std::vector<std::vector<double>> X_train_orig, X_val_orig;
    std::vector<double> y_train_orig, y_val_orig;
    time_series_split(features_all_original, target_prices_all_original, X_train_orig, y_train_orig, X_val_orig, y_val_orig, validation_split_ratio);

    if (X_train_norm.empty()) { std::cerr << "Normalized training set is empty after split. Exiting." << std::endl; return; }

    int input_dim = static_cast<int>(X_train_norm[0].size());
    std::vector<int> layer_sizes = {input_dim, 8, 4, 1};
    std::vector<std::string> activations = {"relu","relu", "linear"};

    double learning_rate = 0.01;
    double dropout_rate = 0.0;
    double momentum_coefficient = 0.9;
    double weight_decay_coefficient = 1e-3;
    bool use_batch_normalization = true;
    size_t batch_size = 64;

    std::cout << "\nInitializing Neural Network with:" << std::endl; /* ... print params ... */
    std::cout << "  Learning Rate: " << learning_rate << std::endl;
    std::cout << "  Batch Size: " << batch_size << std::endl;
    std::cout << "  Dropout Rate (for hidden layers): " << dropout_rate << std::endl;
    std::cout << "  Momentum Coefficient: " << momentum_coefficient << std::endl;
    std::cout << "  Weight Decay (L2) Coefficient: " << weight_decay_coefficient << std::endl;
    std::cout << "  Batch Normalization (for hidden layers): " << (use_batch_normalization ? "Enabled" : "Disabled") << std::endl;


    Predicting_Close_Price_Using_NN::NeuralNetwork nn(layer_sizes, activations, learning_rate, dropout_rate, momentum_coefficient, weight_decay_coefficient, use_batch_normalization);

    int num_epochs = 50;
    int print_every_n_epochs = 5;

    nn.train(X_train_norm, y_train_norm, num_epochs, batch_size, print_every_n_epochs, X_val_norm, y_val_norm);

    std::cout << "\n--- Final Evaluation Metrics (on Normalized Data) ---" << std::endl;
    Predicting_Close_Price_Using_NN::RegressionMetrics train_metrics_norm = nn.evaluate_regression(X_train_norm, y_train_norm);
    std::cout << "Final Training Metrics (Normalized):" << std::endl;
    std::cout << "  MSE:  " << std::fixed << std::setprecision(8) << train_metrics_norm.mse << std::endl;
    std::cout << "  RMSE: " << std::fixed << std::setprecision(8) << train_metrics_norm.rmse << std::endl;
    std::cout << "  MAE:  " << std::fixed << std::setprecision(8) << train_metrics_norm.mae << std::endl;
    std::cout << "  R2:   " << std::fixed << std::setprecision(8) << train_metrics_norm.r2 << std::endl;

    if (!X_val_norm.empty() && !y_val_norm.empty()) {
        Predicting_Close_Price_Using_NN::RegressionMetrics val_metrics_norm = nn.evaluate_regression(X_val_norm, y_val_norm);
        std::cout << "\nFinal Validation Metrics (Normalized):" << std::endl;
        std::cout << "  MSE:  " << std::fixed << std::setprecision(8) << val_metrics_norm.mse << std::endl;
        std::cout << "  RMSE: " << std::fixed << std::setprecision(8) << val_metrics_norm.rmse << std::endl;
        std::cout << "  MAE:  " << std::fixed << std::setprecision(8) << val_metrics_norm.mae << std::endl;
        std::cout << "  R2:   " << std::fixed << std::setprecision(8) << val_metrics_norm.r2 << std::endl;
    }

    // --- NEW: Save Validation Predictions to CSV ---
    if (!X_val_norm.empty() && !y_val_orig.empty()) {
        std::cout << "\nSaving validation predictions to " << output_val_predictions_csv << "..." << std::endl;
        std::ofstream outfile(output_val_predictions_csv);
        if (!outfile.is_open()) {
            std::cerr << "Error: Could not open file " << output_val_predictions_csv << " for writing!" << std::endl;
        } else {
            outfile << "ActualPrice,PredictedPrice_Denormalized,NormalizedPrediction,NormalizedActual\n"; // Header
            outfile << std::fixed << std::setprecision(5); // Set precision for file output

            for (size_t i = 0; i < X_val_norm.size(); ++i) {
                if (X_val_norm[i].empty()) continue;

                std::vector<double> y_pred_vec_norm = nn.predict(X_val_norm[i], false);
                if (y_pred_vec_norm.empty()) continue;

                double y_pred_norm = y_pred_vec_norm[0];
                double y_pred_denorm = denormalize_target_value_min_max(y_pred_norm, target_scaling_info);
                double y_true_original = y_val_orig[i];
                double y_true_norm = y_val_norm[i]; // This is the normalized actual value

                outfile << y_true_original << ","
                        << y_pred_denorm << ","
                        << y_pred_norm << ","
                        << y_true_norm << "\n";
            }
            outfile.close();
            std::cout << "Validation predictions saved successfully." << std::endl;
        }
    } else {
        std::cout << "\nNo validation data to save predictions for." << std::endl;
    }
    // --- END OF SAVE Validation Predictions ---


    std::cout << "\nPredictions on first few validation samples (denormalized values):" << std::endl; /* ... (as before, this part is for console display) ... */
    std::cout << std::setw(18) << "True (Original)" << std::setw(20) << "Predicted (DeNorm)" << std::setw(18) << "Abs Difference" << std::endl;
    std::cout << "------------------------------------------------------------------------------------" << std::endl;
    int eval_samples_count = 0;
    std::vector<std::vector<double>>* features_for_eval_norm = &X_val_norm;
    std::vector<double>* original_targets_for_eval = &y_val_orig;
    if (X_val_norm.empty() && !X_train_norm.empty()) {
        std::cout << "(No validation set, showing predictions on first few training samples)" << std::endl;
        features_for_eval_norm = &X_train_norm;
        original_targets_for_eval = &y_train_orig;
    }
    eval_samples_count = std::min((int)features_for_eval_norm->size(), 5);
    if (eval_samples_count > 0 && original_targets_for_eval->size() < static_cast<size_t>(eval_samples_count)) {
        eval_samples_count = std::min(eval_samples_count, (int)original_targets_for_eval->size());
    }
    for (int i = 0; i < eval_samples_count; ++i) {
        const auto& current_features_norm = (*features_for_eval_norm)[i];
        if (current_features_norm.empty()) continue;
        std::vector<double> y_pred_vec_norm = nn.predict(current_features_norm, false);
        if (y_pred_vec_norm.empty()) continue;
        double y_pred_denorm = denormalize_target_value_min_max(y_pred_vec_norm[0], target_scaling_info);
        double y_true_original = (*original_targets_for_eval)[i];
        double diff_denorm = std::abs(y_pred_denorm - y_true_original);
        std::cout << std::fixed << std::setprecision(5) << std::setw(18) << y_true_original << std::setw(20) << y_pred_denorm << std::setw(18) << diff_denorm << std::endl;
    }

    std::cout << "--- Main Training Pipeline Complete ---" << std::endl;
}


int main() {
    std::cout << "Initializing PricePredictorNN..." << std::endl;

    std::cout << "\n=============== Running Main Training Pipeline from main.cpp ===============" << std::endl;
    run_main_training_pipeline();
    std::cout << "=============== Main Training Pipeline Complete ===============" << std::endl << std::endl;

    std::cout << "PricePredictorNN application main logic continues here..." << std::endl;
    return 0;
}

