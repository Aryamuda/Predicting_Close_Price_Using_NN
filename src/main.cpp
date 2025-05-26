#include <iostream>
#include <vector>
#include <string>
#include <cmath>
#include <iomanip>
#include <stdexcept>
#include <algorithm>
#include <random>

#include "Loss.h"
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



// --- Train on Loaded CSV Data ---
void train_on_csv_data() {
    std::cout << "\n--- Training NeuralNetwork on Loaded CSV Data ---" << std::endl;

    std::string filename = "XAUUSD.csv"; // <--- YOUR XAUUSD FILENAME HERE
    std::vector<std::vector<double>> features_all;
    std::vector<double> target_prices_all;
    int target_column_idx = 3; // <--- YOUR TARGET COLUMN INDEX HERE

    try {
        Predicting_Close_Price_Using_NN::CSVReader::read_regression_data(filename, features_all, target_prices_all, target_column_idx);
        std::cout << "Successfully loaded " << features_all.size() << " samples from " << filename << "." << std::endl;
        if (features_all.empty()) {
            std::cerr << "No data loaded. Exiting training." << std::endl;
            return;
        }
        std::cout << "Number of features per sample: " << features_all[0].size() << std::endl;
        print_vector_main("First few target prices", target_prices_all, 5);

    } catch (const std::exception& e) {
        std::cerr << "Error loading CSV data: " << e.what() << std::endl;
        return;
    }

    // For now, let's use a subset of data if it's too large, to keep training quick for this test
    // This is NOT a proper train/test split.
    size_t num_samples_to_use = features_all.size(); // Use all for now, or std::min((size_t)1000, features_all.size());

    std::vector<std::vector<double>> X_train(features_all.begin(), features_all.begin() + num_samples_to_use);
    std::vector<double> y_train(target_prices_all.begin(), target_prices_all.begin() + num_samples_to_use);

    if (X_train.empty()) {
        std::cerr << "Training set is empty after potential subsetting. Exiting." << std::endl;
        return;
    }

    // Network Architecture
    // Input dimension must match the number of features from the CSV
    int input_dim = static_cast<int>(X_train[0].size());
    // Example architecture: input_dim -> hidden1_size -> hidden2_size -> 1 (output price)
    std::vector<int> layer_sizes = {input_dim, 64, 32, 1}; // Example, tune as needed
    std::vector<std::string> activations = {"relu", "relu", "linear"};
    double learning_rate = 0.001;
    double dropout_rate = 0.0;    // Start without dropout

    Predicting_Close_Price_Using_NN::NeuralNetwork nn(layer_sizes, activations, learning_rate, dropout_rate);

    int num_epochs = 100; // Small number of epochs for this test
    int print_every_n_epochs = 10;

    std::cout << "\nStarting training on " << X_train.size() << " CSV samples..." << std::endl;
    std::cout << "Network Architecture: ";
    for(size_t i=0; i<layer_sizes.size(); ++i) std::cout << layer_sizes[i] << (i == layer_sizes.size()-1 ? "" : " -> ");
    std::cout << " | Activations: ";
    for(size_t i=0; i<activations.size(); ++i) std::cout << activations[i] << (i == activations.size()-1 ? "" : ", ");
    std::cout << std::endl;
    std::cout << "Learning Rate: " << learning_rate << ", Epochs: " << num_epochs << std::endl;


    for (int epoch = 0; epoch < num_epochs; ++epoch) {
        double current_epoch_total_mse_loss = 0.0;
        int processed_samples = 0;

        // Iterate through the training data sequentially (important for time series)
        for (size_t i = 0; i < X_train.size(); ++i) {
            const auto& x_sample = X_train[i];
            double y_true_sample = y_train[i];

            // It's good practice to check if x_sample is valid before passing
            if (x_sample.empty()) {
                std::cerr << "Warning: Empty feature vector at sample index " << i << ". Skipping." << std::endl;
                continue;
            }
            if (static_cast<int>(x_sample.size()) != input_dim) {
                 std::cerr << "Warning: Feature vector at sample index " << i << " has incorrect size. Expected " << input_dim << ", got " << x_sample.size() << ". Skipping." << std::endl;
                 continue;
            }


            // Get prediction before training to calculate loss for this sample
            std::vector<double> y_pred_vec_before_train = nn.predict(x_sample, false);
            if (y_pred_vec_before_train.empty()) {
                 std::cerr << "Warning: Prediction vector is empty for sample index " << i << ". Skipping loss calculation." << std::endl;
                 continue;
            }
            double y_pred_before_train = y_pred_vec_before_train[0];
            current_epoch_total_mse_loss += Predicting_Close_Price_Using_NN::Loss::mean_squared_error(y_true_sample, y_pred_before_train);
            processed_samples++;

            // Train on this sample
            nn.train_one_sample(x_sample, y_true_sample);
        }

        if (processed_samples == 0) {
            std::cerr << "Epoch " << (epoch + 1) << ": No samples were processed. Check data." << std::endl;
            continue;
        }

        double average_epoch_loss = current_epoch_total_mse_loss / processed_samples;

        if ((epoch + 1) % print_every_n_epochs == 0 || epoch == 0 || epoch == num_epochs - 1) {
            std::cout << "Epoch " << std::setw(4) << (epoch + 1) << "/" << num_epochs
                      << " | Average MSE Loss: " << std::fixed << std::setprecision(8) << average_epoch_loss
                      << std::endl;
        }
    }

    std::cout << "\nTraining on CSV data complete." << std::endl;

    // "Evaluate" on the first few training samples (not a proper evaluation)
    std::cout << "\nPredictions on first few training samples after training:" << std::endl;
    std::cout << std::setw(15) << "True Target" << std::setw(18) << "Predicted Target" << std::setw(18) << "Abs Difference" << std::endl;
    std::cout << "------------------------------------------------------------------" << std::endl;
    int eval_samples = std::min((int)X_train.size(), 5);
    double total_eval_mse = 0.0;
    for (int i = 0; i < eval_samples; ++i) {
        if (X_train[i].empty()) continue;
        std::vector<double> y_pred_vec = nn.predict(X_train[i], false);
        if (y_pred_vec.empty()) continue;

        double y_pred = y_pred_vec[0];
        double diff = std::abs(y_pred - y_train[i]);
        total_eval_mse += Predicting_Close_Price_Using_NN::Loss::mean_squared_error(y_train[i], y_pred);
        std::cout << std::fixed << std::setprecision(5)
                  << std::setw(15) << y_train[i]
                  << std::setw(18) << y_pred
                  << std::setw(18) << diff
                  << std::endl;
    }
    if (eval_samples > 0) {
        std::cout << "Average MSE on these " << eval_samples << " samples: " << total_eval_mse / eval_samples << std::endl;
    }

}
// --- End of CSV Data Training Test ---


int main() {
    std::cout << "Initializing main..." << std::endl;

    std::cout << "\nRunning Training on CSV Data" << std::endl;
    train_on_csv_data();
    std::cout << "\nTraining on CSV Data Complete" << std::endl << std::endl;
    // --- End of Temporary Test Calls ---


    std::cout << "PricePredictorNN application main logic continues here..." << std::endl;

    return 0;
}

