#include <iostream>
#include <vector>
#include <string>
#include <cmath>
#include <iomanip>
#include <stdexcept>
#include <numeric>
#include <random>

#include "Loss.h"
#include "NeuralNetwork.h"

// --- Temporary Test Code / Helpers ---

template<typename T, typename U>
void assertTest(bool condition, const std::string& test_name, const T& expected, const U& actual) {
    std::cout << std::fixed << std::setprecision(5);
    if (!condition) {
        std::cerr << "ASSERTION FAILED: " << test_name
                  << " | Expected: " << expected
                  << " | Actual: " << actual << std::endl;
    } else {
        std::cout << "Assertion PASSED: " << test_name
                  << " | Expected: " << expected
                  << " | Actual: " << actual << std::endl;
    }
}


void assertTest(bool condition, const std::string& test_name) {
    if (!condition) {
        std::cerr << "ASSERTION FAILED: " << test_name << std::endl;
    } else {
        std::cout << "Assertion PASSED: " << test_name << std::endl;
    }
}

// Helper to print vectors for debugging
void print_vector_main(const std::string& name, const std::vector<double>& vec) {
    std::cout << name << ": [ ";
    for (size_t i = 0; i < vec.size(); ++i) {
        std::cout << std::fixed << std::setprecision(5) << vec[i] << (i == vec.size() - 1 ? "" : ", ");
    }
    std::cout << " ]" << std::endl;
}


// --- MVP: Train on a Toy Regression Problem ---
void train_on_toy_regression_problem() {
    std::cout << "\n--- Training NeuralNetwork on a Toy Regression Problem ---" << std::endl;

    // Toy Dataset: y = 2x + 1 (approximately, with some small variations or simple points)
    // Inputs (features) are single values.
    std::vector<std::vector<double>> X_train_toy = {
        {0.0}, {0.2}, {0.4}, {0.6}, {0.8}, {1.0}
    };
    // Corresponding target prices
    std::vector<double> y_train_toy = {
        1.0,   1.4,   1.8,   2.2,   2.6,   3.0
    };

    // Network Architecture: 1 input -> 5 hidden ReLU neurons -> 1 linear output neuron
    std::vector<int> layer_sizes = {1, 5, 1}; // Input:1, Hidden:5, Output:1
    std::vector<std::string> activations = {"relu", "linear"};
    double learning_rate = 0.05; // Might need tuning
    double dropout_rate = 0.0;   // No dropout for this simple test

    Predicting_Close_Price_Using_NN::NeuralNetwork nn(layer_sizes, activations, learning_rate, dropout_rate);

    int num_epochs = 2000; // Number of times to iterate over the entire dataset
    int print_every_n_epochs = 200;

    std::cout << "Starting training on toy data..." << std::endl;
    std::cout << "Network: 1 input -> 5 ReLU hidden -> 1 Linear output" << std::endl;
    std::cout << "Learning Rate: " << learning_rate << ", Epochs: " << num_epochs << std::endl;

    // For shuffling data each epoch
    std::vector<size_t> indices(X_train_toy.size());
    std::iota(indices.begin(), indices.end(), 0); // Fill with 0, 1, 2, ...
    std::random_device rd;
    std::mt19937 g(rd());


    for (int epoch = 0; epoch < num_epochs; ++epoch) {

        double current_epoch_total_loss = 0.0;

        for (size_t i = 0; i < X_train_toy.size(); ++i) {
            size_t current_idx = i; // Or indices[i] if shuffling
            const auto& x_sample = X_train_toy[current_idx];
            double y_true_sample = y_train_toy[current_idx];

            // Get prediction before training to calculate loss for this sample
            std::vector<double> y_pred_vec_before_train = nn.predict(x_sample, false); // training_mode=false for loss calc
            double y_pred_before_train = y_pred_vec_before_train[0];
            current_epoch_total_loss += Loss::mean_squared_error(y_true_sample, y_pred_before_train);

            // Train on this sample
            nn.train_one_sample(x_sample, y_true_sample);
        }

        double average_epoch_loss = current_epoch_total_loss / X_train_toy.size();

        if ((epoch + 1) % print_every_n_epochs == 0 || epoch == 0) {
            std::cout << "Epoch " << std::setw(4) << (epoch + 1) << "/" << num_epochs
                      << " | Average MSE Loss: " << std::fixed << std::setprecision(8) << average_epoch_loss
                      << std::endl;
        }
    }

    std::cout << "\nTraining on toy data complete." << std::endl;

    // Test predictions after training
    std::cout << "\nPredictions on toy data after training:" << std::endl;
    std::cout << std::setw(10) << "Input (x)" << std::setw(15) << "True (y)" << std::setw(18) << "Predicted (y_hat)" << std::setw(18) << "Abs Difference" << std::endl;
    std::cout << "------------------------------------------------------------------" << std::endl;
    for (size_t i = 0; i < X_train_toy.size(); ++i) {
        std::vector<double> y_pred_vec = nn.predict(X_train_toy[i], false); // training_mode=false for prediction
        double y_pred = y_pred_vec[0];
        double diff = std::abs(y_pred - y_train_toy[i]);
        std::cout << std::fixed << std::setprecision(5)
                  << std::setw(10) << X_train_toy[i][0]
                  << std::setw(15) << y_train_toy[i]
                  << std::setw(18) << y_pred
                  << std::setw(18) << diff
                  << std::endl;
    }
}
// --- End of MVP Test ---


int main() {
    std::cout << "Initializing main..." << std::endl;


    std::cout << "\nRunning MVP: Training on Toy Regression Problem" << std::endl;
    train_on_toy_regression_problem();
    std::cout << "MVP: Toy Regression Problem Training Completed" << std::endl << std::endl;
    // --- End of Temporary Test Calls ---



    return 0;
}

