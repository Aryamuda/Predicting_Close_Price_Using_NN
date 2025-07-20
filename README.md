# Predicting Close Price Using NN  

## Overview

Predicting Close Price Using NN is a C++ project that implements a neural network from scratch to predict financial asset prices based on time-series data. The current implementation is configured to read historical price data from a CSV file, train a neural network model, and evaluate its performance on a validation set.

## Features

- **Neural Network Core in C++:** A lightweight, customizable neural network implemented purely in C++.
- **Data Handling:** Includes a CSV reader to load and parse datasets.
- **Data Preprocessing:** Implements Min-Max normalization for features and target variables.
- **Time-Series Splitting:** Splits data into training and validation sets chronologically, which is crucial for time-series forecasting.
- **Configurable Architecture:** Easily configure the neural network's layers, activation functions, and hyperparameters.
- **Training & Evaluation:** The model is trained using a configurable number of epochs and evaluated using standard regression metrics (MSE, RMSE, MAE, R²).
- **Prediction Output:** Saves the actual vs. predicted values for the validation set into a CSV file for further analysis.

## How It Works

The main logic is orchestrated in the `run_main_training_pipeline` function:

1.  **Load Data:** Reads feature and target data from `EURUSD.csv`.
2.  **Normalize Data:** Applies Min-Max scaling to all features and the target variable to scale them into a [0, 1] range.
3.  **Split Data:** Performs a time-series split, allocating an initial portion of the data for training and the latter portion for validation (defaulting to a 80/20 split).
4.  **Initialize Network:** Constructs a `NeuralNetwork` object with a defined architecture (e.g., input layer, hidden layers with ReLU, output layer with linear activation) and hyperparameters (learning rate, batch size, etc.).
5.  **Train Model:** Trains the network on the normalized training data for a specified number of epochs.
6.  **Evaluate Performance:** Calculates and displays regression metrics for both the training and validation sets on the normalized data.
7.  **Save Predictions:** Generates predictions on the validation set, denormalizes them back to their original price scale, and saves them alongside the actual prices to `validation_predictions.csv`.

## Getting Started

### Prerequisites

- A C++ compiler that supports C++20.
- CMake (version 3.10 or higher).

### Building the Project

1.  Clone the repository:
    ```bash
    git clone https://github.com/Aryamuda/Predicting_Close_Price_Using_NN.git
    cd Predicting_Close_Price_Using_NN
    ```

2.  Create a build directory and run CMake:
    ```bash
    mkdir build
    cd build
    cmake ..
    ```

3.  Compile the project:
    ```bash
    cmake --build .
    ```

### Running the Application

1.  **Prepare Data:** Ensure you have a CSV file named `XAUUSD.csv` in the root directory of the project (or the execution directory). The file should contain numerical data where one column is the target variable (e.g., closing price) and the rest are features.

2.  Run the executable from the build directory:
    ```bash
    ./Predicting_Close_Price_Using_NN
    ```

## Configuration

The primary configuration is located within the `run_main_training_pipeline` function in `main.cpp`. You can adjust the following parameters:

-   **Dataset:** `input_csv_filename` and `target_column_idx`.
-   **Network Architecture:** `layer_sizes` and `activations`.
-   **Hyperparameters:** `learning_rate`, `num_epochs`, `batch_size`, `dropout_rate`, etc.
-   **Data Split:** `validation_split_ratio`.

## Output

-   **Console:** The program logs the entire process, including data loading, normalization, training progress, and final evaluation metrics.
-   **`validation_predictions.csv`:** A CSV file is generated with the following columns:
    -   `ActualPrice`: The true, original price from the validation set.
    -   `PredictedPrice_Denormalized`: The model's prediction, converted back to the original price scale.
    -   `NormalizedPrediction`: The raw, normalized output from the network.
    -   `NormalizedActual`: The normalized true value.
