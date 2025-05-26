#include <iostream>
#include <vector>
#include <string>
#include <cmath>
#include <iomanip>
#include <stdexcept>
#include <algorithm>
#include <random>
#include "CSVReader.h"


template<typename T, typename U> // Use two template parameters for flexibility
void assertTest(bool condition, const std::string& test_name, const T& expected, const U& actual) {
    std::cout << std::fixed << std::setprecision(5); // Adjusted precision for float output
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

void print_features_summary(const std::string& name, const std::vector<std::vector<double>>& features, int num_rows_to_print = 3) {
    std::cout << name << " (Summary - first " << num_rows_to_print << " rows if available):" << std::endl;
    if (features.empty()) {
        std::cout << "  <No features loaded>" << std::endl;
        return;
    }
    std::cout << "  Total samples: " << features.size() << std::endl;
    if (!features.empty()) {
        std::cout << "  Features per sample: " << features[0].size() << std::endl;
    }
    for (int i = 0; i < std::min((int)features.size(), num_rows_to_print); ++i) {
        print_vector_main("  Sample " + std::to_string(i), features[i]);
    }
}


// --- Temporary Test for CSVReader ---
void temporary_test_csv_reader() {
    std::cout << "\n--- Running Temporary CSVReader Test ---" << std::endl;

    std::string filename = "XAUUSD.csv";
    std::vector<std::vector<double>> features;
    std::vector<double> target_prices;
    int target_column_idx = 4; //'Close' is the 4th column (0-indexed)

    std::cout << "Attempting to read data from: " << filename << std::endl;
    std::cout << "Target column index: " << target_column_idx << std::endl;

    try {
        Predicting_Close_Price_Using_NN::CSVReader::read_regression_data(filename, features, target_prices, target_column_idx);

        std::cout << "\nCSV Data Loaded Successfully:" << std::endl;
        std::cout << "Number of samples loaded: " << features.size() << std::endl;
        assertTest(features.size() == 5, "CSVReader: Correct number of samples loaded");
        assertTest(target_prices.size() == features.size(), "CSVReader: Features and targets count match");

        if (!features.empty()) {
            std::cout << "Number of features per sample (excluding target): " << features[0].size() << std::endl;
            // Original CSV has 5 columns. Target is 1 column. So features should be 5-1=4.
            assertTest(features[0].size() == 4, "CSVReader: Correct number of features per sample");
        }

        std::cout << "\nFirst few loaded features:" << std::endl;
        print_features_summary("Features", features, 3);

        std::cout << "\nFirst few loaded target prices:" << std::endl;
        std::vector<double> target_prices_subset;
        for(int i=0; i < std::min((int)target_prices.size(), 3); ++i) {
            target_prices_subset.push_back(target_prices[i]);
        }
        print_vector_main("Targets (subset)", target_prices_subset);

        // Specific checks based on the sample_price_data.csv
        if (features.size() >= 1 && !features[0].empty() && target_prices.size() >=1) { // Added !features[0].empty()
            if (features[0].size() >= 4) { // Check inner vector size before accessing
                assertTest(std::abs(features[0][0] - 1.0) < 1e-7, "CSVReader: features[0][0] value", 1.0, features[0][0]);
                assertTest(std::abs(features[0][1] - 2.0) < 1e-7, "CSVReader: features[0][1] value", 2.0, features[0][1]);
                assertTest(std::abs(features[0][2] - 3.0) < 1e-7, "CSVReader: features[0][2] value", 3.0, features[0][2]);
                assertTest(std::abs(features[0][3] - 0.1) < 1e-7, "CSVReader: features[0][3] value (was Feature4)", 0.1, features[0][3]);
            } else {
                 assertTest(false, "CSVReader: features[0] does not have enough columns for detailed check.");
            }
            assertTest(std::abs(target_prices[0] - 100.5) < 1e-7, "CSVReader: target_prices[0] value", 100.5, target_prices[0]);
        }
         if (features.size() >= 2 && !features[1].empty() && target_prices.size() >=2) { // Added !features[1].empty()
            // Second row features: 1.1, 2.1, 3.1, 0.2 (Target was 101.2)
            // No need to check features[1] values again if structure is okay
            assertTest(std::abs(target_prices[1] - 101.2) < 1e-7, "CSVReader: target_prices[1] value", 101.2, target_prices[1]);
        }


    } catch (const std::exception& e) {
        std::cerr << "CSVReader test failed with exception: " << e.what() << std::endl;
        assertTest(false, "CSVReader: Data loading without exceptions");
    }

}
// --- End of CSVReader Test ---


int main() {
    std::cout << "Initializing main..." << std::endl;

    std::cout << "\nRunning Temporary CSVReader Tests" << std::endl;
    temporary_test_csv_reader();
    std::cout << "CSVReader Tests Complete" << std::endl << std::endl;
    // --- End of Temporary Test Calls ---


    return 0;
}

