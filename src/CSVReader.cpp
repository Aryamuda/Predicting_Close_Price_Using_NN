// src/CsvReader.cpp
#include "CSVReader.h"
#include <iostream> // For potential error messages or debugging

namespace Predicting_Close_Price_Using_NN {

    // Helper function to trim whitespace from a string (useful for robust parsing)
    static std::string trim_whitespace(const std::string& str) {
        const std::string whitespace = " \t\n\r\f\v";
        size_t start = str.find_first_not_of(whitespace);
        if (start == std::string::npos) { // String is all whitespace
            return "";
        }
        size_t end = str.find_last_not_of(whitespace);
        return str.substr(start, (end - start + 1));
    }


    void CSVReader::read_regression_data(
        const std::string& filename,
        std::vector<std::vector<double>>& features,
        std::vector<double>& target_prices,
        int target_column_index,
        char delimiter) {

        std::ifstream file(filename);
        if (!file.is_open()) {
            throw std::runtime_error("CSVReader Error: Failed to open file: " + filename);
        }

        features.clear();
        target_prices.clear();

        std::string line;
        int line_number = 0;

        // Skip header row
        if (std::getline(file, line)) {
            line_number++;
        } else {
            throw std::runtime_error("CSVReader Error: File is empty or header missing in " + filename);
        }

        size_t expected_num_columns = 0;

        while (std::getline(file, line)) {
            line_number++;
            if (line.empty() || line.find_first_not_of(" \t\n\r\f\v") == std::string::npos) {
                // Skip empty or all-whitespace lines
                continue;
            }

            std::stringstream ss(line);
            std::string cell;
            std::vector<double> all_columns_in_row;

            while (std::getline(ss, cell, delimiter)) {
                try {
                    std::string trimmed_cell = trim_whitespace(cell);
                    if (!trimmed_cell.empty()) {
                        all_columns_in_row.push_back(std::stod(trimmed_cell));
                    } else {

                        all_columns_in_row.push_back(std::stod(trimmed_cell)); // This will throw for empty string
                    }
                } catch (const std::invalid_argument& ia) {
                    throw std::runtime_error("CSVReader Error: Invalid number format '" + cell +
                                             "' at line " + std::to_string(line_number) + " in file: " + filename);
                } catch (const std::out_of_range& oor) {
                     throw std::runtime_error("CSVReader Error: Number out of range '" + cell +
                                             "' at line " + std::to_string(line_number) + " in file: " + filename);
                }
            }

            if (all_columns_in_row.empty()) {
                std::cerr << "Warning: Skipped an effectively empty data row (after parsing delimiters) at line " << line_number << std::endl;
                continue;
            }

            if (expected_num_columns == 0) {
                expected_num_columns = all_columns_in_row.size();
                if (target_column_index < 0 || static_cast<size_t>(target_column_index) >= expected_num_columns) {
                    throw std::runtime_error("CSVReader Error: target_column_index (" + std::to_string(target_column_index) +
                                             ") is out of bounds for " + std::to_string(expected_num_columns) + " columns.");
                }
            } else if (all_columns_in_row.size() != expected_num_columns) {
                throw std::runtime_error("CSVReader Error: Inconsistent number of columns at line " + std::to_string(line_number) +
                                         ". Expected " + std::to_string(expected_num_columns) +
                                         ", got " + std::to_string(all_columns_in_row.size()) + ".");
            }

            std::vector<double> current_features_row;
            for (size_t i = 0; i < all_columns_in_row.size(); ++i) {
                if (static_cast<int>(i) == target_column_index) {
                    target_prices.push_back(all_columns_in_row[i]);
                } else {
                    current_features_row.push_back(all_columns_in_row[i]);
                }
            }
            features.push_back(current_features_row);
        }

        if (features.empty()) {
            // This could happen if the file only had a header, or only empty lines after header.
            throw std::runtime_error("CSVReader Error: No valid data rows read from file: " + filename);
        }
        if (features.size() != target_prices.size()) {
            // This should ideally not happen if logic is correct, but as a final check.
            throw std::runtime_error("CSVReader Error: Mismatch between number of feature rows (" + std::to_string(features.size()) +
                                     ") and target prices (" + std::to_string(target_prices.size()) + "). This indicates an internal logic error.");
        }

        file.close();
    }

} // namespace PricePredictorNN
