// include/CsvReader.h
#ifndef CSVREADER_HPP
#define CSVREADER_HPP

#include <vector>
#include <string>
#include <stdexcept>
#include <fstream>
#include <sstream>

namespace Predicting_Close_Price_Using_NN {

    class CSVReader {
    public:

        static void read_regression_data(
            const std::string& filename,
            std::vector<std::vector<double>>& features,
            std::vector<double>& target_prices,
            int target_column_index,
            char delimiter = ','
        );
    };

}

#endif //CSVREADER_HPP
