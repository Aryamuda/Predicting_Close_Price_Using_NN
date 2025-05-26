// src/Loss.cpp
#include "Loss.h"
#include <cmath>
#include <numeric>

namespace Predicting_Close_Price_Using_NN {

    namespace Loss {

        double mean_squared_error(double y_true, double y_pred) {
            double error = y_pred - y_true;
            return 0.5 * error * error; // 0.5 * (y_pred - y_true)^2
        }

        double mean_squared_error_derivative(double y_true, double y_pred) {
            return y_pred - y_true;
        }

    }

}
