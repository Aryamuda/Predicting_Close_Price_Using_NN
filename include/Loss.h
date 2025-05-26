#ifndef LOSS_HPP
#define LOSS_HPP

#include <vector>
#include <string>

namespace Predicting_Close_Price_Using_NN {

    namespace Loss {

        // Calculates MSE for single pred
        // Formula: 0.5 * (y_pred - y_true)^2
        double mean_squared_error(double y_true, double y_pred);


        //Formula: y_pred - y_true (due to the 0.5 factor in the MSE formula)
        double mean_squared_error_derivative(double y_true, double y_pred);

    }

}

#endif