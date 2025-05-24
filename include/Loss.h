#ifndef LOSS_HPP
#define LOSS_HPP


namespace Loss {

    // MSE for single prediction:
    // y_true target value
    // y_pred predicted value
    double mean_squared_error(double y_true, double y_pred);

    // Calculates the derivative of the MSE loss with respect to the predicted value (y_pred):
    double mean_squared_error_derivative(double y_true, double y_pred);

}

#endif
