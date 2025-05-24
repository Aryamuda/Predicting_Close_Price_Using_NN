#ifndef ACTIVATIONS_HPP
#define ACTIVATIONS_HPP


namespace Activations {

    double relu(double x);

    double relu_derivative(double x);

    double sigmoid(double x);

    double sigmoid_derivative(double activated_output);

    double linear(double x);

    double linear_derivative(double x); // Parameter x is kept for consistent signature, though not used.

}
#endif
