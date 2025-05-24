cmake_minimum_required(VERSION 3.31)
project(PricePredictorNN)

set(CMAKE_CXX_STANDARD 20)

add_executable(Predicting_Close_Price_Using_NN src/main.cpp
        include/Utils.h
        src/Utils.cpp
        include/Activations.h
        src/Activations.cpp
        include/Loss.h
        src/Loss.cpp)

# Specify  directories
include_directories(include)


# Confirm CMake configuration
message(STATUS "CMake configuration complete for PricePredictorNN.")
message(STATUS "C++ Standard: ${CMAKE_CXX_STANDARD}")
