cmake_minimum_required(VERSION 3.31)
project(PricePredictorNN)

set(CMAKE_CXX_STANDARD 20)

add_executable(Predicting_Close_Price_Using_NN src/main.cpp
        include/Utils.h
        src/Utils.cpp
        include/Activations.h
        src/Activations.cpp
        include/Loss.h
        src/Loss.cpp
        include/Layer.h
        src/Layer.cpp
        include/NeuralNetwork.h
        src/NeuralNetwork.cpp
        include/CSVReader.h
        src/CSVReader.cpp
        include/BatchNormLayer.h
        src/BatchNormLayer.cpp
        include/BatchDataLoader.h
        src/BatchDataLoader.cpp)

# Specify  directories
include_directories(include)


# Confirm CMake configuration
message(STATUS "CMake configuration complete for PricePredictorNN.")
message(STATUS "C++ Standard: ${CMAKE_CXX_STANDARD}")

configure_file(
        "${CMAKE_SOURCE_DIR}/EURUSD.csv"
        "${CMAKE_CURRENT_BINARY_DIR}/EURUSD.csv"
        COPYONLY
)