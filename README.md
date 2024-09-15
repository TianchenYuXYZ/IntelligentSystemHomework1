# Logistic Regression Implementation with OpenMP

This repository contains a high-performance implementation of logistic regression, optimized using OpenMP for parallelization. The code is designed to handle large datasets efficiently by leveraging multi-threading. 

## Features

- **Randomized Input Generation**: 
  - The `generate_input` function creates synthetic datasets for training and testing. It uses seed values and parameters to generate reproducible random inputs.
  
- **Dynamic Batch Sizing**: 
  - The model dynamically adjusts the batch size based on the data size. This allows for optimized performance when working with larger datasets:
    - Batch size 32 for D ≥ 1000
    - Batch size 64 for D ≥ 500
    - Batch size 128 for D ≥ 100
    - Batch size 256 for smaller datasets

- **Parallelized Dot Product Calculation**: 
  - The logistic regression model uses OpenMP for parallel computation of the dot product during training, significantly speeding up the process when working with high-dimensional data.

- **Sigmoid Function Implementation**: 
  - The activation function is implemented as the logistic sigmoid to ensure proper probability outputs during logistic regression classification.

- **Custom Initialization of Parameters**: 
  - Parameters (weights) for the logistic regression model are initialized according to the dimensionality of the dataset, ensuring appropriate scaling and faster convergence.

## Usage

This implementation is designed for users who need an efficient logistic regression model for large datasets and who want to leverage parallel computing to improve performance.

1. **Install OpenMP**: Ensure that you have OpenMP installed and configured on your system to take advantage of the parallelization.
2. **Compile and Run**: Use the following commands to compile and run the project:
    ```bash
    g++ -fopenmp lr.cc -o lr
    ./lr
    ```

## Contributing

Feel free to submit issues or pull requests if you find any bugs or would like to improve the functionality of this logistic regression implementation.

## License

This project is licensed under the MIT License. See the `LICENSE` file for more details.
