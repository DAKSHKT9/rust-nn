# Neural Network in Rust with XOR Gate Example

This project is a simple implementation of a neural network in Rust, designed to refresh my Rust skills and explore neural network concepts. The network has been implemented as an XOR gate to demonstrate basic functionality.

## Overview

The primary focus of this project is to provide a foundational neural network structure while preparing for more advanced work, such as enabling GPU-powered machine learning computations using CUDA. The long-term goal is to bring GPU-enabled ML compute to the Internet Computer Protocol (ICP) blockchain.

### Key Features

- **Neural Network Basics**: A simple feedforward neural network that performs binary classification.
- **XOR Gate Example**: Trained to simulate an XOR gate, a classic problem that showcases the power of neural networks for non-linear classification.
- **Rust Implementation**: Fully written in Rust, leveraging the language's performance and safety benefits.

## Next Steps

The immediate next step for this project is to integrate **CUDA** for GPU acceleration, enabling more complex and faster computations. The end goal is to bring GPU-enabled ML computation to the ICP chain.

## Project Structure

The main logic is located in the `main.rs` file, which contains the neural network implementation and the XOR gate demonstration.

### Files

- `src/lib/matrix.rs`: Contains Matrix class and code for all matrix operations
- `src/lib/networks.rs`: Contains Network class and code for backward and forward propogation.
- `src/main.rs`: Implementation of XOR gate example.
  
## Running the Project

To run the XOR gate example and see the neural network in action, follow these steps:

1. Clone the repository:
   ```bash
   git clone https://github.com/DAKSHKT9/rust-nn.git

2. Go to derectory:
   ```bash
   cd rust-nn

3. Run using cargo:
   ```bash
   cargo run





