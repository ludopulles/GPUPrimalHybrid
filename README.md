# LWE Attack Implementation

This repository contains an implementation of a primal attack against the Learning with Errors (LWE) with sparse secrets (binomial or ternary), using GPU acceleration.

Mainly usage of:

- cuBLASter: BLASter implementation adapted for GPUs
- G6K-GPU-Tensor: GPU-accelerated lattice reduction and SVP solvers
- lattice-estimator: for estimating the cost and the choice of parameters for the lattice attacks

## Overview

The attack implements a hybrid primal attack against LWE instances with different sparse secret distributions (ternary and binomial). It supports multi-GPU parallelization for efficient computation.

## Prerequisites

- CUDA-compatible GPU(s) (or cupy with ROCm for AMD GPUs - not tested)
- CUDA toolkit installed (version 11.0 or higher recommended)
- Conda, mamba, or a similar environment manager (conda recommended for the use of install.sh)

## Installation

1. Install conda if you haven't already. You can install it from [here](https://www.anaconda.com/docs/getting-started/miniconda/install).

1. Clone the repository:

   ```bash
   git clone https://github.com/plvie/LWE_attack.git
   cd LWE_attack
   ./install.sh
   pip install -r requirements.txt
   ```

## Main Components

``attack_testing.py``: Contains the main logic for the primal attack, including LWE instance creation, lattice embedding, reduction, SVP, and Babai's algorithm. It contains all functions for single-threaded execution (used for testing parameters, avoid using it for benchmarking).

``attack_for_benchmark.py``: Implements multi-GPU parallelization, distributing the workload across available GPUs and managing worker processes. Use this for actual attacks. (see Usage section below for details)
His code is also more clean and easier to read than attack_testing.py because it does not contain all the testing code.

``attack_params.py``: Contains a list of parameter sets for different LWE instances to be attacked. You can modify this file to add or change the parameters for your specific use cases.

## Usage Details

Set the numbers of workers and chunk size in attack_for_benchmark.py according to your hardware capabilities.

- `num_workers`: Number of parallel workers (ideally a multiple of the number of GPUs and CPU cores, but may be limited by GPUs VRAM)
- `chunk_size`: Number of iterations per worker before checking for a solution (larger chunks reduce overhead it just takes more time to estimate the time remaining)
- `GUESS_BATCH`: Number of batches to process in each guess (affects GPUs memory usage and performance)

### Lattice Embedding

For the primal attack, the system creates lattice embeddings using either:

- `BaiGalCenteredScaledTernary`: For ternary secrets with Gaussian errors
- `BaiGalModuleLWE`: For module-LWE with binomial secrets (and binomial errors)

The embedding and ternary LWE instance creation are taken from the [Cool+Cruel=Dual](
    https://gitlab.com/fvirdia/cool-plus-cruel-equals-dual
) repository. We also more generally adopted their file naming conventions, as this code is based on theirs. Many thanks to the authors.

### GPU-Accelerated Babai's Algorithm

For more efficient solving, the system uses a GPU-accelerated implementation of Babai's nearest plane algorithm.
See ``svp_babai_fp64_nr_projected`` in ``attack_for_benchmark.py`` for more details on the parameters and how they are used.
See also ``kernel_babai.py`` for the implementation of the Babai's Nearest Plane algorithm on the GPU. (This is a GPU version base on BLASter batched Babai's Nearest Plane algorithm)

### Result Handling

The results of the attack are collected and saved in a CSV file, including:

- `run_id`: Unique identifier for the run
- `n`: Dimension of the LWE instance
- `q`: Modulus
- `w`: Hamming weight of the secret
- `secret_type`: Type of secret distribution (ternary or binomial)
- `sigma`: Standard deviation for Gaussian errors (if applicable)
- `eta`: Parameter for binomial secrets (if applicable)
- `available_cores`: Number of available CPU cores (for reference)
- `success`: Whether the attack was successful
- `iterations_used`: Number of iterations taken to find a solution
- `time_elapsed`: Total time taken for the attack
- `estimated_time`: Estimated time based on average time per iteration and expected iterations (this is an estimate and may not reflect actual time taken, especially if the attack is unsuccessful, only used in attack_testing.py)
- `error`: Any error encountered during the attack (if applicable)

## Usage

To run the attack, execute the following command:

```bash
python attack_for_benchmark.py
```

And optionally specify the number of workers and chunk size in the script.

The `atk_params` should contain a list of parameter dictionaries, each specifying:

- `n`: Dimension
- `q`: Modulus
- `w`: Hamming weight of secret
- `secret_type`: "ternary" or "binomial"
- `lwe_sigma`: For Gaussian errors (ternary secrets)
- `eta`: For binomial secrets
- `k_dim`: For module-LWE
- Optionally: `beta`, `eta_svp`, `m`, `k`, `h_` to override automatic parameter selection.
