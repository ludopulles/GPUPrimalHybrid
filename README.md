# LWE Attack Implementation

This repository contains an implementation of a primal attack against the Learning with Errors (LWE) cryptographic primitive, using lattice reduction techniques and GPU acceleration.

Mainly usage of:

- cuBLASter: BLASter reduction adapted for GPUs
- G6K-GPU-Tensor: GPU-accelerated lattice reduction and SVP solvers
- lattice-estimator: for estimating the cost and the choice of parameters for the lattice attacks

## Overview

The attack implements a hybrid primal attack against LWE instances with different secret distributions (ternary and binomial). It leverages BKZ lattice reduction and supports multi-GPU parallelization for efficient computation.

## Prerequisites

- CUDA-compatible GPU(s) (or cupy with ROCm for AMD GPUs - not tested)
- Conda, mamba, or a similar environment manager (conda recommended for the use of install.sh)


## Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/plvie/LWE_attack.git
   cd LWE_attack
   ./install.sh
   pip install -r requirements.txt
   ```

## Main Components

attack.py: Contains the main logic for the primal attack, including LWE instance creation, lattice embedding, reduction, and Babai's algorithm. It contains all functions for single-threaded execution (used for testing parameters, avoid using it for benchmarking).

attack_multithread.py: Implements multi-GPU parallelization, distributing the workload across available GPUs and managing worker processes. Use this for actual attacks. (see Usage section below for details)

attack_params.py: Contains a list of parameter sets for different LWE instances to be attacked.

## Usage Details

Set the numbers of workers and chunk size in attack_multithread.py according to your hardware capabilities. 
- `num_workers`: Number of parallel workers (ideally a multiple of the number of GPUs and CPU cores, but may be limited by GPUs VRAM)
- `chunk_size`: Number of iterations per worker before checking for a solution (larger chunks reduce overhead but may lead to imbalanced workloads)
- `GUESS_BATCH`: Number of batches to process in each guess (affects GPUs memory usage and performance)

Adjust the parameters in attack_params.py to specify the LWE instances you want to attack.


### LWE Instance Creation

The system creates LWE instances based on parameters such as dimension, modulus, and error distribution.

```python
lwe = CreateLWEInstance(
    n,              # dimension
    q,              # modulus
    m,              # number of samples
    w,              # hamming weight of secret
    sigma,          # standard deviation of error (for Gaussian errors)
    type_of_secret,  # "ternary" or "binomial"
    eta=None,       # parameter for binomial distribution
    k_dim=None      # for module-LWE
)
```

### Lattice Embedding

For the primal attack, the system creates lattice embeddings using either:

- `BaiGalCenteredScaledTernary`: For ternary secrets with Gaussian errors
- `BaiGalModuleLWE`: For module-LWE with binomial secrets

### Lattice Reduction

The system applies progressive BKZ reduction to the lattice:

```python
reduced_basis, time_taken = reduction(
    basis,
    beta,              # BKZ block size
    eta,               # expected number of non-zero coordinates
    target,            # target vector
    target_estimation, # estimation of the target norm
    svp=False          # whether to embed the target vector into the basis during reduction
)
```

### GPU-Accelerated Babai's Algorithm

For more efficient solving, the system uses a GPU-accelerated implementation of Babai's nearest plane algorithm:

```python
result, time_taken = svp_babai_fp64_nr_projected(
    basis,
    eta,
    columns_to_keep,
    A,
    b_vec,
    tau,
    n,
    k,
    m,
    secret_possible_values,
    search_space_dim,
    target_estimation
)
```

See svp_babai_fp64_nr_projected in attack_multithread.py for more details on the parameters and how they are used.
See also kernel_babai.py for the implementation of the Babai's Nearest Plane algorithm on the GPU. (This is a GPU version base on BLASter batched Babai's Nearest Plane algorithm)

### Result Handling
The results of the attack are collected and saved in a CSV file, including:

- `run_id`: Unique identifier for the run
- `n`: Dimension of the LWE instance
- `q`: Modulus
- `w`: Hamming weight of the secret
- `secret_type`: Type of secret distribution (ternary or binomial)
- `sigma`: Standard deviation for Gaussian errors (if applicable)
- `eta`: Parameter for binomial secrets (if applicable)
- `available_cores`: Number of available CPU cores
- `success`: Whether the attack was successful
- `iterations_used`: Number of iterations taken to find a solution
- `time_elapsed`: Total time taken for the attack
- `estimated_time`: Estimated time based on average time per iteration and expected iterations (this is an estimate and may not reflect actual time taken, especially if the attack is unsuccessful, only used in attack.py, not in attack_multithread.py)
- `error`: Any error encountered during the attack (if applicable)

## Usage

To run the attack, execute the following command:

```bash
python attack_multithread.py
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
- Optionally: `beta`, `eta_svp`, `m`, `k`, `h_` to override automatic parameter selection
