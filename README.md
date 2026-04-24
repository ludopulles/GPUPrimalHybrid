# Guess + Verify implementation

This repository contains an implementation of the Guess + Verify attack against the Learning with Errors (LWE) with sparse secrets (binomial or ternary), using GPUs.

## Requirements and dependencies

Hardware:

- CUDA-compatible GPU(s), or cupy with ROCm for AMD GPUs (untested)

Software:

- CUDA toolkit, recommended: version >=11.0
- `conda`, `mamba`, or a similar environment manager (recommended is `conda`, see `install.sh`)
- (git submodule) `cuBLASter`: BLASter implementation adapted for GPUs
- (git submodule) `G6K-GPU-Tensor`: GPU-accelerated lattice sieving
- (git submodule) `lattice-estimator`: used to find good attack parameters based on the estimated cost of the attack

## Installation

You will need the CUDA toolkit installed and properly configured on your system to run this code. Make sure to have the appropriate drivers and CUDA version compatible with your GPU.
You can install it via the [official NVIDIA website](https://developer.nvidia.com/cuda-downloads).
If you consider installing a CUDA version lower than 12, please edit the `environment.yml` file to specify the compatible versions of `gcc` and `g++` (e.g., `gcc_linux-64<=11` and `gxx_linux-64<=11` for CUDA 11.x).

1. Run `git submodule update --init`
2. Install `conda` if you haven't already, see [conda docs](https://www.anaconda.com/docs/getting-started/miniconda/install).
3. Run the installation scripts: `./install.sh`
4. Install all required Python packages *in the environment*: `pip install -r requirements.txt`

## Main Components

- `attack.py`: Implements multi-GPU parallelization, distributing the workload across available GPUs and managing worker processes.
	Use this for actual attacks.
	See [Usage section](#usage) below for details.
- `attack_params.py`: Contains a list of parameter sets for different LWE instances to be attacked.
	We advise you to **uncomment just one** parameter set, when using the `attack.py` script (or its variant).
	One may optionally specify the attack parameters, see the commented lines for examples.

There are some variants/dialects of `attack.py` for testing purposes:

- `attack_testing.py`: Contains the main logic for the primal attack, including LWE instance creation, lattice embedding, reduction, SVP, and Babai's algorithm.
	It contains all functions for single-threaded execution (used for testing parameters, avoid using it for benchmarking).
	This code is more detailed than `attack.py` because of all the testing code.
- `attack_reduce_vram.py`: Attempts at reducing the VRAM usage on the GPU, by repeatedly recreating the `cupy` context. This may slightly slow down execution.

## Parameters

Set the numbers of workers and chunk size in `attack.py` according to your hardware capabilities.

- `num_workers`: Number of parallel workers (ideally a multiple of the number of GPUs and CPU cores, but may be limited by GPUs VRAM)
- `GUESS_BATCH`: Number of batches to process in each guess (affects GPUs memory usage and performance)

### Lattice Embedding

The system creates LWE instances in the file `lwe.py`.

- The binomial setting is generated using `generate_CBD_MLWE`
- The ternary setting is generated using `generate_ternary_MLWE`

The embedding and ternary LWE instance creation are taken from the [Cool+Cruel=Dual](
    https://gitlab.com/fvirdia/cool-plus-cruel-equals-dual
) repository. We also more generally adopted their file naming conventions, as this code is based on theirs. Many thanks to the authors.

### Result Handling

The results of the attack are collected and saved in a CSV file, including:

- `run_id`: Unique identifier for the run
- `n`: Dimension of the LWE instance
- `q`: Modulus
- `w`: Hamming weight of the secret
- `secret_type`: Type of secret distribution (ternary or binomial)
- `sigma`: Standard deviation for Gaussian errors (if applicable)
- `eta`: Parameter for binomial secrets (if applicable)
- `success`: Whether the attack was successful
- `iterations_used`: Number of performed iterations before the solution is found
- `time_elapsed`: Wall time taken for the attack
- `error`: Any error encountered during the attack (if applicable)

## Usage

To run the attack, execute the following command in your environment:

```bash
python attack.py
```

### Example usage

For example, to repeat the first "MLWE parameters" row of Table 2, perform these two steps:

1. In the file `attack_params.py`, make sure `atk_params` contains `    type_binomial(2, 2) | {'n': 256, 'q': 3329, 'w': 11},`.
2. Execute in your conda environment:

```bash
python attack.py -o binomial-q12-w11.csv -w2 -v |& tee binomial-q12-w11-details.txt
```

Note: this will also write all terminal output to the file `binomial-q12-w11-details.txt`.
Moreover, we use two workers, which may not be optimal in general.
If you have only one consumer GPU, use one worker. If you have many GPUs, increase the number of workers.

### Command line arguments

To see all command line arguments, execute `python attack.py -h` in your environment.
In particular, you can specify the number of workers (X) with `-wX`.

In `attack_params.py` you can add an instance by specifying the following parameters in a dictionary:

- `n`: Dimension
- `q`: Modulus
- `w`: Hamming weight of secret

In addition you need to specify `type_binomial(eta, k_dim)` for the binomial setting and `type_ternary(lwe_sigma)` for the ternary setting.
You can skip attack parameter selection, by adding the following parameters:

- `beta`: BKZ block size,
- `eta_svp`: only value 2 is supported currently, which corresponds to Babai NP as CVP-solver,
- `m`: number of LWE samples to pick from the structured MLWE(/RLWE) instance,
- `k`: number of positions in the secret to guess,
- `h_`: weight of the guessed part of the secret.
