#!/usr/bash

if [ -z "${CONDA_DEFAULT_ENV:x}" ]; then
	echo "Make sure you are in a conda environment, using 'conda activate X'"
	exit 1
fi

if ! command -v nvcc &> /dev/null; then
    echo "ERROR: nvcc (CUDA compiler) not found."
    echo "Please install the CUDA toolkit: conda install cuda-toolkit."
    exit 1
fi

# Install python package dependencies:
pip install -r requirements.txt -r G6K-GPU-Tensor/requirements.txt

# Install lattice-estimator
pip install ./lattice-estimator

# Install cuBLASter
pip install ./cuBLASter


# Build G6K-GPU-Tensor
echo "Building G6K-GPU-Tensor..."
pushd G6K-GPU-Tensor >/dev/null

# THIS IS A DIRTY HACK IF PATHS DON'T WORK OUT:
REAL_NVCC=$(readlink -f $(which nvcc))
REAL_CUDA_DIR=$(dirname $(dirname "$REAL_NVCC"))
REAL_CXX=$(which g++)
REAL_CC=$(which gcc)
REAL_PYTHON=$(which python)

echo "Patching all hardcoded paths (CUDA, GCC, Python)"
find . -type f \( -name "Makefile" -o -name "*.py" -o -name "*.sh" -o -name "*.mk" \) -exec sed -E -i "s|/usr/local/cuda[-0-9\.]*|$REAL_CUDA_DIR|g" {} +
find . -type f \( -name "Makefile" -o -name "*.mk" -o -name "*.sh" \) -exec sed -i "s|/usr/bin/g++|$REAL_CXX|g" {} +
find . -type f \( -name "Makefile" -o -name "*.mk" -o -name "*.sh" \) -exec sed -i "s|/usr/bin/gcc|$REAL_CC|g" {} +
find . -type f \( -name "Makefile" -o -name "*.sh" \) -exec sed -i "s|python3 |$REAL_PYTHON |g" {} +
find . -type f \( -name "Makefile" -o -name "*.sh" \) -exec sed -i "s|/usr/bin/python3|$REAL_PYTHON|g" {} +

export CUDA_PATH="$REAL_CUDA_DIR"
export CUDA_HOME="$REAL_CUDA_DIR"
export CC="$REAL_CC"
export CXX="$REAL_CXX"

./rebuild.sh -f -y
python setup.py install
popd >/dev/null

echo ""
echo "Done. Don't forget to run all your experiments in conda environment '${CONDA_DEFAULT_ENV}'!"
