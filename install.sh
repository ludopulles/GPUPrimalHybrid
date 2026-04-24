#!/usr/bin/env bash
set -euo pipefail

# Usage: ./install.sh <conda_env_name>
# If no env name is provided, the script uses lwe_attack as name.

# Determine environment name
if [ $# -ge 1 ]; then
  ENV_NAME="$1"
else
  ENV_NAME="lwe_attack"
fi

echo "Creating distributable environment '${ENV_NAME}' from environment.yml..."
conda env create -f environment.yml -n "${ENV_NAME}"

# Activate the new environment
echo "Activating environment '${ENV_NAME}'..."
set +euo pipefail 
eval "$(conda shell.bash hook)"
conda activate "${ENV_NAME}"
set -euo pipefail

echo "Checking for CUDA Toolkit..."
if ! command -v nvcc &> /dev/null; then
    if [ -d "/usr/local/cuda/bin" ]; then
        export PATH="/usr/local/cuda/bin:$PATH"
    fi
fi
if ! command -v nvcc &> /dev/null; then
    echo "ERROR: nvcc (CUDA compiler) not found."
    echo "Please install the CUDA Toolkit or load the correct module (e.g., 'module load cuda')."
    exit 1
fi

# Install python package dependencies:
pip install -r requirements.txt -r G6K-GPU-Tensor/requirements.txt

pip install -e lattice-estimator

# Build G6K-GPU-Tensor
echo "Building G6K-GPU-Tensor..."
pushd G6K-GPU-Tensor >/dev/null
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

chmod +x rebuild.sh
./rebuild.sh -f -y
python setup.py install
popd >/dev/null

# Install cuBLASter in editable mode
echo "Installing cuBLASter in editable mode..."
pushd cuBLASter >/dev/null
make eigen3
pip install --no-build-isolation -e .
popd >/dev/null


echo "All done! Your environment is ready and repositories are cloned and installed."
echo "Don't forget to activate the '${ENV_NAME}' environment before running any code: conda activate ${ENV_NAME}"
