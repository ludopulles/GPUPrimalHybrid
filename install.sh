#!/usr/bin/env bash
set -euo pipefail

# Usage: ./setup_env.sh <conda_env_name>
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
eval "$(conda shell.bash hook)"
conda activate "${ENV_NAME}"

# Array of repositories to clone
REPOS=(
  "https://github.com/plvie/G6K-GPU-Tensor.git"
  "https://github.com/plvie/lattice-estimator.git"
  "https://github.com/plvie/BLASter.git"
)

echo "Cloning repositories..."
for REPO in "${REPOS[@]}"; do
  git clone "$REPO"
done

# Install lattice-estimator and BLASter in editable mode
echo "Installing lattice-estimator in editable mode..."
pushd lattice-estimator >/dev/null
pip install -e .
popd >/dev/null

echo "Installing BLASter in editable mode..."
pushd BLASter >/dev/null
pip install -e .
popd >/dev/null

# Build G6K-GPU-Tensor
echo "Building G6K-GPU-Tensor..."
pushd G6K-GPU-Tensor >/dev/null
./rebuild.sh -f -y
popd >/dev/null

echo "All done! Your environment is ready and repositories are cloned and installed."
