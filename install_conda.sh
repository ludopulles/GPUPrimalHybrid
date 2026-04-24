#!/usr/bash

# Usage: ./install.sh [conda_env_name]
ENV_NAME="${1:-lwe_attack}"

if conda env list | grep -q "^$ENV_NAME"; then
    echo "Environment '$ENV_NAME' already exists. Skipping creation."
else
    echo "Creating conda environment '$ENV_NAME' from environment.yml..."
    conda env create -f environment.yml -n "$ENV_NAME"
fi

if [ "$CONDA_DEFAULT_ENV" != "$ENV_NAME" ]; then
	echo "Run 'conda activate $ENV_NAME', and execute 'bash conda-setup.sh' to finish installation."
else
	source conda-setup.sh
fi
