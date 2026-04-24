#!/bin/bash

cd /host >& /dev/null
if [[  $? -ne 0 ]]; then
	echo 'ERROR: Only execute this script from the docker container, using `bash gnv.sh`!'
	exit 1
fi

# Commands the user needs to execute once the container prompt is launched:
pip install -r requirements.txt -r G6K-GPU-Tensor/requirements.txt
pip install lattice-estimator/

# Note: if Eigen version 5.0 is released, then:
# 1. update the package name either in `Dockerfile.gnv` or,
# 2. modify cuBLASter/Makefile and run `make -C cuBLASter eigen3` (after some modification).
pip install cuBLASter/
