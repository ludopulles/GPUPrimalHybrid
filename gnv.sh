#!/bin/bash

DOCKERCMD=${DOCKER:-docker}
BASENAME=guess-verify

$DOCKERCMD container stop  ${BASENAME}-container
$DOCKERCMD container rm    ${BASENAME}-container
$DOCKERCMD rmi             ${BASENAME}-image
$DOCKERCMD build --platform=linux/amd64 -f Dockerfile.gnv -t ${BASENAME}-image .
$DOCKERCMD run -it \
  -v $(pwd):/host \
  --name ${BASENAME}-container ${BASENAME}-image
