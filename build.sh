#!/bin/bash

set -x
set -e

docker build -t filsilva/cxnetwork .
docker tag filsilva/cxnetwork filsilva/cxnetwork:0.1.0
docker push filsilva/cxnetwork
