#!/bin/bash

export CUDA_VISIBLE_DEVICES=2,3

./tools/v100/ninfer-v100x2-serve.sh
