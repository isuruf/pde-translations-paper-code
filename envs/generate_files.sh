#!/bin/bash

for platform in linux-64 osx-64 osx-arm64 linux-aarch64 linux-ppc64le; do
  CONDA_SUBDIR=$platform conda env create -n paper-test-code --file envs/conda.yml --dry-run --json > file.json
  CONDA_SUBDIR=$platform python envs/generate_files.py
  rm file.json
done
