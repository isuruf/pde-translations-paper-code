#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")/.."

MINIFORGE_HOME=${MINIFORGE_HOME:-${HOME}/.miniforge}
export CONDA_EXE="$MINIFORGE_HOME/bin/conda"

for platform in linux-64 osx-64 osx-arm64 linux-aarch64 linux-ppc64le; do
    for kind in compute plot; do
        "$MINIFORGE_HOME/bin/python" envs/generate_files.py "$platform" "$kind"
    done
done
